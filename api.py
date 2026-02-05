"""
AI-Generated Voice Detection API
=================================
REST API for detecting AI-generated vs Human voices across 5 languages:
Tamil, English, Hindi, Malayalam, Telugu

Endpoint: POST /api/voice-detection
Authentication: x-api-key header
Input: Base64-encoded MP3 audio
Output: JSON with classification result
"""

import os
import base64
import tempfile
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from ai_voice_detector import AIVoiceDetector, AudioConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# API Key for authentication (Change this to your secret key)
API_KEY = os.environ.get("API_KEY", "sk_test_123456789")

# Supported languages
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# ============================================================================
# API MODELS
# ============================================================================

class VoiceDetectionRequest(BaseModel):
    """Request model for voice detection"""
    language: str = Field(..., description="Language of the audio: Tamil, English, Hindi, Malayalam, or Telugu")
    audioFormat: str = Field(..., description="Audio format, must be 'mp3'")
    audioBase64: str = Field(..., description="Base64-encoded MP3 audio data")


class VoiceDetectionResponse(BaseModel):
    """Success response model"""
    status: str = "success"
    language: str
    classification: str  # AI_GENERATED or HUMAN
    confidenceScore: float  # 0.0 to 1.0
    explanation: str


class ErrorResponse(BaseModel):
    """Error response model"""
    status: str = "error"
    message: str


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="AI Voice Detection API",
    description="Detects whether a voice sample is AI-generated or Human across Tamil, English, Hindi, Malayalam, and Telugu",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the detector globally (loads model once)
detector = None


def get_detector():
    """Get or initialize the detector"""
    global detector
    if detector is None:
        logger.info("Initializing AI Voice Detector...")
        try:
            config = AudioConfig(
                max_duration_sec=30.0,
                ai_threshold=0.55
            )
            detector = AIVoiceDetector(config)
            logger.info("AI Voice Detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            raise
    return detector


@app.on_event("startup")
async def startup_event():
    """Initialize detector on startup"""
    logger.info("Starting AI Voice Detection API...")
    try:
        get_detector()
        logger.info("API ready to accept requests")
    except Exception as e:
        logger.warning(f"Detector initialization delayed: {e}")
        logger.info("API started - detector will initialize on first request")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "AI Voice Detection API",
        "version": "1.0.0",
        "supported_languages": SUPPORTED_LANGUAGES,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/voice-detection", response_model=VoiceDetectionResponse)
async def detect_voice(
    request: VoiceDetectionRequest,
    x_api_key: Optional[str] = Header(None, alias="x-api-key")
):
    """
    AI Voice Detection Endpoint
    
    Analyzes a Base64-encoded MP3 audio file and determines if the voice
    is AI-generated or Human.
    
    - **language**: One of Tamil, English, Hindi, Malayalam, Telugu
    - **audioFormat**: Must be "mp3"
    - **audioBase64**: Base64-encoded MP3 audio data
    
    Returns classification result with confidence score and explanation.
    """
    
    # ========================================================================
    # 1. VALIDATE API KEY
    # ========================================================================
    if not x_api_key:
        logger.warning("Request rejected: Missing API key")
        raise HTTPException(
            status_code=401,
            detail={"status": "error", "message": "Missing API key. Include 'x-api-key' header."}
        )
    
    if x_api_key != API_KEY:
        logger.warning(f"Request rejected: Invalid API key")
        raise HTTPException(
            status_code=401,
            detail={"status": "error", "message": "Invalid API key or malformed request"}
        )
    
    # ========================================================================
    # 2. VALIDATE LANGUAGE
    # ========================================================================
    # Normalize language (capitalize first letter)
    language = request.language.strip().capitalize()
    
    if language not in SUPPORTED_LANGUAGES:
        logger.warning(f"Request rejected: Unsupported language '{request.language}'")
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": f"Unsupported language. Must be one of: {', '.join(SUPPORTED_LANGUAGES)}"
            }
        )
    
    # ========================================================================
    # 3. VALIDATE AUDIO FORMAT
    # ========================================================================
    if request.audioFormat.lower() != "mp3":
        logger.warning(f"Request rejected: Invalid audio format '{request.audioFormat}'")
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Invalid audio format. Only 'mp3' is supported."}
        )
    
    # ========================================================================
    # 4. DECODE BASE64 AUDIO
    # ========================================================================
    try:
        audio_bytes = base64.b64decode(request.audioBase64)
        logger.info(f"Decoded audio: {len(audio_bytes)} bytes")
    except Exception as e:
        logger.error(f"Failed to decode Base64 audio: {e}")
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Invalid Base64 encoding in audioBase64 field"}
        )
    
    # Validate audio data is not empty
    if len(audio_bytes) < 100:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Audio data is too small or empty"}
        )
    
    # ========================================================================
    # 5. SAVE TEMPORARY FILE AND ANALYZE
    # ========================================================================
    temp_path = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(audio_bytes)
            logger.info(f"Saved temporary audio file: {temp_path}")
        
        # Get detector and analyze
        voice_detector = get_detector()
        result = voice_detector.detect(temp_path)
        
        logger.info(f"Detection complete: {result['label']} (score: {result['final_score']:.4f})")
        
    except FileNotFoundError as e:
        logger.error(f"Audio file error: {e}")
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Failed to process audio file"}
        )
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": f"Internal error during voice analysis: {str(e)}"}
        )
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.debug(f"Cleaned up temporary file: {temp_path}")
            except Exception:
                pass
    
    # ========================================================================
    # 6. GENERATE RESPONSE
    # ========================================================================
    classification = result['label']  # AI_GENERATED or HUMAN
    confidence_score = round(result['final_score'], 2)
    
    # Generate explanation based on individual scores
    explanation = generate_explanation(result, classification)
    
    logger.info(f"Response: {classification} | Confidence: {confidence_score} | Language: {language}")
    
    return VoiceDetectionResponse(
        status="success",
        language=language,
        classification=classification,
        confidenceScore=confidence_score,
        explanation=explanation
    )


def generate_explanation(result: dict, classification: str) -> str:
    """
    Generate a human-readable explanation for the detection result.
    
    Analyzes individual test scores to provide meaningful feedback.
    """
    scores = result.get('individual_scores', {})
    
    # Collect significant indicators
    ai_indicators = []
    human_indicators = []
    
    # Analyze each score
    score_analysis = {
        'pitch': ('pitch stability', 'Unnatural pitch consistency', 'Natural pitch variation'),
        'spectral': ('spectral smoothness', 'Over-smoothed frequency spectrum', 'Natural spectral texture'),
        'phase': ('phase artifacts', 'Phase discontinuities detected', 'Natural phase coherence'),
        'noise': ('noise patterns', 'Synthetic noise patterns', 'Natural background noise'),
        'ml': ('deep learning', 'AI-like embedding patterns', 'Human-like speech patterns'),
        'prosody_fp': ('prosody', 'Robotic speech rhythm', 'Natural speech prosody'),
        'micro_artifacts': ('micro-artifacts', 'Neural synthesis glitches', 'No synthesis artifacts'),
    }
    
    for key, (name, ai_text, human_text) in score_analysis.items():
        score = scores.get(key)
        if score is not None:
            if score > 0.6:
                ai_indicators.append(ai_text)
            elif score < 0.4:
                human_indicators.append(human_text)
    
    # Build explanation
    if classification == "AI_GENERATED":
        if ai_indicators:
            explanation = f"{ai_indicators[0]}"
            if len(ai_indicators) > 1:
                explanation += f" and {ai_indicators[1].lower()}"
            explanation += " detected"
        else:
            explanation = "Voice characteristics suggest AI generation"
    else:
        if human_indicators:
            explanation = f"{human_indicators[0]}"
            if len(human_indicators) > 1:
                explanation += f" with {human_indicators[1].lower()}"
        else:
            explanation = "Voice characteristics are consistent with human speech"
    
    return explanation


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom error handler for HTTP exceptions"""
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"status": "error", "message": str(exc.detail)})


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"status": "error", "message": f"Internal server error: {str(exc)}"})


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Get port from environment variable (for cloud deployment)
    port = int(os.environ.get("PORT", 8000))
    
    print("\n" + "="*60)
    print("AI VOICE DETECTION API")
    print("="*60)
    print(f"\nAPI Key: {API_KEY[:10]}..." if len(API_KEY) > 10 else f"\nAPI Key: {API_KEY}")
    print(f"Supported Languages: {', '.join(SUPPORTED_LANGUAGES)}")
    print(f"\nPort: {port}")
    print("\nEndpoint: POST /api/voice-detection")
    print("\nStarting server...")
    print("="*60 + "\n")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
