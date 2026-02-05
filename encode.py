import base64

with open("C:\\Users\\nithi\\Downloads\\Voice_Detection\\Audio from Navin Kumar J.mp3", "rb") as audio_file:
    encoded_string = base64.b64encode(audio_file.read())
    print(encoded_string.decode('utf-8'))