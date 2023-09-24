from flask import Flask, render_template, Response
import cv2
import numpy as np
from pydub import AudioSegment
from pydub.playback import play

app = Flask(__name__, static_folder="static")

@app.route('/')
def index():
    return render_template('index.html')

def detect_violence():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
    cap = cv2.VideoCapture(0)
    violence_detected = False
    frames_without_violence = 0
    frames_to_reset = 30
    alarm_sound = AudioSegment.from_file("alarm_sound.mp3", format="mp3")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for (x, y, w, h) in bodies:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if len(faces) > 0 and len(bodies) > 0:
            violence_detected = True
            frames_without_violence = 0
        else:
            frames_without_violence += 1

        if violence_detected:
            play(alarm_sound)

        if frames_without_violence >= frames_to_reset:
            violence_detected = False

        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(detect_violence(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
