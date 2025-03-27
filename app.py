import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response
from keras.preprocessing import image
from twilio.rest import Client
import pyttsx3
import os

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# Twilio API Credentials (replace with your own)
account_sid = "AC431f8830e8afed4a2d71f0b4e3c29d48"
auth_token = "fd08589312b3ce2a34aaec9ab2d031b0"
twilio_phone_number = "+15414226980"
recipient_phone_number = "+919392551869"
whatsapp_sender = "whatsapp:+14155238886"
whatsapp_recipient = "whatsapp:+919392551869"

# Initialize the text-to-speech engine
tts_engine = pyttsx3.init()

# Function to preprocess frame for prediction
def preprocess_frame(frame, target_size=(224, 224)):
    frame_resized = cv2.resize(frame, target_size)
    frame_array = image.img_to_array(frame_resized)
    frame_array = np.expand_dims(frame_array, axis=0)
    frame_array /= 255.0
    return frame_array

# Function to predict contamination
def predict_contamination(frame):
    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame)
    return "Contaminated" if prediction[0][0] > 0.5 else "Good Food"

# Function to send SMS alert
def send_sms_alert(message):
    client = Client(account_sid, auth_token)
    client.messages.create(body=message, from_=twilio_phone_number, to=recipient_phone_number)

# Function to send WhatsApp alert
def send_whatsapp_alert(message):
    client = Client(account_sid, auth_token)
    client.messages.create(body=message, from_=whatsapp_sender, to=whatsapp_recipient)

# Function for voice alert
def voice_alert(message):
    tts_engine.say(message)
    tts_engine.runAndWait()

# Real-time webcam stream and detection
def gen_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predict contamination
        prediction = predict_contamination(frame)

        # Draw AR-like overlay (red rectangle and text)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if prediction == "Contaminated":
            cv2.rectangle(frame, (50, 50), (400, 150), (0, 0, 255), 2)  # Red rectangle for contamination
            cv2.putText(frame, "⚠️ Contaminated Food!", (60, 100), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Send alerts
            message = "⚠️ Alert: Contaminated food detected!"
            send_sms_alert(message)
            send_whatsapp_alert(message)
            voice_alert(message)

        # Convert the frame for streaming
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ret, jpeg = cv2.imencode('.jpg', frame_rgb)
        if not ret:
            continue

        # Yield frame for Flask stream
        frame_data = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')

    cap.release()

# Route to display the webcam feed
@app.route('/')
def index():
    return render_template('index.html')

# Route to stream the webcam feed to the browser
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
