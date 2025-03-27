import cv2
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import streamlit as st
from twilio.rest import Client
import pyttsx3
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Initialize Twilio for SMS/WhatsApp alerts
account_sid = "AC431f8830e8afed4a2d71f0b4e3c29d48"  # Replace with your Twilio Account SID
auth_token = "fd08589312b3ce2a34aaec9ab2d031b0"     # Replace with your Twilio Auth Token
twilio_phone_number = "+15414226980"     # Twilio phone number
recipient_phone_number = "+919392551869"  # Replace with your phone number
whatsapp_sender = "whatsapp:+14155238886"
whatsapp_recipient = "whatsapp:+919392551869"

# Initialize the text-to-speech engine for voice alerts
tts_engine = pyttsx3.init()

# Function to preprocess video frames
def preprocess_frame(frame, target_size=(224, 224)):
    frame_resized = cv2.resize(frame, target_size)
    frame_array = image.img_to_array(frame_resized)
    frame_array = np.expand_dims(frame_array, axis=0)  # Add batch dimension
    frame_array /= 255.0  # Normalize pixel values to [0, 1]
    return frame_array

# Function to predict contamination
def predict_contamination(frame):
    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame)
    return "Contaminated" if prediction[0][0] < 0.5 else "Good Food"

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

# Real-time video capture and detection
def run_live_detection():
    cap = cv2.VideoCapture(0)  # Start capturing video from the webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()  # Read each frame from the webcam
        if not ret:
            print("Error: Could not read frame.")
            break

        # Predict contamination for the current frame
        prediction = predict_contamination(frame)

        # Display the prediction on the video feed
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Prediction: {prediction}", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if prediction == "Contaminated":
            # Send alerts if contamination is detected
            message = "⚠️ Alert: Contaminated food detected!"
            send_sms_alert(message)
            send_whatsapp_alert(message)
            voice_alert(message)

        # Convert frame from BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame using Streamlit
        st.image(frame_rgb, channels="RGB", caption="Live Contamination Detection")

        # Exit when 'q' is pressed in Streamlit
        if st.button('Stop Detection'):
            break

    # Release the video capture and close windows
    cap.release()

# Streamlit App Layout
def main():
    st.title("Real-Time Food Contamination Detection")
    st.write("This app uses a live webcam feed to detect food contamination and send alerts.")
    
    if st.button('Start Detection'):
        run_live_detection()

if __name__ == "__main__":
    main()
