import tkinter as tk
from tkinter import Label, Button, messagebox
import cv2
from PIL import Image, ImageTk
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
from twilio.rest import Client
import pyttsx3

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Twilio setup for SMS/WhatsApp alerts
account_sid = "AC431f8830e8afed4a2d71f0b4e3c29d48"  # Replace with your Twilio Account SID
auth_token = "fd08589312b3ce2a34aaec9ab2d031b0"     # Replace with your Twilio Auth Token
twilio_phone_number = "+15414226980"     # Twilio phone number
recipient_phone_number = "+919392551869" # Your phone number
whatsapp_sender = "whatsapp:+14155238886"
whatsapp_recipient = "whatsapp:+919392551869"

# Text-to-Speech engine for voice alerts
tts_engine = pyttsx3.init()

# Global variables for GUI and webcam
cap = None
running = False

# Function to preprocess the frame for prediction
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_array = image.img_to_array(frame_resized)
    frame_array = np.expand_dims(frame_array, axis=0)
    frame_array /= 255.0  # Normalize
    return frame_array

# Function to predict contamination
def predict_contamination(frame):
    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame)
    return "Good Food" if prediction[0][0] > 0.5 else "Contaminated"

# Function to send alerts via Twilio
def send_alerts(alert_message):
    # SMS Alert
    client = Client(account_sid, auth_token)
    client.messages.create(
        body=alert_message,
        from_=twilio_phone_number,
        to=recipient_phone_number
    )
    # WhatsApp Alert
    client.messages.create(
        body=alert_message,
        from_=whatsapp_sender,
        to=whatsapp_recipient
    )
    # Voice Alert
    tts_engine.say(alert_message)
    tts_engine.runAndWait()

# Function to start the webcam and display the feed
def start_camera():
    global cap, running
    if running:
        messagebox.showinfo("Info", "Camera is already running!")
        return

    cap = cv2.VideoCapture(0)
    running = True
    update_frame()

# Function to stop the webcam
def stop_camera():
    global cap, running
    running = False
    if cap:
        cap.release()
        cap = None
    lbl_video.imgtk = None
    lbl_video.configure(image=None)

# Function to capture frames and predict contamination
def update_frame():
    global cap, running
    if not running:
        return

    ret, frame = cap.read(1)
    if not ret:
        messagebox.showerror("Error", "Unable to access camera!")
        return

    # Convert BGR to RGB for Tkinter
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_image = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=frame_image)
    lbl_video.imgtk = imgtk
    lbl_video.configure(image=imgtk)

    # Predict contamination
    prediction = predict_contamination(frame)
    lbl_prediction.config(text=f"Prediction: {prediction}", fg="green" if prediction == "Good Food" else "red")

    # Send alerts if contaminated
    #if prediction == "Contaminated":
        #send_alerts("Alert: Contaminated food detected!")

    # Schedule the next frame update
    lbl_video.after(100, update_frame)

# Function to exit the application
def exit_app():
    stop_camera()
    root.destroy()

# Initialize Tkinter GUI
root = tk.Tk()
root.title("KitchenEye")
root.geometry("800x600")

# GUI Widgets
lbl_title = Label(root, text="KitchenEye", font=("Arial", 24))
lbl_title.pack(pady=10)

lbl_video = Label(root)
lbl_video.pack(pady=10)

lbl_prediction = Label(root, text="Prediction: None", font=("Arial", 18))
lbl_prediction.pack(pady=10)

btn_start = Button(root, text="Start Camera", command=start_camera, font=("Arial", 14), bg="green", fg="white")
btn_start.pack(side="left", padx=20)

btn_stop = Button(root, text="Stop Camera", command=stop_camera, font=("Arial", 14), bg="red", fg="white")
btn_stop.pack(side="left", padx=20)

btn_exit = Button(root, text="Exit", command=exit_app, font=("Arial", 14), bg="gray", fg="white")
btn_exit.pack(side="left", padx=20)

# Start the Tkinter event loop
root.mainloop()
