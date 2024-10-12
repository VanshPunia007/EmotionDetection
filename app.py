# app.py

from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import uvicorn
import os

# Initialize the FastAPI app
app = FastAPI()

# Load the saved model
model = load_model('emotion_detection_model.h5')

# Define the emotion classes (you can replace these with your actual labels)
class_names = ["anger", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

# Function to preprocess the image
def preprocess_image(file):
    img = image.load_img(file, target_size=(48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(file.file.read())

    # Preprocess the image
    img = preprocess_image(file_location)

    # Predict the class
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    predicted_label = class_names[predicted_class]

    # Remove the temporary image file
    os.remove(file_location)

    return {"predicted_class": predicted_label}
