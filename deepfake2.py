import streamlit as st 
import pickle 
import cv2 
import numpy as np 
import tensorflow
import requests 
 
image_width = 224 
image_height = 224 
 
def classify_image(image_path_or_url):
    try:
        # Check if the provided input is a URL
        if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
            # Download the image from the URL
            response = requests.get(image_path_or_url)
            # Ensure the request was successful
            if response.status_code == 200:
                # Convert image content to numpy array
                image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                # Read the image using OpenCV
                user_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                # Check if the image is valid
                if user_image is None:
                    print("Error: Unable to decode image")
                    return None
            else:
                print("Error: Unable to download image from URL")
                return None
        else:
            # If not a URL, assume it's a local file path
            image_path = image_path_or_url
            # Read the image using OpenCV
            user_image = cv2.imread(image_path)
            # Check if the image is valid
            if user_image is None:
                print("Error: Unable to load image at path:", image_path)
                return None
       
        # Resize image to the required dimensions (224x224)
        user_image = cv2.resize(user_image, (224, 224))
        # Normalize pixel values to be between 0 and 1
        user_image = user_image.astype('float32') / 255.0
        user_image = np.expand_dims(user_image, axis=0)
        # Pass the preprocessed image through the trained model
        predicted_label = model.predict(user_image)
        # Convert predicted probability to class label
        if predicted_label >= 0.5:
            return "Fake"
        else:
            return "Real"
    except Exception as e:
        print("Error:", e)
        return None
     
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
st.title("DEEP_FAKE_DETECT") 
image_path = st.text_input("Paste an URL...") 
submitted = st.button("Submit") 
if submitted and image_path: 
    predicted_label = classify_image(image_path) 
    st.header(predicted_label)
