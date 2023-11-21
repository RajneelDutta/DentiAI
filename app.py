import streamlit as st
import cv2
import io
import numpy as np
from ultralytics import YOLO
from roboflow import Roboflow

# weights()
# # Load YOLO model
# MODEL_PATH = "best_v8.pt"
# model = YOLO(MODEL_PATH)
# model.fuse()

st.write("API_KEY:", st.secrets["api_key"])
rf = Roboflow(api_key="API_KEY")
project = rf.workspace().project("teethdetection-rjah4")
model = project.version(9).model

# Function to display boxes with numbers
def display_boxes_with_numbers(image, results):
   
    # Extract "x" and "y" coordinates
    coordinates_list = []
 
    for prediction in results["predictions"]:
        x_coordinate = prediction["x"]
        y_coordinate = prediction["y"]
        coordinates_list.append({"x": x_coordinate, "y": y_coordinate})
    # Reorder the list based on decreasing order of "x" values
    coordinates_list = sorted(coordinates_list, key=lambda x: x["x"], reverse=True)
 
    # Number the x-y pairs
    numbered_coordinates_list = [{"number": idx + 1, "x": coord["x"], "y": coord["y"]} for idx, coord in enumerate(coordinates_list)]
    print(numbered_coordinates_list)
        #
    for coordinates in numbered_coordinates_list:
       
        x, y = int(coordinates["x"]), int(coordinates["y"])
        font = cv2.FONT_HERSHEY_COMPLEX
        font_scale = 0.7
        font_thickness = 2
        font_color = (255, 255, 255)
        text = str(coordinates["number"])
       
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_position = (x,y)
       
        cv2.putText(image, text, text_position, font, font_scale, font_color, font_thickness)
    # Display the image with bounding boxes and numbers
    st.image(image, channels="BGR")
 
# Streamlit app
def main():
    st.title("Dental AI App")
 
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
 
    if uploaded_file is not None:
        # Convert the uploaded file to a bytes-like object
        image_bytes = io.BytesIO(uploaded_file.read())
 
        # Use cv2.imdecode to read the image from bytes
        image = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), -1)
       
        st.image(image, channels="BGR", caption="Uploaded Image.", use_column_width=True)
 
        # Make predictions
        results = model.predict(image, confidence=40, overlap=30).json()
        print(results)
        #results_list
       
        #results = results_list[0]  
       
        # Display boxes with numbers
        display_boxes_with_numbers(image, results)
 
if __name__ == "__main__":
    main()
