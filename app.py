import streamlit as st
import cv2
import io
import numpy as np
from ultralytics import YOLO
from weights import weights
#from google.colab.patches import cv2_imshow
from roboflow import Roboflow

weights()
# Load YOLO model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)
model.fuse()

# infer on a local image
#print(model.predict("your_image.jpg", confidence=40, overlap=30).json())
def apply_clahe(image):
    # Apply CLAHE to each channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    processed_image = cv2.merge([clahe.apply(channel) for channel in cv2.split(image)])

    return processed_image

# Function to display boxes with numbers
def display_boxes_with_numbers(image, results):
    # Sort boxes based on the x-coordinate of the top-left corner
    sorted_boxes = sorted(results.boxes, key=lambda box: box.xyxy[0][0].item())

    # Iterate over each detected box
    for idx, box in enumerate(sorted_boxes):
        class_id, cords, conf = results.names[box.cls[0].item()], [round(x) for x in box.xyxy[0].tolist()], round(box.conf[0].item(), 2)

        # Draw bounding box
        # cv2.rectangle(image, (cords[0], cords[1]), (cords[2], cords[3]), (0, 255, 0), 2)

        # Draw number overlay
        font = cv2.FONT_HERSHEY_COMPLEX
        font_scale = 0.7
        font_thickness = 2
        font_color = (255, 0, 0)
        text = str(idx + 1)

        # Calculate the position for the text
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_position = ((cords[0] + cords[2]) // 2 - text_size[0] // 2, (cords[1] + cords[3]) // 2 + text_size[1] // 2)

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

        # Apply CLAHE to the image
        processed_image = apply_clahe(image)

        # Display the original and processed images
        st.image([image, processed_image], channels="BGR", caption=["Original Image", "Processed Image"], use_column_width=True)

        # Make predictions on the processed image
        results_list = model.predict(processed_image)
        results = results_list[0]

        # Display boxes with numbers on the processed image
        display_boxes_with_numbers(processed_image.copy(), results)

if __name__ == "__main__":
    main()
