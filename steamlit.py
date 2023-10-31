import streamlit as st
from PIL import Image
import numpy as np
import cv2
from skimage import exposure
from skimage.morphology import binary_opening, binary_closing, binary_erosion, binary_dilation

# Function to perform histogram equalization
def perform_histogram_equalization(image):
    img_array = exposure.equalize_hist(image)
    return img_array

# Function to perform image filtering
def perform_image_filtering(image):
    filtered_image = cv2.GaussianBlur(image, (5, 5), 0)
    return filtered_image

# Function to perform morphological operations
def perform_morphological_operations(image, operation):
    kernel = np.ones((4, 4), np.uint8)
    if operation == "Opening":
        processed_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
    elif operation == "Closing":
        processed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
    elif operation == "Erosion":
        processed_image = cv2.erode(image, kernel, iterations=1)
    elif operation == "Dilation":
        processed_image = cv2.dilate(image, kernel, iterations=1)
    return processed_image

# Sidebar navigation
rad = st.radio("Choose a Filter", ["Histogram Equalization", "Image Filtering", "Morphological Operations"])

# Image upload
uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    img_array = np.array(img)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    threshold_value = 0.3 # You may adjust this threshold value
    binary_image = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)

    st.title("Image Filters")
    st.image(uploaded_image, caption='Uploaded Image')

    if rad == "Histogram Equalization":
        st.header("Histogram Equalization")
        processed_image = perform_histogram_equalization(img_array)
        st.image(processed_image, caption='Histogram Equalization', use_column_width=True)

    elif rad == "Image Filtering":
        st.header("Image Filtering")
        processed_image = perform_image_filtering(img_array)
        st.image(processed_image, caption='Image Filtering', use_column_width=True)

    elif rad == "Morphological Operations":
        st.header("Morphological Operations")
        binary_image = binary_image[1]
        st.image(binary_image, caption='Binary Image', use_column_width=True)
        operations = ["Opening", "Closing", "Erosion", "Dilation"]
        selected_operation = st.radio("Select Morphological Operation", operations)
        processed_image = perform_morphological_operations(binary_image, selected_operation)
        st.image(processed_image, caption=f'Morphological Image - {selected_operation}', use_column_width=True)
