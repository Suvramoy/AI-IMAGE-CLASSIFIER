import numpy as np
import streamlit as st
import cv2
from PIL import Image
from keras.models import load_model

model = load_model('model (2).h5')

# CSS for setting the background image
page_bg_css = r"""
<style>
    .stApp {
        background-image: url("https://freestock.blog/wp-content/uploads/2022/08/robotspaint_artificial_intelligence_65512256-0d3e-433f-a796-e6aa3837dc8e.jpeg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stApp > div > div:first-child {
        background: rgba(255, 255, 255, 0.7);  /* Add a translucent background for better readability */
        border-radius: 10px;
        padding: 20px;
    }
</style>
"""

def Prediction(image):
    image = Image.open(image)
    image = image.convert("RGB")  # Ensure image has 3 color channels (RGB)
    
    img = np.array(image)
    test_img = cv2.resize(img, (64, 64))  # Resize image to match the model input
    test_input = test_img.reshape((1, 64, 64, 3))  # Reshape for the model
    predict = model.predict(test_input) > 0.5  # Make prediction

    prob = model.predict(test_input)

    if predict:
        return "Real Image", prob
    else:
        return "AI Generated Image", 1 - prob

# Main function for the Streamlit app
def main():
    st.markdown(page_bg_css, unsafe_allow_html=True)

    st.title("AI-IMAGE Classifier")

    img_file = st.file_uploader("Upload a JPG Image", type=["jpg", "jpeg", "png"])

    result = ''
    prob = None

    if img_file is not None:
        image = Image.open(img_file)
        image = image.resize((200, 200))
        st.image(image, caption='Uploaded Image', use_column_width=False)

    if st.button("Check Result"):
        if img_file is not None:
            result, prob = Prediction(img_file)
            # Format the probability to display with two decimal places
            prob = f"{prob[0][0]*100:.2f}%"
        else:
            st.text("Image Not Found")
        
    col1,col2 = st.columns(2)
    col1.markdown("**Type**")
    col1.write(result)
    col2.markdown("**Probability**")
    col2.write(prob)

if __name__ == '__main__':
    main()
