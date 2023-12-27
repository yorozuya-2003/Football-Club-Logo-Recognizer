# module imports
import json
import numpy as np

import cv2
from PIL import Image
from rembg.bg import remove

import torch
from torchvision import transforms
import tensorflow as tf
from keras.utils import img_to_array

import streamlit as st
from assets.frontend_config \
    import set_bg_image, custom_title, custom_image_caption, custom_success_message


# loading the model
loaded_model = tf.saved_model.load('model')


# loading class mappings
with open('assets/class_mapping.json', 'r') as file:
    classes_reverse_dict = {int(k):v for k,v in json.load(file).items()}


# classification pipeline
def classify(test_image):
    '''
    This function takes an image as input and returns the predicted class.
    '''
    IMG_SIZE = 64

    transform = transforms.Compose([transforms.ToTensor()])

   # removing image background
    test_image = remove(test_image)

    # making the transparent background black
    black_background_image = Image.new('RGB', test_image.size, (0,0,0))
    black_background_image.paste(test_image, mask=test_image)

    # resizing the image
    test_image = img_to_array(black_background_image)
    resized_test_image = cv2.resize(test_image, (IMG_SIZE, IMG_SIZE))
    final_test_image = transform(resized_test_image)

    dim1, dim2, dim3 = final_test_image.shape
    final_test_image = final_test_image.reshape(1, dim1, dim2, dim3) 

    final_test_image = final_test_image.to(torch.float32)
    final_test_image = final_test_image.numpy()
    final_test_image = tf.transpose(final_test_image, [0, 2, 3, 1])

    prediction = loaded_model(final_test_image)[0]
    result = classes_reverse_dict[np.argmax(prediction)]

    return result


# streamlit app
def main():
    # streamlit user interface
    st.set_page_config(page_title="Football Club Logo Image Classifier", page_icon=":soccer:")

    with open( "ui/style.css" ) as css:
        st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

    st.markdown("""
        <style>
            [data-testid=stMarkdownContainer] {
                font-family: "Montserrat", sans-serif !important;
            }
        </style>
    """, unsafe_allow_html=True)

    custom_title(st, "Football Club Logo Image Classifier")


    st.sidebar.header("**Instructions**")
    st.sidebar.markdown(
        "- Upload a logo image for classification.\n"
        "- Click on the **Identify!** button.\n"
        "- The model will analyze the image and identify the football club.\n"
        "- Enjoy the process!"
    )

    set_bg_image('ui/background.jpg', 'jpg')

    # image uploading and classification
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    c1, c2 = st.columns(2)
    if uploaded_file is not None:
        c1.image(uploaded_file, caption="", use_column_width=True)
        custom_image_caption(c1, "Uploaded Image")

        if c2.button("Identify!"):
            image = Image.open(uploaded_file)
            prediction_result = classify(image)
            custom_success_message(c2, f"{prediction_result.upper()}")


if __name__ == "__main__":
    main()
