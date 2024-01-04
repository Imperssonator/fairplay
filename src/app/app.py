import cv2
import numpy as np
import streamlit as st
from fairplay.cv.extractor import Extractor

uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg"])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    cv2_img = cv2.imdecode(file_bytes, 1)
    et = Extractor(cv2_img)

    # Show the image
    st.image(et.img, channels="BGR")

    # Find axes
    et = et.find_axes()

    st.image(et.imlabel, channels="BGR")
