import streamlit as st
from PIL import Image, ImageOps
from img_classification import teachable_machine_classification

st.title("Breast Cancer Detection R&D D_Kumar IITP")
st.header("Ultrasound Cancer Image Upload")
st.text("Cancer Type Classification ")


uploaded_file = st.file_uploader("Choose a scan ...", type="png")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Scan.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = teachable_machine_classification(image, 'model/keras_model.h5')
    if label == 0:
        st.write("The scan is normal")
    elif label == 1:
        st.write("The scan is malignant")
    else:
        st.write("The scan is benign")
