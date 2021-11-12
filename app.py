import streamlit as st
import numpy as np
import cv2
import keras
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

characters="零一二三四五六七八九十百千万亿"

model = keras.models.load_model('model.h5')

st.title('Chinese MNIST digit recogniser')
st.markdown('Try one of these digits ---> 零 一 二 三 四 五 六 七 八 九 十 百 千 万 亿')

canvas_result = st_canvas(
    fill_color="rgb(0,0,0)",
    stroke_width=4,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=200,
    height=200
    drawing_mode="freedraw",
    display_toolbar=True,
    key='canvas',
)

if st.button('Predict'):
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (64, 64))
    alpha = img[:,:,0]/256
    resized = alpha.reshape(1,64,64,1)
    st.write(f'result: {characters[model.predict(resized)[0].argmax(axis=-1)]}')