import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

def rotate_image(image, degrees):
    return image.rotate(degrees, expand=True)

st.title('YOLO Детектор тритонов')
st.write("Загрузите изображение")

@st.cache_resource
def load_model():
    return YOLO('my_model.pt')

model = load_model()

uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    image = rotate_image(image, -90)
    
    st.image(image, caption="Текущее изображение")
    
    if st.button("Запустить детекцию"):
        with st.spinner("Обработка..."):
            try:
                # Конвертация в numpy array для YOLO
                img_np = np.array(image)
                
                results = model(img_np)
                
                for r in results:
                    im_bgr = r.plot()
                    im_rgb = Image.fromarray(im_bgr[..., ::-1])
                    st.image(im_rgb, caption="Результаты детекции", use_column_width=True)
                        
            except Exception as e:
                st.error(f"Ошибка: {str(e)}")
