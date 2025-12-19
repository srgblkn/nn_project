import streamlit as st
from PIL import Image
import torch
from model.model import MyResNet
from model.preprocessing import preprocess, CLASS_NAMES

@st.cache_resource()
def load_model():
    model = torch.load('model/full_model.pth', map_location='cpu', weights_only=False)
    #model.load_state_dict(torch.load('model/model_weights.pth', map_location = 'cpu'))
    model.eval()
    return model

model = load_model()
st.title('Классификация изображений')
image = st.file_uploader('Upload file', type = ['jpg', 'jpeg', 'png'])

if image:
    img = Image.open(image).convert('RGB')
    st.image(img, caption='Исходное изображение', use_column_width=True)

    x = preprocess(img)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)
        conf, idx = torch.max(prob, 1)
        label = CLASS_NAMES[idx.item()]
    
    st.success(f'Предсказанный класс: **{label}** (уверенность {conf.item():.2%})')
#     img = preprocess(img)
#     pred = model(img)
#     return pred


# if image:
#     image = Image.open(image)
#     prediction = predict(image)
#     st.image(image)
#     st.write(prediction)
