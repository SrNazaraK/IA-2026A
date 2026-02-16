import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Configuración de la página
st.title("♻️ Clasificador de Residuos IA")
st.write("Sube una foto para identificar si es plástico, papel, vidrio u orgánico.")

# 1. Cargar el modelo entrenado
@st.cache_resource
def load_my_model():
    # Asegúrate de que el nombre coincida con el archivo guardado en el paso anterior
    return tf.keras.models.load_model('modelo_residuos.h5')

model = load_my_model()

# Definir las etiquetas (deben estar en el mismo orden que las carpetas del dataset)
labels = ['Cartón', 'Vidrio', 'Metal', 'Papel', 'Plástico', 'Orgánico']

# 2. Interfaz de carga de archivos
uploaded_file = st.file_uploader("Selecciona una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen cargada
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen subida', use_container_width=True)
    st.write("Clasificando...")

    # 3. Preprocesamiento de la imagen para el modelo
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalización [cite: 21]
    img_array = np.expand_dims(img_array, axis=0)

    # 4. Predicción
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    # 5. Mostrar resultado
    resultado = labels[np.argmax(score)]
    confianza = 100 * np.max(score)
    
    st.success(f"Resultado: **{resultado}**")
    st.info(f"Nivel de confianza: {confianza:.2f}%")    