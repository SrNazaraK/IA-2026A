# Clasificador de Residuos con Inteligencia Artificial - URU 2026

Este proyecto desarrolla un sistema de visión artificial capaz de clasificar residuos en 6 categorías: Vidrio, Papel, Cartón, Metal, Plástico y Trash (Varios). Utiliza **Transfer Learning** con la arquitectura **MobileNetV2**.

## 🚀 Requisitos Técnicos
El proyecto fue desarrollado bajo los siguientes estándares:
* **Lenguaje:** Python 3.11
* **Librerías:** TensorFlow, Keras, OpenCV, Scikit-learn, Matplotlib, Seaborn, NumPy, Pandas.
* **Interfaz:** Streamlit

## 📊 Rendimiento del Modelo
El modelo alcanzó una precisión de entrenamiento del **88%** y una validación del **75%**. 
* **Fortalezas:** Alta precisión en detección de Papel y Metal.
* **Debilidades:** Confusión entre Vidrio/Plástico por transparencia y dificultades con la categoría 'Trash' por ambigüedad visual.

## 🛠️ Instalación y Uso
1. Clonar el repositorio.
2. Crear un entorno virtual: `python -m venv venv`
3. Activar el entorno: `.\venv\Scripts\activate`
4. Instalar dependencias: `pip install -r requirements.txt`
5. Ejecutar la aplicación:
   ```bash
   streamlit run app.py 