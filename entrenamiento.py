import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns

# 1. PREPROCESAMIENTO Y CARGA DE DATOS [cite: 20, 21, 22]
# Ajusta 'path_to_dataset' a la carpeta donde tengas TrashNet o tus fotos
dataset_path = 'data/residuos' 
img_size = (224, 224)
batch_size = 32

# Aumentación de datos para mejorar la diversidad [cite: 23]
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
    validation_split=0.2 # División entrenamiento/prueba [cite: 22]
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

test_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# 2. DISEÑO DEL MODELO (CNN con Transfer Learning) [cite: 24, 25, 26]
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Congelamos las capas base

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax') # Capa de salida 
])

# 3. ENTRENAMIENTO [cite: 27]
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Iniciando entrenamiento...")
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# 4. EVALUACIÓN Y MÉTRICAS [cite: 33, 34, 39]
loss, accuracy = model.evaluate(test_generator)
print(f"Precisión en el test: {accuracy*100:.2f}%")

# Guardar el modelo para la interfaz [cite: 37]
model.save('modelo_residuos.h5')

# Visualización de resultados [cite: 43]
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Rendimiento del Modelo')
plt.legend()
plt.show()