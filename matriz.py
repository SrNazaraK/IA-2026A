import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# 1. CARGAR EL MODELO Y LOS DATOS
model = load_model('modelo_residuos.h5')
dataset_path = 'data/residuos' # Asegúrate de que esta sea tu ruta
img_size = (224, 224)

# Configuramos el generador para los datos de prueba (validation)
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

test_generator = test_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    subset='validation', # Usamos el 20% que el modelo NO vio en el entrenamiento
    shuffle=False  # ¡IMPORTANTE! No mezclar para que coincidan etiquetas y predicciones
)

# 2. HACER PREDICCIONES
print("Evaluando el modelo...")
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1) # Clase predicha
y_true = test_generator.classes     # Clase real

# 3. CREAR MATRIZ DE CONFUSIÓN
cm = confusion_matrix(y_true, y_pred)
class_names = list(test_generator.class_indices.keys())

# 4. GRAFICAR
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusión - Clasificación de Residuos')
plt.ylabel('Clase Real')
plt.xlabel('Predicción de la IA')
plt.savefig('matriz_confusion.png') # Se guarda como imagen para tu informe
plt.show()

# 5. MOSTRAR MÉTRICAS (Precision, Recall, F-score)
print("\nReporte de Clasificación:")
print(classification_report(y_true, y_pred, target_labels=class_names))