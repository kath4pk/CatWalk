import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow as tf
layers = tf.keras.layers
import cv2
import os

# 1.- Configuración del dataset y parámetros
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42

DATASET_DIR = 'C:/Users/kjime/OneDrive/Documentos/archive/Apparel images dataset new'

def load_data():
    """
    Carga el dataset de imágenes de ropa desde el directorio especificado.
    Divide el dataset en conjuntos de entrenamiento y validación,
    y aplica la normalización de píxeles.
    """
    # Verifica si el directorio del dataset existe
    if not os.path.exists(DATASET_DIR):
        print(f"Error: El directorio del dataset no se encontró en '{DATASET_DIR}'.")
        print("Asegúrate de que la ruta sea correcta y que el dataset esté descomprimido en esa ubicación.")
        print("El script espera que las imágenes estén organizadas en subcarpetas, donde cada subcarpeta es una clase (ej. 'Apparel images dataset new/blue_pants/', 'Apparel images dataset new/t_shirt/', etc.).")
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    print(f"Cargando datos desde: {DATASET_DIR}")
    
    # Carga el dataset de entrenamiento
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        labels='inferred',
        label_mode='int',
        image_size=(IMG_SIZE, IMG_SIZE),
        interpolation='bilinear',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=0.2,
        subset='training'
    )
    
    # Carga el dataset de prueba/validación
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        labels='inferred',
        label_mode='int',
        image_size=(IMG_SIZE, IMG_SIZE),
        interpolation='bilinear',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=0.2,
        subset='validation'
    )
    
    # Obtiene los nombres de las clases inferidos del directorio
    class_names = train_ds.class_names
    print(f"Clases detectadas: {class_names}")
    print(f"Número de clases: {len(class_names)}")

    # Crea una capa de normalización para escalar los valores de los píxeles de 0-255 a 0-1
    normalization_layer = layers.Rescaling(1./255)
    
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds, class_names

# Carga los datos al inicio del script
train_ds, test_ds, class_names = load_data()

# 3.- Mostrar datos del dataset (9 imágenes aleatorias con sus etiquetas)
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy())
        
        # Asegúrate de que el índice de la etiqueta esté dentro del rango de class_names
        if labels[i].numpy() < len(class_names):
            plt.title(class_names[labels[i].numpy()])
        else:
            plt.title(f"Etiqueta fuera de rango: {labels[i].numpy()}")
        plt.axis('off')
plt.show()

# 4.- Construir el modelo de red neuronal
def build_model(num_classes):
    """
    Construye el modelo de red neuronal utilizando EfficientNetB0 como modelo base
    para 'transfer learning', incluyendo capas de aumento de datos y dropout.
    El modelo base es entrenable desde el principio con una tasa de aprendizaje baja.
    """
    # Capas de aumento de datos: se aplican en cada época al dataset de entrenamiento
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

    # Carga el modelo EfficientNetB0 pre-entrenado en ImageNet
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    base_model.trainable = True

    # Crea el modelo completo: aumento de datos + modelo base + cabezal de clasificación
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    
    return model

# Construye el modelo, pasando el número de clases detectadas
model = build_model(len(class_names))

# 5.- Compilar y entrenar el modelo (una sola fase, con tasa de aprendizaje baja para fine-tuning)
print("\n--- Iniciando Entrenamiento de CatWalk ---")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

total_epochs = 5
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=total_epochs,
    verbose=1
)
print("Catwalk entrenada ˃ 𖥦 ˂")

# 6.- Evaluar el modelo
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n🔎 Precisión en test: {test_acc*100:.2f}%")

# Gráficas de entrenamiento (para una sola fase de entrenamiento)
plt.figure(figsize=(12, 4))

# Gráfico de precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
plt.title('Precisión durante el Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

# Gráfico de pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de Validación')
plt.title('Pérdida durante el Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# 8.- Función para reconocimiento en tiempo real desde la cámara
def recognize_from_webcam(model, class_names, img_size):
    """
    Realiza el reconocimiento de ropa en tiempo real utilizando la cámara web.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara. Intentando otros índices...")
        for i in range(1, 5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Cámara en índice {i} abierta correctamente.")
                break
        else:
            print("Error: No se encontró ninguna cámara disponible o no se pudo abrir. Asegúrate de que no esté en uso y que los controladores estén instalados.")
            return

    print("\n--- Reconocimiento de ropa en tiempo real˃ 𖥦 ˂ ---")
    print("Presiona 'q' para salir de la ventana de la cámara.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el fotograma. Saliendo...")
            break

        # Voltea el fotograma horizontalmente para una visualización más natural (como un espejo)
        frame = cv2.flip(frame, 1)

        # Preprocesamiento del fotograma para la predicción del modelo
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = tf.image.resize(rgb_frame, (img_size, img_size))
        normalized_frame = tf.cast(resized_frame, tf.float32) / 255.0
        input_tensor = np.expand_dims(normalized_frame, axis=0)

        # Realiza la predicción con el modelo
        preds = model.predict(input_tensor, verbose=0)
        predicted_class_idx = np.argmax(preds)
        
        # Asegúrate de que el índice predicho esté dentro del rango de las clases conocidas
        if predicted_class_idx < len(class_names):
            predicted_class_name = class_names[predicted_class_idx]
            confidence = np.max(preds) * 100
        else:
            predicted_class_name = "Clase Desconocida" 
            confidence = 0.0
            print(f"Advertencia: Índice de clase predicho fuera de rango: {predicted_class_idx}")

        # Formatea el texto a mostrar en la ventana de la cámara
        text = f"Prediccion: {predicted_class_name} ({confidence:.2f}%)"


        # Dibuja el texto en el fotograma original (en formato BGR)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Reconocimiento de Ropa en Tiempo Real', frame)

        # Sale del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera los recursos de la cámara y cierra todas las ventanas de OpenCV
    cap.release()
    cv2.destroyAllWindows()

# Llama a la función de reconocimiento en tiempo real después de que el modelo ha sido entrenado
recognize_from_webcam(model, class_names, IMG_SIZE)