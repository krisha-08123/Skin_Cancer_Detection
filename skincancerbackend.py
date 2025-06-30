import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix



import shutil
import os

src = r"C:\Users\Admin\Downloads\archive"  
dst = "archive"

if os.path.exists(dst):
    shutil.rmtree(dst)

shutil.copytree(src, dst)

print(os.listdir(dst))

if not os.path.exists("static"):
    os.makedirs("static")


IMAGE_SIZE = (180, 180)
BATCH_SIZE = 32



datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)



train_data = datagen.flow_from_directory(
    dst,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE, 
    class_mode="categorical", 
    subset="training"
)


val_data = datagen.flow_from_directory(
    dst, 
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE, 
    class_mode="categorical",
    subset="validation"
)


model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_data.num_classes, activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_data, validation_data=val_data, epochs=5)


model.save('skin_cancer_model.h5', include_optimizer=False)
print("Model saved successfully.")


plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label="Training Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Over Epochs")
plt.legend()
plt.savefig("static/accuracy_plot.png")  
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Model Loss Over Epochs")
plt.legend()
plt.savefig("static/loss_plot.png")  
plt.show()

y_true = val_data.classes
y_pred = model.predict(val_data)
y_pred_classes = np.argmax(y_pred, axis=1)

report = classification_report(y_true, y_pred_classes, target_names=train_data.class_indices.keys())
print(report)
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=train_data.class_indices.keys(), yticklabels=train_data.class_indices.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("static/confusion_matrix.png")  
plt.show()
print("Training done!")







