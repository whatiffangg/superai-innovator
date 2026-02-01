import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks
from PIL import Image
import shutil
import json


# ---------- CONFIG ----------
IMG_SIZE = (224, 224)
BATCH_SIZE = 48
EPOCHS = 120
NUM_CLASSES = 5
TRAIN_DIR = '/home/saksorn.bu@FUSION.LAB/rubber_ai/Test_model/train'
VAL_DIR = '/home/saksorn.bu@FUSION.LAB/rubber_ai/Test_model/val'
MODEL_SAVE_PATH = '/home/saksorn.bu@FUSION.LAB/rubber_ai/Test_model/rubber_leaf_model_best.h5'
# ----------------------------


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb'
)

# Model Architecture (RGB => 3 channels)
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks: EarlyStopping + Save Best Model
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=25,
    restore_best_weights=True
)

checkpoint = callbacks.ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Train
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stop, checkpoint]
)

# Final Save (optional, full model after training)
model.save('rubber_leaf_model_final.h5')

print("Training completed and model saved.")


# Save the history to a JSON file
with open('history.json', 'w') as f:
    json.dump(history.history, f)

