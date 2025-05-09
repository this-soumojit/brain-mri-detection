import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
    BackupAndRestore,
    TerminateOnNaN
)
import os
import json
import numpy as np
from datetime import datetime


# ======================
# 1. Configuration
# ======================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20  # Reduced for your system's performance
NUM_CLASSES = 4
CHECKPOINT_DIR = './training_checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ======================
# 2. Data Preparation
# ======================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,  # Increased from 30
    width_shift_range=0.25,  # Increased from 0.2
    height_shift_range=0.25,
    shear_range=0.25,
    zoom_range=0.3,  # Increased from 0.2
    horizontal_flip=True,
    vertical_flip=True,  # New augmentation
    fill_mode='nearest',
    brightness_range=[0.7, 1.3],  # Wider range
    channel_shift_range=40.0  # Increased color variation
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '../Training',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True  # Important for better training
)

validation_generator = test_datagen.flow_from_directory(
    '../Testing',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ======================
# 3. Model Architecture
# ======================
def build_model():
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
        pooling='avg'  # Better than Flatten
    )

    # Unfreeze top layers for better fine-tuning
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    model = models.Sequential([
        base_model,
        layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),  # Added for stability
        layers.Dropout(0.6),  # Slightly higher for regularization
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Lower learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.AUC(name='auc'),  # Additional metric
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    return model

# ======================
# 4. Training Setup
# ======================
def get_callbacks():
    return [
        ModelCheckpoint(
            os.path.join(CHECKPOINT_DIR, 'best_model_epoch_{epoch:02d}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            save_weights_only=False,
            verbose=1
        ),
        BackupAndRestore(CHECKPOINT_DIR),  # For resuming training
        EarlyStopping(
            monitor='val_loss',
            patience=15,  # Increased patience
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
        CSVLogger('training_log.csv'),
        TerminateOnNaN()  # Safety feature
    ]

# ======================
# 5. Training Execution
# ======================
def train_model():
    # Check for existing checkpoints
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) 
                      if f.startswith('best_model_epoch_')]
    
    if checkpoint_files:
        latest_checkpoint = max(
            [os.path.join(CHECKPOINT_DIR, f) for f in checkpoint_files],
            key=os.path.getctime
        )
        print(f"\nResuming training from: {latest_checkpoint}")
        model = tf.keras.models.load_model(latest_checkpoint)
        initial_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0])
    else:
        print("\nStarting new training session")
        model = build_model()
        initial_epoch = 0

    print("\nModel Summary:")
    model.summary()

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        validation_data=validation_generator,
        callbacks=get_callbacks(),
        verbose=1
    )

    # Save final model with timestamp
    model.save(f'brain_tumor_resnet50_{datetime.now().strftime("%Y%m%d_%H%M")}.h5')
    return history

# ======================
# 6. Run Training
# ======================
if __name__ == "__main__":
    history = train_model()
    
    # Post-training analysis
    print("\nTraining completed. Evaluating final model...")
    test_loss, test_acc, test_auc, test_precision, test_recall = model.evaluate(validation_generator)
    print(f"\nTest Accuracy: {test_acc:.2%}")
    print(f"Test AUC: {test_auc:.2%}")
    print(f"Test Precision: {test_precision:.2%}")
    print(f"Test Recall: {test_recall:.2%}")
