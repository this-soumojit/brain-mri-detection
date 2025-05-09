import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# Directory paths
train_dir = 'C:/Users/Soumojit Ghosh/Desktop/brain-mri-detection/training'
validation_dir = 'C:/Users/Soumojit Ghosh/Desktop/brain-mri-detection/testing'

# Data augmentation and image preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    # Add some more augmentation to help generalization
    brightness_range=[0.8, 1.2],  # Random brightness adjustment
    channel_shift_range=20.0  # Random color adjustment
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Data generators for training and validation
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # ResNet50 expects 224x224 images
    batch_size=32,
    class_mode='categorical'  # For multi-class classification
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load pre-trained ResNet50 without the top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
base_model.trainable = False

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')  # 4 classes: glioma, meningioma, notumor, pituitary
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks: ModelCheckpoint and EarlyStopping
checkpoint_path = 'best_model.h5'
checkpoint = ModelCheckpoint(
    checkpoint_path, 
    save_best_only=True, 
    save_weights_only=False, 
    monitor='val_loss', 
    mode='min', 
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5,  # Stop if no improvement in 5 epochs
    restore_best_weights=True,  # Restore the weights from the best epoch
    verbose=1
)

# Check if a checkpoint exists and load the weights if available
if os.path.exists(checkpoint_path):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model.load_weights(checkpoint_path)

# Train the model with the callbacks
history = model.fit(
    train_generator,
    epochs=50,  # Initial number of epochs, you can change it to a larger number
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stopping],
    verbose=1
)

# Save the final trained model (optional, but it's a good practice)
model.save('final_brain_tumor_resnet50.h5')




