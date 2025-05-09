
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import sys
# import os
# import json
# import logging

# # Suppress TensorFlow logs
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.get_logger().setLevel(logging.ERROR)

# # Get root directory and model path
# root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# model_path = os.path.join(root_dir, 'brain_tumor_resnet50.h5')

# # Load the model
# model = tf.keras.models.load_model(model_path)

# # Get image path from command-line
# image_path = sys.argv[1]

# # (Optional) Debug log to stderr, NOT stdout
# print(f"Processing image: {image_path}", file=sys.stderr)

# # Load and preprocess the image
# img = image.load_img(image_path, target_size=(224, 224))
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
# img_array = img_array / 255.0  # Normalize

# # Make prediction
# predictions = model.predict(img_array)
# predicted_class = np.argmax(predictions, axis=1)

# # Class mapping
# class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
# predicted_label = class_labels[predicted_class[0]]

# # Final output (only this line will go to stdout)
# print(json.dumps({"prediction": predicted_label}))









# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import sys
# import os
# import json
# import logging

# # Suppress TensorFlow logs
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.get_logger().setLevel(logging.ERROR)

# # Get root directory and model path
# root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# model_path = os.path.join(root_dir, 'brain_tumor_resnet50_20250415_1757.h5')

# # Load the model
# model = tf.keras.models.load_model(model_path)

# # Get image path from command-line
# image_path = sys.argv[1]

# # (Optional) Debug log to stderr, NOT stdout
# print(f"Processing image: {image_path}", file=sys.stderr)

# # Load and preprocess the image
# img = image.load_img(image_path, target_size=(224, 224))
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
# img_array = img_array / 255.0  # Normalize

# # Make prediction
# predictions = model.predict(img_array)
# predicted_class = np.argmax(predictions, axis=1)
# confidence = float(np.max(predictions))  # Get the highest probability score

# # Class mapping
# class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
# predicted_label = class_labels[predicted_class[0]]

# # Final output with confidence score
# print(json.dumps({
#     "prediction": predicted_label,
#     "confidence": confidence  # Added confidence score
# }))





import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import sys
import os
import json
import logging

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)

# Get root directory and model path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_path = os.path.join(root_dir,'model', 'brain_tumor_resnet50_20250415_1757.h5')

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}", file=sys.stderr)
    sys.exit(1)

# Load the model
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}", file=sys.stderr)
    sys.exit(1)

# Get image path from command-line
image_path = sys.argv[1]

# (Optional) Debug log to stderr, NOT stdout
print(f"Processing image: {image_path}", file=sys.stderr)

# Check if the image file exists
if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}", file=sys.stderr)
    sys.exit(1)

# Load and preprocess the image
try:
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
except Exception as e:
    print(f"Error loading or preprocessing image: {e}", file=sys.stderr)
    sys.exit(1)

# Make prediction
try:
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    confidence = float(np.max(predictions))  # Get the highest probability score
except Exception as e:
    print(f"Error making prediction: {e}", file=sys.stderr)
    sys.exit(1)

# Class mapping
class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
predicted_label = class_labels[predicted_class[0]]

# Final output with confidence score
print(json.dumps({
    "prediction": predicted_label,
    "confidence": confidence  # Added confidence score
}))
