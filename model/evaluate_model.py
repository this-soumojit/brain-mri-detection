import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# [INFO] Set paths
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_path = os.path.join(root_dir, 'model', 'brain_tumor_resnet50_20250415_1757.h5')
test_dir = os.path.join(root_dir, 'Testing')  # Adjust if your test folder is elsewhere

# [INFO] Load the model
model = tf.keras.models.load_model(model_path)
print("[INFO] Loaded model successfully.")

# [INFO] Prepare test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# [INFO] Evaluate the model
loss, acc, auc, precision, recall = model.evaluate(validation_generator)

# [INFO] Print evaluation results
print(f"\n[RESULTS]")
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {acc:.2%}")
print(f"Test AUC      : {auc:.2%}")
print(f"Precision     : {precision:.2%}")
print(f"Recall        : {recall:.2%}")

# [Optional] Classification Report
y_pred = model.predict(validation_generator)
y_pred_classes = y_pred.argmax(axis=1)
y_true = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

print("\n[CLASSIFICATION REPORT]")
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

# [Optional] Confusion Matrix
print("[CONFUSION MATRIX]")
print(confusion_matrix(y_true, y_pred_classes))
