# model_evaluate.py

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
data = np.load('forgery_dataset.npz')
X_test, y_test = data['X_test'], data['y_test']

# Load model
model = load_model('forgery_cnn_augmented.h5')

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")

# Predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Confusion Matrix & Report
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
