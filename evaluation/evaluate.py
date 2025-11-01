import torch
from sklearn.metrics import classification_report, confusion_matrix
import os
from PIL import Image

def evaluate_model(model, test_dir, transform, device):
    label_map = {'negative': 0, 'positive': 1}
    reverse_label_map = {v: k for k, v in label_map.items()}
    samples = []
    
    for label_name in label_map:
        folder = os.path.join(test_dir, label_name)
        for img_name in os.listdir(folder):
            path = os.path.join(folder, img_name)
            samples.append((path, label_map[label_name]))

    y_true = []
    y_pred = []

    with torch.no_grad():
        for path, label in samples:
            image = Image.open(path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            y_true.append(label)
            y_pred.append(predicted.item())

    print("✅ Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=label_map.keys()))

    print("🧮 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    return y_true, y_pred, samples