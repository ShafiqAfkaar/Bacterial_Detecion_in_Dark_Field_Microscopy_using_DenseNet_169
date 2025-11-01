import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image
import torch

def visualize_predictions(model, samples, transform, device, label_map, num_images=25):
    reverse_label_map = {v: k for k, v in label_map.items()}
    chosen_samples = random.sample(samples, min(num_images, len(samples)))
    
    fig, axs = plt.subplots(5, 5, figsize=(15, 15))
    fig.suptitle("Predicted vs True Labels", fontsize=18)
    
    for idx, (path, true_label) in enumerate(chosen_samples):
        if idx >= 25:
            break
            
        image = Image.open(path).convert("RGB")
        display_img = image.copy()
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
        
        pred_label = predicted.item()
        label_text = f"Pred: {reverse_label_map[pred_label]}\nTrue: {reverse_label_map[true_label]}"
        axs[idx // 5, idx % 5].imshow(display_img)
        axs[idx // 5, idx % 5].set_title(label_text, color="green" if pred_label == true_label else "red")
        axs[idx // 5, idx % 5].axis("off")
    
    plt.tight_layout()
    return fig