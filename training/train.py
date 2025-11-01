import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

def calculate_class_weights(train_data, device):
    label_counts = [0, 0]
    for _, label in train_data:
        label_counts[label] += 1
    
    total_samples = sum(label_counts)
    class_weights = torch.tensor([
        total_samples / (2 * count) if count > 0 else 0.0 for count in label_counts
    ], dtype=torch.float).to(device)
    return class_weights

def train_model(model, train_loader, val_loader, device, config):
    class_weights = calculate_class_weights(train_loader.dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                loss = criterion(val_outputs, val_labels)
                val_loss += loss.item()
                _, val_predicted = torch.max(val_outputs, 1)
                val_correct += (val_predicted == val_labels).sum().item()
                val_total += val_labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), config.MODEL_PATH)
    print("✅ Final model saved successfully!")
    
    return model