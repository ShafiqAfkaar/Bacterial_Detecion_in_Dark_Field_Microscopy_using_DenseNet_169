import torch
import torch.nn as nn
from torchsummary import summary

def model_summary(model, input_size=(3, 96, 96)):
    """
    Print model summary
    """
    try:
        summary(model, input_size=input_size)
    except:
        print("Note: torchsummary not available. Install with: pip install torchsummary")
        print(f"Model architecture: {model}")

def count_parameters(model):
    """
    Count total and trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📈 Model Parameters:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    return total_params, trainable_params

def save_model(model, path, optimizer=None, scheduler=None, epoch=None, metrics=None):
    """
    Save model checkpoint
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }
    
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, path)
    print(f"✅ Model saved to: {path}")

def load_model(model, path, optimizer=None, scheduler=None):
    """
    Load model checkpoint
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"✅ Model loaded from: {path}")
    return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})