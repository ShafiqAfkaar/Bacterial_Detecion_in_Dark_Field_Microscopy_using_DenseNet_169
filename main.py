import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.densenet_model import DenseNet169_Base
from utils.data_loader import BacteriaDataset
from utils.transforms import get_train_transform, get_val_test_transform
from training.train import train_model
from evaluation.evaluate import evaluate_model
from utils.visualization import visualize_predictions
from training.config import Config

def main():
    parser = argparse.ArgumentParser(description='Bacteria Classification Project')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'visualize'],
                       help='Mode: train, evaluate, or visualize')
    parser.add_argument('--data_dir', type=str, default=Config.DATA_DIR,
                       help='Path to dataset directory')
    parser.add_argument('--model_path', type=str, default=Config.MODEL_PATH,
                       help='Path to model for evaluation/visualization')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get transforms
    train_transform = get_train_transform(Config.IMG_SIZE)
    val_test_transform = get_val_test_transform(Config.IMG_SIZE)

    if args.mode == 'train':
        print("🚀 Starting training...")
        
        # Load data
        train_data = BacteriaDataset(os.path.join(args.data_dir, 'train'), transform=train_transform)
        val_data = BacteriaDataset(os.path.join(args.data_dir, 'val'), transform=val_test_transform)
        
        train_loader = DataLoader(train_data, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=Config.BATCH_SIZE, shuffle=False)
        
        # Initialize model
        model = DenseNet169_Base(num_classes=2).to(device)
        
        # Train
        train_model(model, train_loader, val_loader, device, Config)
        
    elif args.mode == 'evaluate':
        print("📊 Starting evaluation...")
        
        # Load model
        model = DenseNet169_Base(num_classes=2, pretrained=False).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        
        # Evaluate
        evaluate_model(model, os.path.join(args.data_dir, 'test'), val_test_transform, device)
        
    elif args.mode == 'visualize':
        print("📈 Generating visualizations...")
        
        # Load model
        model = DenseNet169_Base(num_classes=2, pretrained=False).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        
        # Load test samples for visualization
        label_map = {'negative': 0, 'positive': 1}
        test_data = BacteriaDataset(os.path.join(args.data_dir, 'test'), transform=val_test_transform)
        
        # Create visualization
        fig = visualize_predictions(model, test_data.samples, val_test_transform, device, label_map)
        
        # Save and show
        fig.savefig('prediction_visualization.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved as 'prediction_visualization.png'")
        import matplotlib.pyplot as plt
        plt.show()

if __name__ == "__main__":
    main()
