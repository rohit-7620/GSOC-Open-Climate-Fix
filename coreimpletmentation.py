# data/datasets.py
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import xarray as xr

class ClimateDataset(Dataset):
    """PyTorch Dataset for climate data."""
    
    def __init__(self, 
                 path, 
                 variables=['temperature', 'precipitation'],
                 seq_length=24, 
                 prediction_horizon=7,
                 transform=None,
                 target_transform=None):
        """
        Args:
            path (str): Path to the dataset files
            variables (list): List of climate variables to use
            seq_length (int): Number of time steps for input sequence
            prediction_horizon (int): Number of time steps to predict
            transform (callable, optional): Transform to apply to inputs
            target_transform (callable, optional): Transform to apply to targets
        """
        self.path = path
        self.variables = variables
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        self.transform = transform
        self.target_transform = target_transform
        
        # Load dataset
        self.data = self._load_data()
        self.input_dim = len(variables)
        
        # Generate samples
        self._prepare_samples()
        
    def _load_data(self):
        """Load and preprocess climate data."""
        try:
            # For xarray datasets (like ERA5, NOAA)
            ds = xr.open_dataset(f"{self.path}/climate_data.nc")
            # Select only the variables we want
            ds = ds[self.variables]
            return ds
        except:
            # Fallback to CSV if netCDF is not available
            df = pd.read_csv(f"{self.path}/climate_data.csv")
            return df
    
    def _prepare_samples(self):
        """Prepare input-target pairs for training."""
        # Convert xarray to numpy for easier processing
        if isinstance(self.data, xr.Dataset):
            data_array = np.stack([self.data[var].values for var in self.variables], axis=-1)
        else:
            data_array = self.data[self.variables].values
        
        # Create sequences for training
        self.samples = []
        for i in range(len(data_array) - self.seq_length - self.prediction_horizon + 1):
            x = data_array[i:i+self.seq_length]
            y = data_array[i+self.seq_length:i+self.seq_length+self.prediction_horizon]
            self.samples.append((x, y))
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Return a sample from the dataset."""
        x, y = self.samples[idx]
        
        # Convert to tensors
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        # Apply transforms if available
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
            
        return x, y

# models/transformer.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return x

class ClimateTransformer(nn.Module):
    """Transformer model for climate forecasting."""
    
    def __init__(self, 
                 input_dim, 
                 hidden_dim=256, 
                 output_dim=None,
                 num_heads=8, 
                 num_layers=6, 
                 dropout=0.1):
        """
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Size of hidden dimension
            output_dim (int, optional): Number of output features. If None, same as input_dim
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dropout (float): Dropout rate
        """
        super(ClimateTransformer, self).__init__()
        
        if output_dim is None:
            output_dim = input_dim
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, src_mask=None):
        """
        Args:
            src: Source sequence [batch_size, seq_len, input_dim]
            src_mask: Source mask for transformer
            
        Returns:
            output: Predicted sequence [batch_size, seq_len, output_dim]
        """
        # Embed input
        x = self.input_embedding(src)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        if src_mask is None:
            # Create causal mask (for autoregressive prediction)
            seq_len = src.size(1)
            src_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
            src_mask = src_mask.to(src.device)
            
        output = self.transformer_encoder(x, src_mask)
        
        # Project to output dimension
        output = self.output_projection(output)
        
        return output

# train/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
import os
from datetime import datetime

from climate_forecast.utils.metrics import compute_metrics

class Trainer:
    """Trainer class for climate forecasting models."""
    
    def __init__(
        self,
        model,
        dataset,
        val_dataset=None,
        batch_size=32,
        learning_rate=1e-3,
        num_epochs=100,
        device=None,
        checkpoint_dir='checkpoints',
        use_wandb=False
    ):
        """
        Args:
            model: PyTorch model
            dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            device: Device to train on ('cuda' or 'cpu')
            checkpoint_dir: Directory to save checkpoints
            use_wandb: Whether to use Weights & Biases for logging
        """
        self.model = model
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model = self.model.to(self.device)
        
        # Create data loaders
        self.train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        else:
            self.val_loader = None
            
        # Setup loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # Initialize logger
        if use_wandb:
            import wandb
            wandb.init(project="climate-forecast-pytorch")
            wandb.config.update({
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "model": model.__class__.__name__
            })
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def train(self):
        """Train the model."""
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            pbar = tqdm(self.train_loader)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                pbar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
            
            train_loss /= len(self.train_loader)
            
            # Validation phase
            val_loss = 0.0
            val_metrics = {}
            
            if self.val_loader:
                val_loss, val_metrics = self._validate()
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)
            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)
            
            # Log results
            self._log_results(epoch, train_loss, val_loss, val_metrics)
            
        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        return self.model
    
    def _validate(self):
        """Run validation."""
        self.model.eval()
        val_loss = 0.0
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                
                # Store predictions and targets for metrics
                all_targets.append(targets.cpu().numpy())
                all_predictions.append(outputs.cpu().numpy())
        
        val_loss /= len(self.val_loader)
        
        # Compute additional metrics
        all_targets = np.concatenate(all_targets)
        all_predictions = np.concatenate(all_predictions)
        metrics = compute_metrics(all_targets, all_predictions)
        
        return val_loss, metrics
    
    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if is_best:
            checkpoint_path = f"{self.checkpoint_dir}/best_model.pt"
        else:
            checkpoint_path = f"{self.checkpoint_dir}/model_epoch{epoch+1}_{timestamp}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def _log_results(self, epoch, train_loss, val_loss, val_metrics):
        """Log training results."""
        print(f"\nEpoch {epoch+1}/{self.num_epochs}")
        print(f"Train Loss: {train_loss:.6f}")
        
        if val_loss:
            print(f"Validation Loss: {val_loss:.6f}")
            for metric_name, metric_value in val_metrics.items():
                print(f"Validation {metric_name}: {metric_value:.6f}")
        
        if self.use_wandb:
            import wandb
            log_dict = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            log_dict.update({f'val_{k}': v for k, v in val_metrics.items()})
            wandb.log(log_dict)

# utils/metrics.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred, epsilon=1e-10):
    """Calculate Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = np.abs(y_true) > epsilon
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.maximum(np.abs(y_true[mask]), epsilon))) * 100

def compute_metrics(y_true, y_pred):
    """Compute multiple evaluation metrics."""
    metrics = {
        'rmse': rmse(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': mape(y_true, y_pred)
    }
    return metrics

# examples/train_model.py
import os
import sys
import argparse
import torch

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from climate_forecast.data import ClimateDataset
from climate_forecast.models import ClimateTransformer
from climate_forecast.train import Trainer

def main(args):
    """Main training function."""
    print(f"Training with configuration: {args}")
    
    # Create datasets
    train_dataset = ClimateDataset(
        path=args.data_path,
        variables=args.variables.split(','),
        seq_length=args.seq_length,
        prediction_horizon=args.pred_horizon
    )
    
    # Create validation split if needed
    if args.val_split > 0:
        # Simple random split for demonstration
        dataset_size = len(train_dataset)
        val_size = int(dataset_size * args.val_split)
        train_size = dataset_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    else:
        val_dataset = None
    
    # Create model
    model = ClimateTransformer(
        input_dim=len(args.variables.split(',')),
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.use_wandb
    )
    
    # Train model
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a climate forecasting model")
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--variables', type=str, default='temperature,precipitation', 
                        help='Comma-separated list of variables to use')
    parser.add_argument('--seq_length', type=int, default=24, help='Input sequence length')
    parser.add_argument('--pred_horizon', type=int, default=7, help='Prediction horizon length')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation set ratio')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', 
                        help='Directory to save checkpoints')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    
    args = parser.parse_args()
    main(args)
