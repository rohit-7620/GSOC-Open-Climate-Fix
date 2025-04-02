"""
# Climate Forecasting with PyTorch - Demo

This notebook demonstrates the core functionality of the climate forecasting model for the GSoC 2025 project.
"""

# Import necessary libraries
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split

# Add project root to path (when running in colab or as a notebook)
project_root = ".."
if project_root not in sys.path:
    sys.path.append(project_root)

# Import project modules
from climate_forecast.data import ClimateDataset
from climate_forecast.models import ClimateTransformer
from climate_forecast.train import Trainer
from climate_forecast.utils.metrics import compute_metrics
from climate_forecast.utils.visualization import plot_forecast

"""
## 1. Generate Sample Data

For demonstration purposes, we'll generate synthetic climate data.
"""

# Generate synthetic climate data
def generate_sample_data(n_samples=1000, n_features=2, seed=42):
    """Generate synthetic climate data for demonstration."""
    np.random.seed(seed)
    
    # Time as the x-axis
    time = np.arange(n_samples)
    
    # Generate temperature with seasonal patterns and trend
    temp_seasonal = 15 + 10 * np.sin(2 * np.pi * time / 365)  # Yearly cycle
    temp_trend = 0.01 * time  # Warming trend
    temp_noise = np.random.normal(0, 1, n_samples)  # Random variations
    temperature = temp_seasonal + temp_trend + temp_noise
    
    # Generate precipitation with seasonal patterns
    precip_seasonal = 5 + 3 * np.sin(2 * np.pi * time / 365 + np.pi)  # Opposite season to temp
    precip_noise = np.random.exponential(1, n_samples)  # Precipitation is often skewed
    precipitation = np.maximum(0, precip_seasonal + precip_noise)  # No negative precipitation
    
    # Create a DataFrame
    df = pd.DataFrame({
        'date': pd.date_range(start='2020-01-01', periods=n_samples),
        'temperature': temperature,
        'precipitation': precipitation
    })
    
    return df

# Generate and save sample data
sample_data = generate_sample_data(n_samples=1095)  # ~3 years of daily data
print(sample_data.head())

# Plot sample data
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(sample_data['date'], sample_data['temperature'])
plt.title('Temperature')
plt.ylabel('Temperature (°C)')

plt.subplot(2, 1, 2)
plt.plot(sample_data['date'], sample_data['precipitation'])
plt.title('Precipitation')
plt.ylabel('Precipitation (mm)')

plt.tight_layout()
plt.show()

# Save sample data
os.makedirs("data/processed", exist_ok=True)
sample_data.to_csv("data/processed/sample_climate_data.csv", index=False)

"""
## 2. Create PyTorch Dataset
"""

# Create a custom PyTorch dataset class for the sample data
class SampleClimateDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_length=30, pred_horizon=7):
        self.data = data
        self.seq_length = seq_length
        self.pred_horizon = pred_horizon
        self.features = ['temperature', 'precipitation']
        
        # Normalize data
        self.mean = data[self.features].mean()
        self.std = data[self.features].std()
        self.normalized_data = (data[self.features] - self.mean) / self.std
        
        # Create samples
        self._prepare_samples()
        
    def _prepare_samples(self):
        data_array = self.normalized_data.values
        self.samples = []
        
        for i in range(len(data_array) - self.seq_length - self.pred_horizon + 1):
            x = data_array[i:i+self.seq_length]
            y = data_array[i+self.seq_length:i+self.seq_length+self.pred_horizon]
            self.samples.append((x, y))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    def denormalize(self, data):
        """Convert normalized values back to original scale."""
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        return data * self.std.values + self.mean.values

# Create dataset
dataset = SampleClimateDataset(sample_data, seq_length=30, pred_horizon=7)
print(f"Dataset size: {len(dataset)}")
print(f"Sample shape - Input: {dataset[0][0].shape}, Target: {dataset[0][1].shape}")

# Split data
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [
