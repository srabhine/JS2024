import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import polars as pl
import tensorflow as tf

class FinancialDataset(Dataset):
    def __init__(self, data, feature_names, label_name, sequence_length=10):
        self.sequence_length = sequence_length
        
        # Normalize features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(data[feature_names])
        
        # Convert to tensors
        self.features = torch.FloatTensor(self.features)
        self.labels = torch.FloatTensor(data[label_name].values)
        
    def __len__(self):
        return len(self.features) - self.sequence_length
        
    def __getitem__(self, idx):
        # Get sequence of features
        feature_seq = self.features[idx:idx + self.sequence_length]
        # Get corresponding label (next timestep prediction)
        label = self.labels[idx + self.sequence_length]
        
        return feature_seq, label

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)  # Changed to include batch dimension first
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]  # Changed indexing to match batch_first=True
        return self.dropout(x)
    


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Project input features to d_model dimensions
        self.encoder = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layer
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # Project to d_model dimensions
        x = self.encoder(x)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)  # [batch_size, seq_len, d_model]
        
        # Pass through transformer
        x = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]
        
        # Take the last sequence element for prediction
        x = x[:, -1, :]  # [batch_size, d_model]
        
        # Decode to output
        x = self.decoder(x)  # [batch_size, 1]
        
        return x

def load_data(train_path, start_id, end_id):
    """
    Load and combine parquet files from specified date_id range
    
    Args:
        train_path (str): Base path to the training data directory
        start_id (int): Starting date_id
        end_id (int): Ending date_id (inclusive)
    
    Returns:
        pd.DataFrame: Combined and processed DataFrame
    """
    folder_paths = [
        f"{train_path}/date_id={partition_id}/00000000.parquet"
        for partition_id in range(start_id, end_id + 1)
    ]
    
    # Check if files exist before loading
    valid_paths = []
    for path in folder_paths:
        try:
            if pl.scan_parquet(path) is not None:
                valid_paths.append(path)
        except Exception as e:
            print(f"Warning: Could not load file {path}: {str(e)}")
    
    if not valid_paths:
        raise ValueError("No valid parquet files found in the specified range")
    
    # Load and combine data
    lazy_frames = [pl.scan_parquet(path) for path in valid_paths]
    combined_lazy_df = pl.concat(lazy_frames)
    data = combined_lazy_df.collect().to_pandas()
    
    # Forward fill and fill remaining NaNs with 0
    feature_names = [f"feature_{i:02d}" for i in range(79)]
    data[feature_names] = data[feature_names].ffill().fillna(0)
    
    return data


class R2Score:
    def __init__(self):
        self.r2_metric = tf.keras.metrics.R2Score(class_aggregation='uniform_average')
        
    def reset(self):
        self.r2_metric.reset_states()
        
    def update(self, y_true, y_pred):
        # Convert PyTorch tensors to numpy arrays
        y_true = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
        y_pred = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        
        # Convert to TensorFlow tensors
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        
        # Update metric state
        self.r2_metric.update_state(y_true, y_pred)
        
    def compute(self):
        return self.r2_metric.result().numpy()

def evaluate_model(model, data_loader, device, criterion):
    """Evaluate model and return loss and R²"""
    model.eval()
    all_losses = []
    r2_metric = R2Score()
    
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            labels = labels.view(-1, 1)
            outputs = model(features)
            loss = criterion(outputs, labels)
            all_losses.append(loss.item())
            
            r2_metric.update(labels, outputs)
    
    avg_loss = np.mean(all_losses)
    r2 = r2_metric.compute()
    
    return avg_loss, r2

def train_model(model, train_loader, test_loader, device, epochs=1000):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)
    
    min_val_loss = float('inf')
    early_stop_count = 0
    best_r2 = -float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        train_r2_metric = R2Score()
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            labels = labels.view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            train_r2_metric.update(labels, outputs.detach())
        
        # Calculate training metrics
        train_loss = np.mean(train_losses)
        train_r2 = train_r2_metric.compute()
        
        # Validation phase
        val_loss, val_r2 = evaluate_model(model, test_loader, device, criterion)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping based on validation loss
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_r2 = val_r2
            early_stop_count = 0
        else:
            early_stop_count += 1
            
        if early_stop_count >= 5:
            print(f"Early stopping at epoch {epoch + 1}")
            print(f"Best validation R² achieved: {best_r2:.4f}")
            break
            
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train - Loss: {train_loss:.4f}, R²: {train_r2:.4f}")
        print(f"Valid - Loss: {val_loss:.4f}, R²: {val_r2:.4f}")
        print("-" * 50)
    
    return model


# Usage example
def main():
    # Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_names = [f"feature_{i:02d}" for i in range(79)]
    sequence_length = 10
    batch_size = 32
    
    # Load data
    train_path = "data/lags_features/training_parquet"
    train_data = load_data(train_path, start_id=501, end_id=507) 
    valid_data = load_data(train_path, start_id=508, end_id=510)
    # Create datasets
    train_dataset = FinancialDataset(train_data, feature_names, 'responder_6', sequence_length)
    valid_dataset = FinancialDataset(valid_data, feature_names, 'responder_6', sequence_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = TransformerModel(
        input_dim=len(feature_names),
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.2
    ).to(device)
    
    # Train model
    trained_model = train_model(model, train_loader, valid_loader, device)
    
    return trained_model

if __name__ == "__main__":
    main()