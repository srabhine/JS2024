import numpy as np
import pandas as pd
import polars as pl
from typing import Optional, List, Union, Tuple
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from libs.io_lib.paths import LAGS_FEATURES_TRAIN, LAGS_FEATURES_VALID
from libs.math_lib.core import r2_zero
from libs.one_big_lib import stack_features_by_sym, FEATS, TARGET, SYMBOLS
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
from pathlib import Path


def create_sequences(X: pd.DataFrame,
                     y: Optional[pd.Series] = None,
                     sequence_length: int = 10,
                     stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM from time series data.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features dataframe with MultiIndex (date_id, time_id)
    y : pd.Series
        Target values aligned with X
    sequence_length : int
        Length of sequences to create
    stride : int
        Step size between sequences
        
    Returns:
    --------
    X_seq : np.ndarray 
        Shape (n_samples, sequence_length, n_features)
    y_seq : np.ndarray
        Shape (n_samples,)
    """
    # Convert to numpy for easier manipulation
    X_values = X.values
    if y is not None:
        y_values = y.values
    else:
        y_values = None
        y_seq = None
    
    n_samples = (len(X_values) - sequence_length) // stride + 1
    n_features = X_values.shape[1]
    
    # Pre-allocate arrays
    X_seq = np.zeros((n_samples, sequence_length, n_features))
    if y is not None:
        y_seq = np.zeros(n_samples)
    
    # Create sequences
    for i in range(n_samples):
        start_idx = i * stride
        end_idx = start_idx + sequence_length
        X_seq[i] = X_values[start_idx:end_idx]
        if y is not None:
            y_seq[i] = y_values[end_idx - 1]  # Target is the last value in sequence
        
    return X_seq, y_seq



# Load data
df_all = pl.scan_parquet(LAGS_FEATURES_TRAIN).collect().to_pandas()
vld_all = pl.scan_parquet(LAGS_FEATURES_VALID).collect().to_pandas()

# df = df_all.query('date_id>=1500')
df = df_all.query('date_id>=1645')
df_by_sym = stack_features_by_sym(df)


# Utilisation de la fonction
sequence_length = 10  # Longueur des séquences
stride = 1  # Pas entre chaque séquence

# As long as this is squared (df_by_sym, then we are fine)

# Train: do locally
kwargs = {}
path = ''
suffix = 'LSTM'
models = {s: None for s in SYMBOLS}
for s in SYMBOLS:
    print(f'Symbol {s}')
    df_sym = df_by_sym.swaplevel(axis=1)[s].ffill()
    X = df_sym[FEATS]
    y = df_sym[TARGET]
    X_sequences, y_sequences = create_sequences(X, y, sequence_length,
                                                stride)
    if s == SYMBOLS[0]:
        models[s] = create_model()
        models[s].fit(X_sequences, y_sequences, **kwargs)
        # Try your setup_an_train
        model_name = f'{suffix}_{s}'
        models[s].save(path + model_name)
    else:
        models[s] = create_model()
        models[s].save(path + model_name)


# Submission in predict() function

# Mimic submission (exclude this)
# From here
cols = ['time_id', 'date_id', 'weight', 'is_', 'symbol_id'] + \
       FEATS + [TARGET]
df_tmp = pd.DataFrame(np.zeros((len(SYMBOLS), len(cols))),
                      columns=cols)
df_tmp2 = df_tmp[['time_id', 'date_id', 'symbol_id', 'weight'] +
                 FEATS + [TARGET]]
df_tmp2['symbol_id'] = range(len(SYMBOLS))
test_df = df_tmp2.copy()
test_df['symbol_id'] = test_df['symbol_id'] + 1
# this will remove one symbol and add a new one


test_df_stacked = stack_features_by_sym(test_df)
df_by_sym_test = pd.concat((df_by_sym_test.iloc[-sequence_length:, :],
                     test_df_stacked), axis=0).fillna(0.0)
# Careful here
# Joining should be fine, but we might get NaN's

# Pre-submission
global df_by_sym_test
df_by_sym_test = df_by_sym.iloc[-sequence_length-1:]

# Predict step
# y_pred initialized as usual
# TODO:
y_pred = pd.Series(0, index=test_df['symbol_id']) # BUT CHANGE with correct symbols

X = df_by_sym_test[FEATS]
y = df_by_sym_test[TARGET]
X_swap = X.swaplevel(axis=1)
symbols_tmp = X_swap.columns.droplevel(1).unique()
for s in SYMBOLS:
    if s in symbols_tmp and s in test_df['symbol_id']:
        X_tmp = X_swap[s]
        X_sequences, _ = create_sequences(X_swap,None,
                                          sequence_length, stride)
        y_pred[s] = models[s].predict(X_sequences[-1].reshape(1,
                                                               sequence_length,
                                                               -1))


# Up to here we test locally
# ------------------------------------------------------------

# # Create indices for submission
# X = df_by_sym[FEATS]
# y = df_by_sym[TARGET]
# X_swap = X.swaplevel(axis=1)
# cols = X_swap.columns
# map_cols = {X_swap.columns.get_loc((s, n)): (s, n)  for s in SYMBOLS
#             for n
#             in FEATS}
# inv_map_cols = {v:k for (k, v) in map_cols.items()}
# ix_by_sym = {}
# for s in SYMBOLS:
#     ix_by_sym[s] = []
#     for n in FEATS:
#         ix_by_sym[s].append(inv_map_cols[(s, n)])

# # Example
# X_sequences[:, :, ix_by_sym[34]].shape


# Pre-submission
global df_by_sym_test
df_by_sym_test = df_by_sym.iloc[-sequence_length-1:]



# test_df = pd.DataFrame()  # index=int, len=len(symbols),
# # columns=features + other stuff
#
# # Mimic submission (exclude this)
#
# # From here
# cols = ['time_id', 'date_id', 'weight', 'is_', 'symbol_id'] + \
#        FEATS + [TARGET]
# df_tmp = pd.DataFrame(np.zeros((len(SYMBOLS), len(cols))),
#                       columns=cols)
# df_tmp2 = df_tmp[['time_id', 'date_id', 'symbol_id', 'weight'] +
#                  FEATS + [TARGET]]
# df_tmp2['symbol_id'] = range(len(SYMBOLS))
# test_df = df_tmp2.copy()
# # To here


# Submission in predict() function
test_df_stacked = stack_features_by_sym(test_df)
df_by_sym_test = pd.concat((df_by_sym_test.iloc[-sequence_length:, :],
                     test_df_stacked), axis=0).fillna(0.0)
# Careful here
# Joining should be fine, but we might get NaN's

# Predict step
# y_pred initialized as usual
# TODO:
y_pred = pd.Series(0, index=test_df['symbol_id']) # BUT CHANGE with correct symbols

X = df_by_sym_test[FEATS]
y = df_by_sym_test[TARGET]
X_swap = X.swaplevel(axis=1)
symbols_tmp = X_swap.columns.droplevel(1).unique()
for s in SYMBOLS:
    if s in symbols_tmp and s in test_df['symbol_id']:
        X_tmp = X_swap[s]
        X_sequences, _ = create_sequences(X_swap,None,
                                          sequence_length, stride)
        y_pred[s] = models[s].predict(X_sequences[-1].reshape(1,
                                                               sequence_length,
                                                               -1))


# Example of submission
# df_tmp = df_by_sym.iloc[-1:, :]
# df_tmp = df_tmp.reset_index()


# test_df_stacked = stack_features_by_sym(test_df)
#
# df_tmp3 = pd.concat((df_by_sym.iloc[-sequence_length:, :],
#                      test_df_stacked), axis=0)
#
# model.predict(df_tmp3.values[])

# feature_names = FEATS
# cols = IX_IDS_BY_SYM + ['weight'] + feature_names + [TARGET]
# # cols = check_cols(cols, data_all)
# data_by_sym_tmp = df_tmp2[cols]
# data_by_sym_tmp.set_index(IX_IDS_BY_SYM, append=True, drop=True,
#                       inplace=True)
# data_by_sym_tmp = data_by_sym_tmp.droplevel(0, axis='index')
# data_by_sym_tmp = data_by_sym_tmp.unstack(level=['symbol_id'])
# data_by_sym_tmp.ffill().fillna(0)
#
# stack_features_by_sym(df_tmp)



def set_seed(seed: int = 42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, path: str, epoch: int, optimizer, loss: float):
    """
    Save model checkpoint with additional training information.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to save
    path : str
        Path to save the model
    epoch : int
        Current epoch number
    optimizer : torch.optim.Optimizer
        The optimizer used in training
    loss : float
        Current loss value
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")

def load_model(model, optimizer, path: str):
    """
    Load model checkpoint and return the epoch and loss.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model architecture to load weights into
    optimizer : torch.optim.Optimizer
        The optimizer to load state into
    path : str
        Path to the saved model
        
    Returns:
    --------
    epoch : int
        The epoch number when the model was saved
    loss : float
        The loss value when the model was saved
    """
    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
            self.counter = 0



# Dataset personnalisé
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Modèle LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Get last output
        last_output = lstm_out[:, -1, :]
        
        # Pass through fully connected layer
        out = self.fc(last_output)
        return out.squeeze()

# Fonction d'entraînement
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass et optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

# Préparation des données
def prepare_data(X_sequences, y_sequences, batch_size=32, train_split=0.8):
    # Conversion en tenseurs et création des datasets
    dataset = SequenceDataset(X_sequences, y_sequences)
    
    # Split train/validation
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Création des dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader


def calculate_r2(model, data_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    from sklearn.metrics import r2_score
    return r2_score(all_targets, all_preds)



def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, 
                save_dir: str = 'models', patience: int = 10):
    """Training function with early stopping and best model saving"""
    train_losses = []
    val_losses = []
    train_r2s = []
    val_r2s = []
    
    # Create save directory if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        # Calculate average losses
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        
        # Calculate R²
        train_r2 = calculate_r2(model, train_loader, device)
        val_r2 = calculate_r2(model, val_loader, device)
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_r2s.append(train_r2)
        val_r2s.append(val_r2)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, R²: {train_r2:.4f}')
        print(f'Val Loss: {val_loss:.4f}, R²: {val_r2:.4f}')
        print('-' * 50)
        
        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            # Load best model
            model.load_state_dict(early_stopping.best_model)
            break
    
    # Save final (best) model
    save_path = save_dir / 'best_model.pth'
    save_model(model, str(save_path), epoch, optimizer, early_stopping.best_loss)
    print(f"Best model saved with validation loss: {early_stopping.best_loss:.4f}")
    
    return train_losses, val_losses, train_r2s, val_r2s

# Configuration et entraînement

# 2/3 functions: setup, train, and predict

def setup_and_train(X_sequences, y_sequences, seed: int = 42):
    # Set seed for reproducibility
    set_seed(seed)
    
    input_size = X_sequences.shape[2]
    hidden_size = 64
    num_layers = 2
    num_epochs = 100
    batch_size = 32
    learning_rate = 0.001
    patience = 10  # Early stopping patience
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = prepare_data(X_sequences, y_sequences, batch_size)
    
    model = LSTMModel(input_size, hidden_size, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    save_dir = 'C:/Users/srabh/OneDrive/Documents/Jane_Street_Data_Challenge/scripts/Sammy/models'
    
    return model, *train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        num_epochs, 
        device,
        save_dir= save_dir,
        patience=patience
    )




# Entrainement et visualisation
model, train_losses, val_losses, train_r2s, val_r2s = setup_and_train(X_sequences, y_sequences, seed=42)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Visualiser les résultats
plt.figure(figsize=(15, 5))

# Plot des pertes
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Plot des R²
plt.subplot(1, 2, 2)
plt.plot(train_r2s, label='Training R²')
plt.plot(val_r2s, label='Validation R²')
plt.xlabel('Epoch')
plt.ylabel('R²')
plt.title('Training and Validation R²')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Afficher les métriques finales
print(f'Final Metrics:')
print(f'Training Loss: {train_losses[-1]:.4f}, R²: {train_r2s[-1]:.4f}')
print(f'Validation Loss: {val_losses[-1]:.4f}, R²: {val_r2s[-1]:.4f}')



















'''
# Vérification des résultats
print(f"Shape des séquences X: {X_sequences.shape}")  # (n_samples, sequence_length, n_features)
print(f"Shape des séquences y: {y_sequences.shape}")  # (n_samples,)

# Vérification d'une séquence
print("\nExemple de la première séquence:")
print("Premier timestep:", X_sequences[0, 0])
print("Deuxième timestep:", X_sequences[0, 1])
print("Target:", y_sequences[0])
'''

