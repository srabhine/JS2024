import os
import polars as pl
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

def set_random_seeds(seed=42):
    # Set the random seed for reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Before creating and training your model, call the function
set_random_seeds(42)


def load_data(path, start_dt, end_dt):
    data = pl.scan_parquet(path).select(
        pl.all(),).filter(
        pl.col("date_id").gt(start_dt),
        pl.col("date_id").le(end_dt),
    ).fill_null(0).fill_null(0)

    data = data.collect().to_pandas()

    data.replace([np.inf, -np.inf], 0, inplace=True)
    return data

# def create_sequences(data, sequence_length):
#     sequences = []
#     labels = []
#
#     for i in range(len(data) - sequence_length):
#         sequence = data[i:i + sequence_length]
#         label = data[i + sequence_length]
#         sequences.append(sequence)
#         labels.append(label)
#
#     return np.array(sequences), np.array(labels)

def create_sequences(data, target, time_steps=7):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(target[i])
    return np.array(X), np.array(y)


def process_group(group, time_steps=20):
    # Sort the group if necessary
    group = group.sort_values(by=['date_id', 'time_id']).reset_index(drop=True)
    
    # Calculate the rolling standard deviation
    group['std'] = group['responder_6_lag_1'].shift(1).rolling(window=100).std().fillna(0)
    
    X_list, y_list, meta_list = [], [], []
    
    # Create sequences
    for i in tqdm(range(len(group) - time_steps)):
        X = group[['responder_6_lag_1', 'std']].iloc[i:(i + time_steps)].values
        y = group['responder_6'].iloc[i + time_steps]
        
        # Collect metadata just for the prediction point
        symbol_id = group['symbol_id'].iloc[i + time_steps]
        date_id = group['date_id'].iloc[i + time_steps]
        time_id = group['time_id'].iloc[i + time_steps]
        
        X_list.append(X)
        y_list.append(y)
        meta_list.append({'symbol_id': symbol_id, 'date_id': date_id, 'time_id': time_id})
    
    return pd.DataFrame({'X': X_list, 'y': y_list, 'meta': meta_list})

is_linux = False
if is_linux:
    path = f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/data/jane-street-real-time-market-data-forecasting/train.parquet"
    training_resp_lag_path = "/home/zt/pyProjects/JaneSt/Team/data/CustomData_2_RespLags_onTime/trainData"
    model_saving_path = "/home/zt/pyProjects/JaneSt/Team/scripts/George/1_HS_Plan/Torch-NN-models"
else:
    training_resp_lag_path = "E:\Python_Projects\JS2024\GITHUB_C\data\CustomData_2_RespLags_onTime\\trainData"


feature_names = [f"feature_{i:02d}" for i in range(79)]
responder_lags = [f"responder_{idx}_lag_1" for idx in range(9)]
label_name = 'responder_6'
weight_name = 'weight'


data = load_data(training_resp_lag_path, start_dt=1600, end_dt=1698)


data = data[['date_id', 'time_id', 'symbol_id', 'responder_6','responder_6_lag_1']]
results = data.groupby('symbol_id').apply(lambda group: process_group(group, time_steps=20)).reset_index(drop=True)
# Access the metadata, X and y if required
X = np.concatenate(results['X'].values)
y = results['y'].values
all_meta = results['meta'].tolist()





seq_length = 20  # Length of each sequence
num_features = 2  # Number of features per time step
embedding_size = 10  # Size of embedding vector


sequence_input = Input(shape=(seq_length, num_features), name='sequence_input')

symbol_embedding = Embedding(input_dim=len(label_encoder.classes_), output_dim=embedding_size)(symbol_input)
lstm_out = LSTM(50, activation='relu')(sequence_input)
concat = Concatenate()([lstm_out, tf.squeeze(symbol_embedding, axis=1)]) # [N, dim]

symbol_out = []
for i in range(39):
    target_symbol_input = tf.where(tf.equal(input_tensor["symbol_id"], tf.cast(tf.ones_like(input_tensor["symbol_id"])*i, tf.string)), concat, tf.zeros_like(concat))
    target_symbol_out = Dense(units=32, activation='swish', name="symbol_"+str(i))(target_symbol_input) # [N, 32]
    symbol_out.append(target_symbol_out)
symbol_out_combined = tf.stack(symbol_out, axis=-1) # [N, 32, 1]
symbol_out_combined = tf.reduce_sum(symbol_out_combined, axis=-1) # [N ,32]
concat = symbol_out_combined

output = Dense(1)(concat)
model = Model(inputs=[sequence_input, symbol_input], outputs=output)
# n