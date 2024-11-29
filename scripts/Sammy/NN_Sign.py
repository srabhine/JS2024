import os
import polars as pl
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, classification_report
from tensorflow.keras import layers, models, optimizers, regularizers, callbacks
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def create_model(input_dim, lr, weight_decay):
    # Create a sequential model
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('swish'))
    model.add(layers.Dropout(0.1))  # Assuming dropouts[1] is valid
    model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('swish'))
    model.add(layers.Dropout(0.1))  # Assuming dropouts[1] is valid
    model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(weight_decay)))
    # Output layer
    model.add(layers.Dense(3, activation='softmax'))

    # Compile model with Mean Squared Error loss
    # model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mse', metrics=[WeightedR2()])
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def load_data(train_path,start_id,end_id):
    # df = pl.scan_parquet(f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/Lag_XGB/data/FOLD3").collect().to_pandas()
    folder_paths = [
        f"{train_path}/train_parquet_{partition_id}.parquet"
        for partition_id in range(start_id, end_id + 1)
    ]
    lazy_frames = [pl.scan_parquet(path) for path in folder_paths]
    combined_lazy_df = pl.concat(lazy_frames)

    data = combined_lazy_df.collect().to_pandas()

    data= data.ffill().fillna(0)
    return data

def decode_predictions(y_pred, encoder, classes):
    """
    Decode the one-hot encoded predictions back to original classes
    """
    ix = np.argmax(y_pred, axis=1)
    y_pred_decoded = np.zeros(len(ix))
    for i in range(len(ix)):
        y_pred_decoded[i] = classes[ix[i]]
    return y_pred_decoded


def evaluate_model(model, X_valid, y_valid_original, encoder, classes):
    """
    Evaluate the model and print various metrics
    """
    # Get predictions
    y_pred_proba = model.predict(X_valid)
    
    # Decode predictions
    y_pred_decoded = decode_predictions(y_pred_proba, encoder, classes)
    
    # Print evaluation metrics
    print("\nModel Evaluation:")
    print("----------------")
    print("Accuracy Score:", accuracy_score(y_valid_original, y_pred_decoded))
    print("\nDetailed Classification Report:")
    print(classification_report(y_valid_original, y_pred_decoded))
    
    return y_pred_decoded



train_path = "C:/Users/srabh/OneDrive/Documents/Jane_Street_Data_Challenge/data/transformed_data"
# train_path = "E:\Python_Projects\JS2024\GITHUB_C\data\\transformed_data"
model_saving_path = "C:/Users/srabh/OneDrive/Documents/Jane_Street_Data_Challenge/data/lags_features/models/nn_sign"
# model_saving_path = "E:\Python_Projects\JS2024\GITHUB_C\scripts\George\models\\2_base_model_trans_fet"
model_saving_name = "nn_sign_{epoch:02d}.keras"

feature_names = [f"feature_{i:02d}" for i in range(79)]
label_name = 'responder_6'
weight_name = 'weight'

data_train = load_data(train_path,start_id=6,end_id=7)
data_valid = load_data(train_path,start_id=8,end_id=9)

data_train = data_train[data_train['symbol_id']==1]
data_valid = data_valid[data_valid['symbol_id']==1]


encoder = OneHotEncoder(handle_unknown='error')
classes = [-1, 0, 1]
enc = LabelEncoder()
encoder.fit(np.array(classes).reshape(-1, 1))

X_train = data_train[ feature_names ]
y_train = data_train[ label_name    ]
y_train= np.sign(y_train)
y_train_original = y_train.copy() 
Y_train = encoder.transform(y_train.values.reshape(-1,1)).toarray()


w_train = data_train[ "weight"      ]
del data_train

X_valid = data_valid[ feature_names ]
y_valid = data_valid[ label_name    ]
y_valid=np.sign(y_valid)
y_valid_original = y_valid.copy() 
Y_valid = encoder.transform(y_valid.values.reshape(-1,1)).toarray()
w_valid = data_valid[ "weight"      ]
del data_valid

y_s = np.sign(y_train)



lr = 0.01
weight_decay = 1e-6
input_dimensions = X_train.shape[1]
model = create_model(input_dimensions, lr, weight_decay)


ca = [
    tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=25, mode='max'),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=f'{model_saving_path}/{model_saving_name}',
        monitor='val_loss', save_best_only=False),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',  # Metric to be monitored
        factor=0.1,  # Factor by which the learning rate will be reduced
        patience=10,  # Number of epochs with no improvement after which learning rate will be reduced
        verbose=1,  # Verbosity mode
        min_lr=1e-6  # Lower bound on the learning rate
    )

]



model.fit(
    x=X_train,  # Input features for training
    y=Y_train,                          # Target labels for training
    sample_weight=w_train,              # Sample weights for training
    validation_data=(X_valid, Y_valid, w_valid),  # Validation data
    batch_size=8029,                      # Batch size
    epochs=100,                        # Number of epochs
    callbacks=ca,                # Callbacks list, if any
    verbose=1,                           # Verbose output during training
    shuffle=True
)

# Evaluate the model
print("\nEvaluating model performance...")
y_pred_decoded = evaluate_model(model, X_valid, y_valid_original, encoder, classes)




