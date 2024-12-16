import os
import pickle
import numpy as np
import pandas as pd
import wandb
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Timer
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import polars

def set_random_seeds(seed=42):
    # Set the random seed for reproducibility
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Before creating and training your model, call the function
set_random_seeds(42)

class custom_args():
    def __init__(self):
        self.usegpu = True
        self.gpuid = 0
        self.seed = 42
        self.model = 'nn'
        self.use_wandb = False
        self.project = 'js-xs-nn-with-lags'
        self.dname = "./input_df/"
        self.loader_workers = 4
        self.bs = 8192
        self.lr = 1e-3
        self.weight_decay = 5e-4
        self.dropouts = [0.1, 0.1]
        self.n_hidden = [512, 512, 256]
        self.patience = 25
        self.max_epochs = 2000
        self.N_fold = 5

class CustomDataset(Dataset):
    def __init__(self, df, accelerator):
        self.features = torch.FloatTensor(df[feature_names].values).to(accelerator)
        self.labels = torch.FloatTensor(df[label_name].values).to(accelerator)
        self.weights = torch.FloatTensor(df[weight_name].values).to(accelerator)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        w = self.weights[idx]
        return x, y, w


class DataModule(LightningDataModule):
    def __init__(self, train_df, batch_size, valid_df=None, accelerator='cpu'):
        super().__init__()
        self.df = train_df
        self.batch_size = batch_size
        self.dates = self.df['date_id'].unique()
        self.accelerator = accelerator
        self.train_dataset = None
        self.valid_df = None
        if valid_df is not None:
            self.valid_df = valid_df
        self.val_dataset = None

    def setup(self, fold=0, N_fold=5, stage=None):
        # Split dataset
        selected_dates = [date for ii, date in enumerate(self.dates) if ii % N_fold != fold]
        df_train = self.df.loc[self.df['date_id'].isin(selected_dates)]
        self.train_dataset = CustomDataset(df_train, self.accelerator)
        if self.valid_df is not None:
            df_valid = self.valid_df
            self.val_dataset = CustomDataset(df_valid, self.accelerator)

    def train_dataloader(self, n_workers=0):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=n_workers)

    def val_dataloader(self, n_workers=0):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=n_workers)

# Custom R2 metric for validation
def r2_val(y_true, y_pred, sample_weight):
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    return r2


class NN(LightningModule):
    def __init__(self, input_dim, hidden_dims, dropouts, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        layers = []
        in_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            # layers.append(nn.BatchNorm1d(in_dim))
            if i > 0:
                layers.append(nn.SiLU())
            if i < len(dropouts):
                layers.append(nn.Dropout(dropouts[i]))
            layers.append(nn.Linear(in_dim, hidden_dim))
            # layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)
        self.lr = lr
        self.weight_decay = weight_decay
        self.validation_step_outputs = []

    def forward(self, x):
        return 5 * self.model(x).squeeze(-1)

    def training_step(self, batch):
        x, y, w = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y, reduction='none') * w  #
        loss = loss.mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch):
        x, y, w = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y, reduction='none') * w
        loss = loss.mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=x.size(0))
        self.validation_step_outputs.append((y_hat, y, w))
        return loss

    def on_validation_epoch_end(self):
        """Calculate validation WRMSE at the end of the epoch."""
        y = torch.cat([x[1] for x in self.validation_step_outputs]).cpu().numpy()
        if self.trainer.sanity_checking:
            prob = torch.cat([x[0] for x in self.validation_step_outputs]).cpu().numpy()
        else:
            prob = torch.cat([x[0] for x in self.validation_step_outputs]).cpu().numpy()
            weights = torch.cat([x[2] for x in self.validation_step_outputs]).cpu().numpy()
            # r2_val
            val_r_square = r2_val(y, prob, weights)
            self.log("val_r_square", val_r_square, prog_bar=True, on_step=False, on_epoch=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',}}

    def on_train_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        epoch = self.trainer.current_epoch
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in self.trainer.logged_metrics.items()}
        formatted_metrics = {k: f"{v:.5f}" for k, v in metrics.items()}
        print(f"Epoch {epoch}: {formatted_metrics}")


def load_data(path, start_dt, end_dt):
    data = polars.scan_parquet(path).select(
        polars.all(),
    ).filter(
        polars.col("date_id").gt(start_dt),
        polars.col("date_id").le(end_dt),
    ).fill_null(0)

    data = data.collect().to_pandas()

    data.replace([np.inf, -np.inf], 0, inplace=True)
    return data



train_path = f"/home/zt/pyProjects/JaneSt/Team/data/CustomData_1_RespLags/training"
validation_path = "/home/zt/pyProjects/JaneSt/Team/data/CustomData_1_RespLags/validation"
save_model_directory = '/home/zt/pyProjects/JaneSt/Team/scripts/George/1_HS_Plan/1_1_NN_NoBatchNorm'

feature_names = [f"feature_{i:02d}" for i in range(79)] + [f"responder_{idx}_lag_1" for idx in range(9)]
label_name = 'responder_6'
weight_name = 'weight'

df = load_data(train_path, 800, 1646)
valid = load_data(validation_path,1647, 1698)
my_args = custom_args()
args = my_args

# checking device
device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() and args.usegpu else 'cpu')
accelerator = 'gpu' if torch.cuda.is_available() and args.usegpu else 'cpu'
loader_device = 'cpu'


# Initialize Data Module
data_module = DataModule(df, batch_size=args.bs, valid_df=valid, accelerator=loader_device)

import gc
del df
gc.collect()
pl.seed_everything(args.seed)
for fold in range(0, 2):
    data_module.setup(fold, args.N_fold)
    # Obtain input dimension
    input_dim = data_module.train_dataset.features.shape[1]
    # Initialize Model
    model = NN(
        input_dim=input_dim,
        hidden_dims=args.n_hidden,
        dropouts=args.dropouts,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    # Initialize Logger
    if args.use_wandb:
        wandb_run = wandb.init(project=args.project, config=vars(args), reinit=True)
        logger = WandbLogger(experiment=wandb_run)
    else:
        logger = None
    # Initialize Callbacks
    early_stopping = EarlyStopping('val_loss', patience=args.patience, mode='min', verbose=False)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, verbose=False, filename=f"{save_model_directory}/nn_{fold}.model")
    timer = Timer()
    # Initialize Trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=[args.gpuid] if args.usegpu else None,
        logger=logger,
        callbacks=[early_stopping, checkpoint_callback, timer],
        enable_progress_bar=True
    )
    # Start Training
    trainer.fit(model, data_module.train_dataloader(args.loader_workers), data_module.val_dataloader(args.loader_workers))
    # You can find trained best model in your local path
    print(f'Fold-{fold} Training completed in {timer.time_elapsed("train"):.2f}s')
# test