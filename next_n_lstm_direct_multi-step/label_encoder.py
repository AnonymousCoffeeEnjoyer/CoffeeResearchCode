import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
try:
    from .config import config, hparams
except ImportError:
    from config import config, hparams
from sklearn.preprocessing import LabelEncoder

label_encoder_pc = LabelEncoder()
label_encoder_deltas = LabelEncoder()

batch_size = hparams['batch_size']
sequence_length = config['sequence_length']

class PrefetchingDataset(Dataset):
    def __init__(self, pc, delta_in):
        self.pcs = pc
        self.delta_in = delta_in
       
    def __len__(self):
        # Number of valid starting positions for a window of length
        # N - sequence_length - next_n_predict + 1
        return max(0, len(self.delta_in) - sequence_length - config['next_n_predict'] + 1)

    def __getitem__(self, idx):
        pcs = self.pcs[idx:idx+sequence_length].long()
        deltas = self.delta_in[idx:idx+sequence_length].long()

        # Target future deltas and PCs for the next n steps 
        delta_targets = (
            self.delta_in[idx+sequence_length: idx+sequence_length + config['next_n_predict']]
            .long()
        )
        pc_targets = self.pcs[idx+sequence_length: idx+sequence_length + config['next_n_predict']].long()

        return (pcs, deltas, delta_targets, pc_targets)

def load_data(data, batch_size, drop_last):
    data = data.copy() 

    # Create column of encoded pc and delta_in
    data.loc[:, 'pc_encoded'] = label_encoder_pc.transform(data['pc'].values)
    data.loc[:, 'delta_in_encoded'] = label_encoder_deltas.transform(data['delta_in'].values)
    pc = torch.tensor(data['pc_encoded'].values)
    delta_in = torch.tensor(data['delta_in_encoded'].values)
    dataset = PrefetchingDataset(pc, delta_in)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last)
    
    # Get unique target keys
    target_keys = set(data['delta_in'].unique())
    
    # Return (data_loader, num_pc, num_delta_in, num_output_next, target_keys)
    return data_loader, len(label_encoder_pc.classes_), len(label_encoder_deltas.classes_), len(label_encoder_deltas.classes_), target_keys

def encode_data(train_data, val_data, test_data):
    # Fit encoders on the full dataset (train + val + test) for a fixed vocab
    pc_vocab = sorted(set().union(train_data['pc'].values, val_data['pc'].values, test_data['pc'].values))
    delta_vocab = sorted(
        set().union(train_data['delta_in'].values, val_data['delta_in'].values, test_data['delta_in'].values)
    )
    label_encoder_pc.fit(pc_vocab)
    label_encoder_deltas.fit(delta_vocab)

    # For training we keep drop_last=True to maintain a fixed batch size,
    # for validation/test we allow smaller final batches.
    train_iter, num_pc, num_delta_in, num_output_next, target_keys = load_data(train_data, batch_size=batch_size, drop_last=True)
    val_iter, _, _, _, _ = load_data(val_data, batch_size=batch_size, drop_last=False)
    test_iter, _, _, _, _ = load_data(test_data, batch_size=batch_size, drop_last=False)
    config['num_pc'] = num_pc
    config['num_delta_in'] = num_delta_in
    
    print('number of unique pc: ', num_pc)
    print('number of unique input delta: ', num_delta_in)
    return train_iter, val_iter, test_iter, target_keys


def save_label_encoders(output_dir: str, filename: str = "label_encoders.json") -> str:
    """Save LabelEncoder vocabularies to JSON.
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    payload = {
        "pc_classes": label_encoder_pc.classes_.tolist(),
        "delta_in_classes": label_encoder_deltas.classes_.tolist(),
    }
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return out_path


def load_label_encoders(path: str) -> None:
    """Load LabelEncoder vocabularies from JSON file."""
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if "pc_classes" not in payload or "delta_in_classes" not in payload:
        raise ValueError(f"Invalid encoder file (missing keys): {path}")

    # Use numpy arrays because sklearn LabelEncoder expects numpy-like classes_.
    label_encoder_pc.classes_ = np.asarray(payload["pc_classes"])
    label_encoder_deltas.classes_ = np.asarray(payload["delta_in_classes"])