config = {
    "sequence_length": 16,
    "next_n_predict": 8,
    "top_k_pred": 10,
}

hparams = {
    # Training / runtime hyperparameters
    "batch_size": 256,
    "delta_embed_dim": 128,
    "pc_embed_dim": 128,
    "hidden_dim": 256,
    "num_heads": 8,
    "dropout": 0.1,
    "learning_rate": 0.001,
    "pc_loss_weight": 1.0,
    # Exponential LR decay per epoch (gamma). Set to 1.0 to disable decay.
    "lr_decay": 0.9,
    "epochs": 20,
}

path_keeper = {
    'targetpath': None
}