import pandas as pd


def load_traces(data_path, nrows=None, shuffle=False):
    """Load a trace CSV and split into train/val/test.

    Expected columns:
        - pc
        - delta_in

    Type normalization:
        - delta_in is converted to int.
        - pc is converted to int (hex).

    Splits:
        60% train / 10% val / 30% test.
    """
    traces = pd.read_csv(data_path, nrows=nrows)

    # Turn delta_in from decimal strings into int.
    traces["delta_in"] = pd.to_numeric(traces["delta_in"], errors="raise").astype(int)

    # Turn PCs from hex strings into int.
    pc_series = traces["pc"].astype(str).str.strip()
    traces["pc"] = pc_series.map(lambda x: int(x, 16)).astype(int)

    # Shuffle is always going to be false.
    if shuffle:
        traces = traces.sample(frac=1)

    dataset_length = len(traces)

    # 60% training, 10% validation, 30% testing
    train_end = int(dataset_length * 0.6)
    val_end = int(dataset_length * 0.7)

    train_data = traces.iloc[:train_end]
    val_data = traces.iloc[train_end:val_end]
    test_data = traces.iloc[val_end:]

    print("Training data length: ", len(train_data))
    print("Validation data length: ", len(val_data))
    print("Testing data length: ", len(test_data))

    return train_data, val_data, test_data