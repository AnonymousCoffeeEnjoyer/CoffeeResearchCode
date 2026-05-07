# Coffee: Going beyond the next hit for OS prefetching

This repository is the official implementation of [Coffee: Going beyond the next hit for OS prefetching](https://arxiv.org/abs/2030.12345).  (Need to update the link here)

## Requirements

To install requirements in a virtual environment using Bash on Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ./requirements.txt
```

<!--
Original Windows PowerShell install command:

```powershell
python -m pip install -r .\requirements.txt
```
-->

Datasets:

The datasets are available from [GitHub Releases?](https://example.com). There are seven datasets and each have two CSVs containing access pattern traces: `25_filtered_delta.csv` and `50_filtered_delta.csv`.
The distinction is explained in [Coffee](https://example.com).
Each dataset should at least have these columns:

- `pc` (hex string)
- `delta_in` (integer)

(Need to update this part)

All training scripts take:

- `trace_csv_path`: path to input CSV (example: `./dataset/25_filtered_delta.csv`)
- `output_dir`: directory where logs/checkpoints are written (this is created if it is missing)

## Training

Training and validation are done using the same command. If the provided output directory already contains a model, then it is used for validation. Otherwise, a new model is trained and then validated.
Run commands from the repository root (`CoffeeResearchCode`).

<!--
### LSTM models (PowerShell)

```powershell
# Direct next-N baseline
python .\next_n_lstm\train_lstm.py .\dataset\25_filtered_delta.csv .\outputLSTM

# Autoregressive rollout model
python .\next_n_lstm_autoregressive\train.py .\dataset\25_filtered_delta.csv .\outputLSTMAuto

# Direct multi-step model
python .\next_n_lstm_direct_multi-step\train.py .\dataset\25_filtered_delta.csv .\outputLSTMMulti
```

### Pattention models (PowerShell)

```powershell
# Original Pattention baseline
python .\next_n_Pattention\train.py .\dataset\25_filtered_delta.csv .\outputPatten

# Autoregressive rollout model
python .\next_n_Pattention_autoregressive\train.py .\dataset\25_filtered_delta.csv .\outputPattenAuto

# Direct multi-step model
python .\next_n_Pattention_direct_multi-step\train.py .\dataset\25_filtered_delta.csv .\outputPattenMulti
```
-->

### LSTM models (Linux/macOS bash)

```bash
source .venv/bin/activate # Only necessary if using a new Bash shell.

# Autoregressive rollout model
python3 ./next_n_lstm_autoregressive/train.py ./dataset/25_filtered_delta.csv ./outputLSTMAuto

# Direct multi-step model
python3 ./next_n_lstm_direct_multi-step/train.py ./dataset/25_filtered_delta.csv ./outputLSTMMulti
```

### Pattention models (Linux/macOS bash)

```bash
source .venv/bin/activate # Only necessary if using a new Bash shell.

# Autoregressive rollout model
python3 ./next_n_Pattention_autoregressive/train.py ./dataset/25_filtered_delta.csv ./outputPattenAuto

# Direct multi-step model
python3 ./next_n_Pattention_direct_multi-step/train.py ./dataset/25_filtered_delta.csv ./outputPattenMulti
```

## Outputs

Each run writes to its output directory:

- `output.txt` training/validation/test logs
- model checkpoints (`*.pt`)
- label encodings (`*.json`)

The top-1-next-n accuracies are reported near the bottom of `output.txt` as a scalar tensor value between 0 and 1.

## License

MIT License

Copyright (c) 2026 Anonymous Authors

<!-- Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. -->
