# Coffee: Going beyond the next hit for OS prefetching

This repository is the official implementation of [Coffee: Going beyond the next hit for OS prefetching](https://arxiv.org/abs/2030.12345).  (Need to update the link here)

<!-- >📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials -->

## Requirements

To install requirements:

```bash
python3 -m pip install -r ./requirements.txt
```

<!--
Original Windows PowerShell install command:

```powershell
python -m pip install -r .\requirements.txt
```
-->

<!-- >📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc... -->

Datasets:

(Need to update this part)


All training scripts take:

- `trace_csv_path`: path to input CSV (example: `./dataset/25_filtered_delta.csv`)
- `output_dir`: directory where logs/checkpoints are written (this is created if it is missing)

Expected CSV columns:

- `pc` (hex string)
- `delta_in` (integer)

## Training

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
# Autoregressive rollout model
python3 ./next_n_lstm_autoregressive/train.py ./dataset/25_filtered_delta.csv ./outputLSTMAuto

# Direct multi-step model
python3 ./next_n_lstm_direct_multi-step/train.py ./dataset/25_filtered_delta.csv ./outputLSTMMulti
```

### Pattention models (Linux/macOS bash)

```bash
# Autoregressive rollout model
python3 ./next_n_Pattention_autoregressive/train.py ./dataset/25_filtered_delta.csv ./outputPattenAuto

# Direct multi-step model
python3 ./next_n_Pattention_direct_multi-step/train.py ./dataset/25_filtered_delta.csv ./outputPattenMulti
```

## Outputs

Each run writes to its output directory:

- `output.txt` training/validation/test logs
- model checkpoints (`*.pt`)
- label encoder artifacts (where applicable)

<!-- ## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z.

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models. -->

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it.

## Contributing


(I'm not sure what to put in full name)

MIT License

Copyright (c) 2026 [fullname]

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

