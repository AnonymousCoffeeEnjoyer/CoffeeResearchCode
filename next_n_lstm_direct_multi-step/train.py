import os
import sys

import torch

try:
    from .config import config, hparams, path_keeper
    from .load_trace import load_traces
    from .label_encoder import encode_data, save_label_encoders
    from .model import EmbeddingLSTM
    from .validation import evaluate_model, print_eval_summary
except ImportError:
    from config import config, hparams, path_keeper
    from load_trace import load_traces
    from label_encoder import encode_data, save_label_encoders
    from model import EmbeddingLSTM
    from validation import evaluate_model, print_eval_summary


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python train_lstm.py <trace_csv_path> <output_dir>")
        return 1

    filepath = os.path.abspath(sys.argv[1])
    targetpath = os.path.abspath(sys.argv[2])
    os.makedirs(targetpath, exist_ok=True)

    print("Absolute path to trace file: ", filepath)
    print("Absolute path to target output dir: ", targetpath)

    output_path = os.path.join(targetpath, "output.txt")
    original_stdout = sys.stdout
    try:
        with open(output_path, "w", buffering=1, encoding="utf-8") as result_output:
            sys.stdout = result_output
            print("Redirected output to ", output_path)

            path_keeper["targetpath"] = targetpath
            path_keeper["filepath"] = filepath

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using {device} device")

            train_data, val_data, test_data = load_traces(filepath, shuffle=False)
            train_iter, val_iter, test_iter, _ = encode_data(train_data, val_data, test_data)

            enc_path = save_label_encoders(targetpath)
            print(f"Saved label encoders to {enc_path}")

            model = EmbeddingLSTM(
                num_delta_in=int(config["num_delta_in"]),
                num_pc_in=int(config["num_pc"]),
                embed_dim=int(hparams["delta_embed_dim"]),
                pc_embed_dim=int(hparams["pc_embed_dim"]),
                hidden_dim=int(hparams["hidden_dim"]),
                num_layers=int(hparams["num_layers"]),
                dropout=float(hparams["dropout"]),
                max_steps=int(config["next_n_predict"]),
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=float(hparams["learning_rate"]))
            lr_decay = float(hparams.get("lr_decay", 1.0))
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

            epochs = int(hparams["epochs"])

            top_k = int(config.get("top_k_pred", 1))
            horizon = int(config["next_n_predict"])
            best_val = float("inf")
            best_path = os.path.join(targetpath, "LSTM_model_best_val.pt")

            for epoch in range(epochs):
                model.train()
                train_loss_sum = 0.0
                train_steps = 0

                for pcs, deltas, delta_targets, _pc_targets in train_iter:
                    deltas = deltas.to(device)
                    pcs = pcs.to(device)
                    delta_targets = delta_targets.to(device)

                    # Direct next-N supervision.
                    _, _, loss = model(deltas, pcs, steps=horizon, targets=delta_targets)
                    if loss is None:
                        continue

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    train_loss_sum += float(loss.item())
                    train_steps += 1

                avg_train_loss = train_loss_sum / max(1, train_steps)
                print(f"Epoch {epoch + 1} Train loss: {avg_train_loss:.6f}")

                val_losses, val_acc = evaluate_model(model, val_iter, computing_device=device)
                print_eval_summary(f"Epoch {epoch + 1} Validation:", val_losses, val_acc, top_k=top_k)

                v = val_losses.get("nextn")
                if v is not None and float(v) < best_val:
                    best_val = float(v)
                    torch.save(model.state_dict(), best_path)
                    print(
                        f"Epoch {epoch + 1}: saved best checkpoint (nextn={best_val:.6f}) to {best_path}"
                    )

                # Decay learning rate once per epoch.
                scheduler.step()

            model_path = os.path.join(targetpath, "LSTM_model.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Saved model to {model_path}")

            test_losses, test_acc = evaluate_model(model, test_iter, computing_device=device)
            print_eval_summary("Test:", test_losses, test_acc, top_k=top_k)

            return 0
    finally:
        # Ensure interpreter shutdown doesn't try to flush a closed file.
        sys.stdout = original_stdout


if __name__ == "__main__":
    raise SystemExit(main())
