import os
import sys

import torch

try:
    from .config import config, hparams, path_keeper
    from .load_trace import load_traces
    from .label_encoder import encode_data, save_label_encoders
    from .model import EmbeddingPattention
    from .validation import evaluate_model, print_eval_summary
except ImportError:
    from config import config, hparams, path_keeper  # pylint: disable=import-error
    from load_trace import load_traces  # pylint: disable=import-error
    from label_encoder import encode_data, save_label_encoders  # pylint: disable=import-error
    from model import EmbeddingPattention  # pylint: disable=import-error
    from validation import evaluate_model, print_eval_summary  # pylint: disable=import-error


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python train.py <trace_csv_path> <output_dir>")
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

            model = EmbeddingPattention(
                num_pc=int(config["num_pc"]),
                num_delta_in=int(config["num_delta_in"]),
                delta_embed_dim=int(hparams["delta_embed_dim"]),
                pc_embed_dim=int(hparams["pc_embed_dim"]),
                hidden_dim=int(hparams["hidden_dim"]),
                num_heads=int(hparams["num_heads"]),
                dropout=float(hparams["dropout"]),
                pc_loss_weight=float(hparams.get("pc_loss_weight", 1.0)),
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=float(hparams["learning_rate"]))
            lr_decay = float(hparams.get("lr_decay", 1.0))
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

            epochs = int(hparams["epochs"])

            top_k = int(config.get("top_k_pred", 1))

            best_val_loss = float("inf")
            best_model_path = os.path.join(targetpath, "Pattention_model_best.pt")

            for epoch in range(epochs):
                model.train()
                train_loss_sum = 0.0
                train_steps = 0

                for pcs, deltas, delta_targets, pc_targets in train_iter:
                    pcs = pcs.to(device)
                    deltas = deltas.to(device)
                    delta_targets = delta_targets.to(device)
                    pc_targets = pc_targets.to(device)

                    # Next-1 supervision only.
                    delta_t1 = delta_targets[:, 0]
                    pc_t1 = pc_targets[:, 0]

                    _, _, loss = model(pcs, deltas, delta_target=delta_t1, pc_target=pc_t1)
                    if loss is None:
                        continue

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    train_loss_sum += float(loss.item())
                    train_steps += 1

                val_losses, val_acc = evaluate_model(model, val_iter, computing_device=device)
                print_eval_summary(f"Epoch {epoch + 1} Validation:", val_losses, val_acc, top_k=top_k)

                # Save best checkpoint by validation next-1 total CE (delta + weighted pc).
                current_val = float(val_losses.get("next1_total", 0.0))
                if current_val < best_val_loss:
                    best_val_loss = current_val
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Saved best model to {best_model_path} (val next1_total={best_val_loss:.6f})")

                # Decay learning rate once per epoch.
                scheduler.step()

            model_path = os.path.join(targetpath, "Pattention_model.pt")
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
