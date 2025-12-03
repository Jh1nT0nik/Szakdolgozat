import os
import argparse
from glob import glob
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


N_RECOG = 32
FEAT_DIM = N_RECOG * (N_RECOG - 1) // 2


def upper_triangle_vectorize(mat: np.ndarray) -> np.ndarray:

    if mat.shape != (N_RECOG, N_RECOG):
        raise ValueError(f"syncmap shape mismatch: {mat.shape}, expected {(N_RECOG, N_RECOG)}")
    iu = np.triu_indices(N_RECOG, k=1)
    v = mat[iu].astype(np.float32)
    if v.shape[0] != FEAT_DIM:
        raise RuntimeError("Feature dim mismatch.")
    return v


def list_syncmap_files(label_dir: str) -> List[str]:

    sync_dir = os.path.join(label_dir, "syncmaps")
    files = sorted(glob(os.path.join(sync_dir, "*.npy")))
    files = [f for f in files if not os.path.basename(f).endswith("_mean_syncmap.npy")]
    return files


class SyncMapDataset(Dataset):
    def __init__(self, files: List[str], y: int):
        self.files = files
        self.y = int(y)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        mat = np.load(p)
        x = upper_triangle_vectorize(mat)
        return torch.from_numpy(x), torch.tensor(self.y, dtype=torch.long), os.path.basename(p)


class MLP(nn.Module):
    def __init__(self, in_dim: int = FEAT_DIM, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def stratified_split(files0: List[str], files1: List[str], val_ratio: float, seed: int
                     ) -> Tuple[List[str], List[str], List[str], List[str]]:

    rng = np.random.RandomState(seed)
    f0 = np.array(files0)
    f1 = np.array(files1)
    rng.shuffle(f0)
    rng.shuffle(f1)

    n0_val = int(round(len(f0) * val_ratio))
    n1_val = int(round(len(f1) * val_ratio))

    val0 = f0[:n0_val].tolist()
    tr0  = f0[n0_val:].tolist()

    val1 = f1[:n1_val].tolist()
    tr1  = f1[n1_val:].tolist()

    return tr0, val0, tr1, val1


@torch.no_grad()
def eval_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    all_y = []
    all_p = []
    for x, y, _name in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        all_y.append(y.cpu().numpy())
        all_p.append(pred.cpu().numpy())
    y = np.concatenate(all_y) if all_y else np.array([], dtype=int)
    p = np.concatenate(all_p) if all_p else np.array([], dtype=int)

    cm = np.zeros((2, 2), dtype=int)
    for yt, pt in zip(y, p):
        cm[int(yt), int(pt)] += 1

    acc = float((y == p).mean()) if y.size else 0.0

    per_class = {}
    for c in [0, 1]:
        denom = cm[c, :].sum()
        per_class[c] = float(cm[c, c] / denom) if denom > 0 else 0.0

    return {"confusion": cm, "acc": acc, "per_class_acc": per_class}


def save_confusion_csv(cm: np.ndarray, label0: str, label1: str, out_csv: str, acc: float,
                       acc0: float, acc1: float) -> None:

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(f"actual\\pred,{label0},{label1}\n")
        f.write(f"{label0},{cm[0,0]},{cm[0,1]}\n")
        f.write(f"{label1},{cm[1,0]},{cm[1,1]}\n")
        f.write("\n")
        f.write(f"overall_accuracy,{acc:.6f}\n")
        f.write(f"accuracy_{label0},{acc0:.6f}\n")
        f.write(f"accuracy_{label1},{acc1:.6f}\n")


def main():
    ap = argparse.ArgumentParser(description="MLP tréning syncmap (32x32) felső-háromszög feature-ökkel (496 dim).")
    ap.add_argument("--data_root", required=True,
                    help="November24 gyökér (pl. /home/langa1/project/November24 vagy November24_learned01).")
    ap.add_argument("--label0", required=True, help="Első hang label (pl. AE).")
    ap.add_argument("--label1", required=True, help="Második hang label (pl. EH).")
    ap.add_argument("--out_dir", default="NOV_28", help="Kimeneti mappa neve/útvonala.")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = os.path.abspath(args.data_root)
    out_dir = os.path.abspath(args.out_dir)
    pair_tag = f"{args.label0}_vs_{args.label1}"
    run_dir = os.path.join(out_dir, pair_tag)
    os.makedirs(run_dir, exist_ok=True)

    label0_dir = os.path.join(data_root, args.label0)
    label1_dir = os.path.join(data_root, args.label1)

    files0 = list_syncmap_files(label0_dir)
    files1 = list_syncmap_files(label1_dir)

    if len(files0) == 0:
        raise FileNotFoundError(f"Nincs syncmap a következő alatt: {label0_dir}/syncmaps/*.npy (mean kizárva)")
    if len(files1) == 0:
        raise FileNotFoundError(f"Nincs syncmap a következő alatt: {label1_dir}/syncmaps/*.npy (mean kizárva)")

    print(f"{args.label0} minták: {len(files0)}")
    print(f"{args.label1} minták: {len(files1)}")
    print(f"Feature dim: {FEAT_DIM}")

    tr0, val0, tr1, val1 = stratified_split(files0, files1, args.val_ratio, args.seed)
    print(f"Train: {args.label0}={len(tr0)}, {args.label1}={len(tr1)} | Val: {args.label0}={len(val0)}, {args.label1}={len(val1)}")

    train_ds = torch.utils.data.ConcatDataset([SyncMapDataset(tr0, 0), SyncMapDataset(tr1, 1)])
    val_ds   = torch.utils.data.ConcatDataset([SyncMapDataset(val0, 0), SyncMapDataset(val1, 1)])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = MLP(in_dim=FEAT_DIM, hidden=args.hidden, dropout=args.dropout).to(device)

    n0 = len(tr0)
    n1 = len(tr1)
    w0 = (n0 + n1) / (2.0 * max(1, n0))
    w1 = (n0 + n1) / (2.0 * max(1, n1))
    class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    history = {"train_loss": [], "val_acc": []}

    for ep in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        n_batches = 0

        for x, y, _name in train_loader:
            x = x.to(device)
            y = y.to(device)

            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()

            ep_loss += float(loss.item())
            n_batches += 1

        ep_loss /= max(1, n_batches)
        history["train_loss"].append(ep_loss)

        metrics = eval_model(model, val_loader, device)
        val_acc = metrics["acc"]
        history["val_acc"].append(val_acc)

        print(f"Epoch {ep:03d}/{args.epochs} | train_loss={ep_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))

    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pt"), map_location=device))
    final_metrics = eval_model(model, val_loader, device)

    cm = final_metrics["confusion"]
    acc = final_metrics["acc"]
    acc0 = final_metrics["per_class_acc"][0]
    acc1 = final_metrics["per_class_acc"][1]

    print("\nFinal validation:")
    print("Confusion matrix:\n", cm)
    print(f"Overall acc: {acc:.4f}")
    print(f"Acc {args.label0}: {acc0:.4f}")
    print(f"Acc {args.label1}: {acc1:.4f}")

    out_cm = os.path.join(run_dir, "confusion_matrix.csv")
    save_confusion_csv(cm, args.label0, args.label1, out_cm, acc, acc0, acc1)

    plt.figure()
    plt.plot(history["train_loss"])
    plt.xlabel("epoch")
    plt.ylabel("train_loss")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "train_loss.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(history["val_acc"])
    plt.xlabel("epoch")
    plt.ylabel("val_acc")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "val_acc.png"), dpi=200)
    plt.close()

    with open(os.path.join(run_dir, "run_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"data_root={data_root}\n")
        f.write(f"label0={args.label0}\n")
        f.write(f"label1={args.label1}\n")
        f.write(f"val_ratio={args.val_ratio}\n")
        f.write(f"epochs={args.epochs}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"lr={args.lr}\n")
        f.write(f"hidden={args.hidden}\n")
        f.write(f"dropout={args.dropout}\n")
        f.write(f"weight_decay={args.weight_decay}\n")
        f.write(f"seed={args.seed}\n")
        f.write("\n")
        f.write(f"final_acc={acc:.6f}\n")
        f.write(f"final_acc_{args.label0}={acc0:.6f}\n")
        f.write(f"final_acc_{args.label1}={acc1:.6f}\n")
        f.write(f"confusion_csv={out_cm}\n")

    print("\nSaved to:", run_dir)
    print(" - best_model.pt")
    print(" - confusion_matrix.csv")
    print(" - train_loss.png, val_acc.png")


if __name__ == "__main__":
    main()
