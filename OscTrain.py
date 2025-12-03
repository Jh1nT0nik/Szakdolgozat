import os
import argparse
from typing import Tuple, Optional, List

import numpy as np
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd

try:
    from torchdiffeq import odeint
except ImportError as e:
    raise ImportError(
        "Hiányzik a torchdiffeq csomag. Telepítsd:\n"
        "    pip install torchdiffeq"
    ) from e


N_INPUT = 16
N_RECOG = 32
N_TOTAL = N_INPUT + N_RECOG

C_STAR = 48 * 4
EPSILON = 0.35
T_TOTAL = 1.0
N_TIMEPOINTS = 5000
TRANSIENT_FRACTION = 0.5

OUTPUT_ROOT = "out"

DTYPE = torch.float64
ODE_METHOD = "rk4"

RECOG_FREQS_HZ_INIT = np.array([
    100.0,
    193.5483871,
    287.09677419,
    380.64516129,
    474.19354839,
    567.74193548,
    661.29032258,
    754.83870968,
    848.38709677,
    941.93548387,
    1035.48387097,
    1129.03225806,
    1222.58064516,
    1316.12903226,
    1409.67741935,
    1503.22580645,
    1596.77419355,
    1690.32258065,
    1783.87096774,
    1877.41935484,
    1970.96774194,
    2064.51612903,
    2158.06451613,
    2251.61290323,
    2345.16129032,
    2438.70967742,
    2532.25806452,
    2625.80645161,
    2719.35483871,
    2812.90322581,
    2906.4516129,
    3000.0,
], dtype=np.float64)

def build_coupling_matrix(n_input: int, n_recog: int, c_star: float) -> torch.Tensor:
    n_total = n_input + n_recog
    C = torch.full((n_total, n_total), c_star, dtype=DTYPE)
    C.fill_diagonal_(0.0)
    C[:n_input, :n_input] = 0.0
    return C

C_GLOBAL = build_coupling_matrix(N_INPUT, N_RECOG, C_STAR)


class Kuramoto(torch.nn.Module):
    def __init__(self, omega: torch.Tensor, C: torch.Tensor):
        super().__init__()
        self.register_buffer("omega", omega)
        self.register_buffer("C", C)

    def forward(self, t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        theta_i = theta.unsqueeze(1)
        theta_j = theta.unsqueeze(0)
        delta = theta_j - theta_i
        coupling = torch.sum(self.C * torch.sin(delta), dim=1)
        dtheta = self.omega + coupling
        return dtheta



def load_csv_freqs_only(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    cols = [f"f{i+1}" for i in range(N_INPUT)]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Hiányzik oszlop: {c} a fájlban: {path}")
    freqs = df[cols].to_numpy(dtype=np.float64)
    return freqs


def save_syncmap_figure(sync_map: np.ndarray,
                        freqs_hz: np.ndarray,
                        out_path: str,
                        title: str = "") -> None:
    assert sync_map.shape == (N_RECOG, N_RECOG)
    assert freqs_hz.shape[0] == N_RECOG

    img = 1.0 - sync_map

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, origin="lower")

    indices = np.arange(N_RECOG)
    step = 5
    tick_pos = indices[::step]
    tick_labels = [f"{i+1}\n{freqs_hz[i]:.1f} Hz" for i in tick_pos]

    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_yticklabels(tick_labels, fontsize=6)

    ax.set_xlabel("Felismerő index / frekvencia (Hz)")
    ax.set_ylabel("Felismerő index / frekvencia (Hz)")

    if title:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_delta_map_figure(delta_map: np.ndarray,
                          freqs_hz: np.ndarray,
                          out_path: str,
                          title: str = "") -> None:
    assert delta_map.shape == (N_RECOG, N_RECOG)
    assert freqs_hz.shape[0] == N_RECOG

    vmax = np.max(np.abs(delta_map)) + 1e-6

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(delta_map, cmap="bwr", vmin=-vmax, vmax=vmax, origin="lower")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    indices = np.arange(N_RECOG)
    step = 5
    tick_pos = indices[::step]
    tick_labels = [f"{i+1}\n{freqs_hz[i]:.1f} Hz" for i in tick_pos]

    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_yticklabels(tick_labels, fontsize=6)

    ax.set_xlabel("Felismerő index / frekvencia (Hz)")
    ax.set_ylabel("Felismerő index / frekvencia (Hz)")

    if title:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_simulation_for_speaker(freqs_hz_input: np.ndarray,
                               recog_freqs_hz: np.ndarray) -> np.ndarray:

    freqs_hz_input = np.asarray(freqs_hz_input, dtype=np.float64)
    assert freqs_hz_input.shape[0] == N_INPUT
    assert recog_freqs_hz.shape[0] == N_RECOG

    omega_input = 2 * np.pi * freqs_hz_input
    omega_recog = 2 * np.pi * recog_freqs_hz
    omega_all = np.concatenate([omega_input, omega_recog]).astype(np.float64)
    omega_all_t = torch.from_numpy(omega_all).to(DTYPE)

    theta0 = 2 * np.pi * torch.rand(N_TOTAL, dtype=DTYPE)

    t = torch.linspace(0.0, T_TOTAL, N_TIMEPOINTS, dtype=DTYPE)

    model = Kuramoto(omega_all_t, C_GLOBAL)

    with torch.no_grad():
        theta_traj = odeint(model, theta0, t, method=ODE_METHOD)

    theta_traj_np = theta_traj.cpu().numpy()
    t_np = t.cpu().numpy()

    start_idx = int(N_TIMEPOINTS * TRANSIENT_FRACTION)
    theta_steady = theta_traj_np[start_idx:, :]

    theta_i = theta_steady[:, :, None]
    theta_j = theta_steady[:, None, :]
    delta = theta_i - theta_j
    sin_delta = np.sin(delta)

    var_over_time = sin_delta.var(axis=0)
    sync_full = (var_over_time < EPSILON).astype(np.uint8)
    np.fill_diagonal(sync_full, 1)

    sync_map_rec = sync_full[N_INPUT:, N_INPUT:]
    return sync_map_rec


def compute_mean_syncmap_for_label(freqs_all: np.ndarray,
                                   recog_freqs_hz: np.ndarray,
                                   label: str) -> np.ndarray:

    num_speakers = freqs_all.shape[0]
    syncmaps = []
    for idx in tqdm(range(num_speakers),
                    desc=f"{label} speakers",
                    unit="speaker"):
        S = run_simulation_for_speaker(freqs_all[idx, :], recog_freqs_hz)
        syncmaps.append(S.astype(np.float64))
    mean_map = np.mean(syncmaps, axis=0)
    return mean_map


def build_targets_from_means(mean1: np.ndarray,
                             mean2: np.ndarray,
                             delta_thresh: float):

    assert mean1.shape == mean2.shape == (N_RECOG, N_RECOG)

    delta = mean1 - mean2
    TA = np.zeros_like(delta, dtype=np.int8)
    TB = np.zeros_like(delta, dtype=np.int8)

    TA[delta > delta_thresh] = 1
    TB[delta < -delta_thresh] = 1

    disc_mask = (TA == 1) | (TB == 1)

    return TA, TB, delta, disc_mask


def update_frequencies_from_D(f_recog_hz: np.ndarray,
                              D: np.ndarray,
                              disc_mask: np.ndarray,
                              eta_pos: float,
                              eta_neg: float,
                              fmin: float = 50.0,
                              fmax: float = 4000.0) -> np.ndarray:

    f = f_recog_hz.copy()
    for i in range(N_RECOG):
        for j in range(i + 1, N_RECOG):
            if not disc_mask[i, j]:
                continue
            d = D[i, j]
            if d == 1:
                fi, fj = f[i], f[j]
                m = 0.5 * (fi + fj)
                f[i] += eta_pos * (m - fi)
                f[j] += eta_pos * (m - fj)
            elif d == -1:
                fi, fj = f[i], f[j]
                diff = fi - fj
                f[i] += eta_neg * diff
                f[j] -= eta_neg * diff

    f = np.clip(f, fmin, fmax)
    return f


def train_recognizer_frequencies(
    freqs1: np.ndarray,
    freqs2: np.ndarray,
    TA: np.ndarray,
    TB: np.ndarray,
    disc_mask: np.ndarray,
    f_init_hz: np.ndarray,
    label1: str,
    label2: str,
    epochs: int = 3,
    eta_pos: float = 0.05,
    eta_neg: float = 0.02,
    seed: int = 42,
):

    rng = np.random.RandomState(seed)

    f_recog = f_init_hz.copy()
    n1 = freqs1.shape[0]
    n2 = freqs2.shape[0]

    for ep in range(1, epochs + 1):
        print(f"\n=== Epoch {ep}/{epochs} ===")

        # label1 trialok
        idx1 = rng.permutation(n1)
        for idx in tqdm(idx1, desc=f"Train {label1}", unit="speaker"):
            S = run_simulation_for_speaker(freqs1[idx, :], f_recog)
            D = TA.astype(int) - S.astype(int)
            D[~disc_mask] = 0
            f_recog = update_frequencies_from_D(
                f_recog, D, disc_mask, eta_pos, eta_neg
            )

        # label2 trialok
        idx2 = rng.permutation(n2)
        for idx in tqdm(idx2, desc=f"Train {label2}", unit="speaker"):
            S = run_simulation_for_speaker(freqs2[idx, :], f_recog)
            D = TB.astype(int) - S.astype(int)
            D[~disc_mask] = 0
            f_recog = update_frequencies_from_D(
                f_recog, D, disc_mask, eta_pos, eta_neg
            )

        print(f"Epoch {ep} után felismerő frekik [Hz]:")
        print(np.round(f_recog, 2))

    return f_recog


def main():
    parser = argparse.ArgumentParser(
        description="Két hang (top16_LABEL1.csv, top16_LABEL2.csv) alapján "
                    "oscillátoros tanulás (recognizer frekvenciák)."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="A topN16 mappa (ahol a top16_*.csv fájlok vannak).",
    )
    parser.add_argument(
        "--label1",
        type=str,
        required=True,
        help="Első hang label (pl. AE).",
    )
    parser.add_argument(
        "--label2",
        type=str,
        required=True,
        help="Második hang label (pl. IH).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=OUTPUT_ROOT,
        help=f"Kimeneti gyökér mappa (alap: {OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--delta_thresh",
        type=float,
        default=0.2,
        help="Küszöb a mean map különbségére (|mu1-mu2|>delta_thresh számít diszkriminatívnak).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Tanulási epochok száma.",
    )
    parser.add_argument(
        "--eta_pos",
        type=float,
        default=0.5,
        help="Lépésköz, ha szinkront akarunk (pozitív).",
    )
    parser.add_argument(
        "--eta_neg",
        type=float,
        default=0.2,
        help="Lépésköz, ha szinkront akarunk megszüntetni.",
    )
    parser.add_argument(
        "--auto_train",
        action="store_true",
        help="Ne kérdezzen rá, a pattern-ek kiszámítása után azonnal induljon a tanulás.",
    )

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_root = os.path.abspath(args.output_root)
    label1 = args.label1
    label2 = args.label2

    csv1 = os.path.join(input_dir, f"top16_{label1}.csv")
    csv2 = os.path.join(input_dir, f"top16_{label2}.csv")

    if not os.path.isfile(csv1):
        raise FileNotFoundError(f"Nem találom: {csv1}")
    if not os.path.isfile(csv2):
        raise FileNotFoundError(f"Nem találom: {csv2}")

    print(f"Input dir:   {input_dir}")
    print(f"Output root: {output_root}")
    print(f"Hangok:      {label1} vs {label2}")
    print(f"CSV1:        {csv1}")
    print(f"CSV2:        {csv2}")

    pair_root = os.path.join(output_root, "Tanulás_osc", f"{label1}_vs_{label2}")
    patterns_dir = os.path.join(pair_root, "patterns")
    train_dir = os.path.join(pair_root, "training")
    os.makedirs(patterns_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    freqs1 = load_csv_freqs_only(csv1)
    freqs2 = load_csv_freqs_only(csv2)
    print(f"{label1} beszélők: {freqs1.shape[0]}")
    print(f"{label2} beszélők: {freqs2.shape[0]}")

    print("\nÁtlagolt syncmap számítás (kezdeti felismerő frekikkel)...")
    mean1 = compute_mean_syncmap_for_label(freqs1, RECOG_FREQS_HZ_INIT, label1)
    mean2 = compute_mean_syncmap_for_label(freqs2, RECOG_FREQS_HZ_INIT, label2)

    np.save(os.path.join(patterns_dir, f"mean_map_{label1}.npy"), mean1)
    np.save(os.path.join(patterns_dir, f"mean_map_{label2}.npy"), mean2)

    save_syncmap_figure(
        mean1, RECOG_FREQS_HZ_INIT,
        os.path.join(patterns_dir, f"mean_map_{label1}.png"),
        title=f"{label1} – átlagolt syncmap"
    )
    save_syncmap_figure(
        mean2, RECOG_FREQS_HZ_INIT,
        os.path.join(patterns_dir, f"mean_map_{label2}.png"),
        title=f"{label2} – átlagolt syncmap"
    )

    TA, TB, delta_map, disc_mask = build_targets_from_means(
        mean1, mean2, args.delta_thresh
    )

    np.save(os.path.join(patterns_dir, f"delta_map_{label1}_minus_{label2}.npy"),
            delta_map)
    save_delta_map_figure(
        delta_map, RECOG_FREQS_HZ_INIT,
        os.path.join(patterns_dir, f"delta_map_{label1}_minus_{label2}.png"),
        title=f"Δ = {label1} - {label2}"
    )

    np.save(os.path.join(patterns_dir, f"TA_{label1}.npy"), TA)
    np.save(os.path.join(patterns_dir, f"TB_{label2}.npy"), TB)
    np.save(os.path.join(patterns_dir, "disc_mask.npy"), disc_mask.astype(np.uint8))

    print("\nMintapattern-ek és delta map elmentve ide:")
    print(f"  {patterns_dir}")
    print("Nézd meg a PNG-ket, hogy a két hang között hol vannak valódi különbségek.")

    if not args.auto_train:
        ans = input("\nIndulhat a tanulás ezekre a patternekre? [y/N]: ").strip().lower()
        if not ans.startswith("y"):
            print("Tanulás kihagyva. Csak a minták lettek kiszámolva.")
            return

    print("\nTanulás indul...")
    f_learned = train_recognizer_frequencies(
        freqs1, freqs2,
        TA, TB, disc_mask,
        RECOG_FREQS_HZ_INIT,
        label1, label2,
        epochs=args.epochs,
        eta_pos=args.eta_pos,
        eta_neg=args.eta_neg,
    )

    np.save(os.path.join(train_dir, "recog_freqs_learned_hz.npy"), f_learned)
    with open(os.path.join(train_dir, "recog_freqs_learned_hz.txt"), "w") as f:
        for i, fr in enumerate(f_learned):
            f.write(f"{i+1}\t{fr:.6f}\n")

    print("\nTanulás kész.")
    print(f"Tanult felismerő frekvenciák elmentve ide: {train_dir}")


if __name__ == "__main__":
    main()
