import os
import re
import argparse
from typing import Tuple

import numpy as np
import torch

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

C_STAR = 0.1
EPSILON = 0.35
T_TOTAL = 1.0
N_TIMEPOINTS = 800
TRANSIENT_FRACTION = 0.5

OUTPUT_ROOT = "out"


def extract_label_from_filename(filename: str) -> str:

    base = os.path.basename(filename)
    m = re.match(r"top16_([^.]*)\.csv", base)
    if not m:
        return base
    return m.group(1)


def build_coupling_matrix(n_input: int, n_recog: int, c_star: float) -> torch.Tensor:

    n_total = n_input + n_recog
    C = torch.full((n_total, n_total), c_star, dtype=torch.float32)
    C.fill_diagonal_(0.0)

    C[:n_input, :n_input] = 0.0

    return C


def make_recognizer_omega(input_omega_rad_s: torch.Tensor, n_recog: int) -> torch.Tensor:

    n_input = input_omega_rad_s.numel()
    reps = int(np.ceil(n_recog / n_input))
    tiled = input_omega_rad_s.repeat(reps)[:n_recog]
    return tiled


class Kuramoto(torch.nn.Module):


    def __init__(self, omega: torch.Tensor, C: torch.Tensor):
        super().__init__()
        # bufferként tároljuk, hogy ne legyenek gradiens alatt
        self.register_buffer("omega", omega)
        self.register_buffer("C", C)

    def forward(self, t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        # theta: (N,)
        theta_i = theta.unsqueeze(1)   # (N,1)
        theta_j = theta.unsqueeze(0)   # (1,N)
        delta = theta_j - theta_i      # (N,N)
        coupling = torch.sum(self.C * torch.sin(delta), dim=1)
        dtheta = self.omega + coupling
        return dtheta


def run_simulation_for_speaker(freqs_hz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:

    assert freqs_hz.shape[0] == N_INPUT, "A CSV sorának pontosan 16 frekvenciát kell tartalmaznia."

    omega_input = 2 * np.pi * freqs_hz  # rad/s
    omega_input_t = torch.from_numpy(omega_input.astype(np.float32))

    omega_recog_t = make_recognizer_omega(omega_input_t, N_RECOG)

    omega_all = torch.cat([omega_input_t, omega_recog_t], dim=0)

    C = build_coupling_matrix(N_INPUT, N_RECOG, C_STAR)

    theta0 = 2 * np.pi * torch.rand(N_TOTAL, dtype=torch.float32)

    t = torch.linspace(0.0, T_TOTAL, N_TIMEPOINTS, dtype=torch.float32)

    model = Kuramoto(omega_all, C)

    with torch.no_grad():
        theta_traj = odeint(model, theta0, t, method="dopri5")

    theta_traj_np = theta_traj.cpu().numpy()
    t_np = t.cpu().numpy()

    dt = float(t_np[1] - t_np[0])
    freq_rad_s = np.diff(theta_traj_np, axis=0) / dt
    freq_hz = freq_rad_s / (2 * np.pi)

    start_idx = int(N_TIMEPOINTS * TRANSIENT_FRACTION)
    theta_steady = theta_traj_np[start_idx:, :]

    theta_i = theta_steady[:, :, None]
    theta_j = theta_steady[:, None, :]
    delta = theta_i - theta_j
    sin_delta = np.sin(delta)

    var_over_time = sin_delta.var(axis=0)  # (N, N)

    sync_matrix = (var_over_time < EPSILON).astype(np.uint8)
    np.fill_diagonal(sync_matrix, 1)

    sync_vector = sync_matrix.sum(axis=1) - 1

    sim_data = dict(
        t=t_np,
        theta=theta_traj_np,
        freq_rad_s=freq_rad_s,
        freq_hz=freq_hz,
        omega=omega_all.cpu().numpy(),
    )

    return sync_matrix, sync_vector, sim_data


def load_csv_freqs(path: str) -> np.ndarray:

    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < N_INPUT:
        raise ValueError(
            f"A(z) {path} fájlban csak {data.shape[1]} oszlop van, "
            f"de legalább {N_INPUT} kell."
        )
    return data[:, :N_INPUT]


def ensure_dirs(base_label_dir: str):

    os.makedirs(os.path.join(base_label_dir, "syncmaps"), exist_ok=True)
    os.makedirs(os.path.join(base_label_dir, "syncvectors"), exist_ok=True)
    os.makedirs(os.path.join(base_label_dir, "simdata"), exist_ok=True)




def process_directory(input_dir: str, output_root: str = OUTPUT_ROOT):
    input_dir = os.path.abspath(input_dir)
    output_root = os.path.abspath(output_root)

    print(f"Input mappa:  {input_dir}")
    print(f"Output gyökér: {output_root}")
    os.makedirs(output_root, exist_ok=True)

    csv_files = [
        f for f in os.listdir(input_dir)
        if f.startswith("top16_")
        and f.endswith(".csv")
        and "detailed" not in f
    ]
    csv_files.sort()

    if not csv_files:
        print("Nem találtam 'top16_*.csv' fájlt (a 'detailed' nélküliektől eltekintve).")
        return

    for csv_name in csv_files:
        csv_path = os.path.join(input_dir, csv_name)
        label = extract_label_from_filename(csv_name)
        label_dir = os.path.join(output_root, label)
        ensure_dirs(label_dir)

        print(f"\n=== Fájl: {csv_name} | Label: {label} ===")
        freqs_all = load_csv_freqs(csv_path)

        num_speakers = freqs_all.shape[0]
        print(f"Beszélők száma: {num_speakers}")

        for idx in range(num_speakers):
            freqs_hz = freqs_all[idx, :]
            base_name = f"spk_{idx:03d}"

            print(f"  - Szimuláció beszélő #{idx} ({base_name}) ...", end="", flush=True)

            sync_matrix, sync_vector, sim_data = run_simulation_for_speaker(freqs_hz)

            syncmap_path = os.path.join(label_dir, "syncmaps", base_name + "_syncmap.npy")
            syncvec_path = os.path.join(label_dir, "syncvectors", base_name + "_syncvec.npy")
            simdata_path = os.path.join(label_dir, "simdata", base_name + "_simdata.npz")

            np.save(syncmap_path, sync_matrix)
            np.save(syncvec_path, sync_vector)
            np.savez(simdata_path, **sim_data)

            print(" kész.")


def main():
    parser = argparse.ArgumentParser(description="Kuramoto-szerű oszcillátor háló szimuláció top16_*.csv fájlokra.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="A topN16 mappa elérési útja (ahol a top16_*.csv fájlok vannak).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=OUTPUT_ROOT,
        help=f"Kimeneti gyökér mappa (alapértelmezés: {OUTPUT_ROOT} a futtatási mappában).",
    )
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_root)


if __name__ == "__main__":
    main()
