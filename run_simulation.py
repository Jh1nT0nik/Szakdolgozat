# run_simulation.py

import os
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torchdiffeq import odeint

import plotly.graph_objects as go

import os, re, sys
from datetime import datetime

OUT_PARENT = "out"

def _sanitize(name: str) -> str:
    name = name.strip()
    # fájlrendszer-barát név
    name = re.sub(r'[^0-9A-Za-z._-]+', '_', name)
    return name or datetime.now().strftime("run_%Y%m%d_%H%M%S")

def _pick_run_name() -> str:
    for arg in sys.argv[1:]:
        if arg.startswith("--run-name="):
            return _sanitize(arg.split("=", 1)[1])
    try:
        name = input("Adj egy futásnevet (Enter = időbélyeg): ")
    except Exception:
        name = ""
    return _sanitize(name)

def create_run_dir() -> str:
    os.makedirs(OUT_PARENT, exist_ok=True)
    base = _pick_run_name()
    run_dir = os.path.join(OUT_PARENT, base)
    if os.path.exists(run_dir):
        i = 1
        while os.path.exists(run_dir):
            run_dir = os.path.join(OUT_PARENT, f"{base}_{i:03d}")
            i += 1
    os.makedirs(run_dir, exist_ok=False)
    return run_dir

TEST_SMALL =  False

OSCNUM = 6

C_SCALE = 2

MAX_TIME = 2.083

N_OUT = 5000

RTOL = 1e-5
ATOL = 1e-7

DEVICE_MODE = "auto"

OUT_DIR = "out"


def choose_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("CUDA nem elérhető, CPU-ra esek vissza.")
            return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_frequency_grid():

    numbers = list(range(5, 46))
    extras = [10, 15, 20, 25]
    grid = [(a, b, *extras) for a in numbers for b in numbers]  # 41*41 = 1681
    if TEST_SMALL:
        grid = [(500, 500, *extras), (1000, 800, *extras), (1500, 2000, *extras), (2500, 3000, *extras)]
    return [np.array(x, dtype=np.float64) for x in grid]


def build_coupling_matrix(oscnum: int, scale: float) -> np.ndarray:
    C = np.ones((oscnum, oscnum), dtype=np.float64)
    np.fill_diagonal(C, 0.0)
    C[0, 1] = 0.0
    C[1, 0] = 0.0
    C *= scale
    return C


def initial_phases(oscnum: int, seed: int = 13) -> np.ndarray:
    rng = np.random.default_rng(seed)
    random_tail = rng.random(oscnum - 2) * 2.0 * np.pi
    theta0 = np.concatenate(([0.0, 0.0], random_tail)).astype(np.float64)
    return theta0


def simulator_torch(theta0_np: np.ndarray,
                    omega_np: np.ndarray,
                    Cij_t: torch.Tensor,
                    t_t: torch.Tensor,
                    device: torch.device,
                    rtol=1e-5,
                    atol=1e-7,
                    return_theta=False):
    dtype = torch.float32
    theta0_t = torch.tensor(theta0_np, dtype=dtype, device=device)
    omega_t = torch.tensor(omega_np, dtype=dtype, device=device)

    def f(_t, th):
        diff = th.unsqueeze(0) - th.unsqueeze(1)
        inter = (Cij_t * torch.sin(diff)).sum(dim=1)
        return omega_t + inter

    theta_sol = odeint(f, theta0_t, t_t, method='dopri5', rtol=rtol, atol=atol)
    theta_np = theta_sol.detach().cpu().numpy().astype(np.float64)

    t_np = t_t.detach().cpu().numpy().astype(np.float64)
    instfreq = np.gradient(theta_np, t_np, axis=0)

    if return_theta:
        return instfreq, theta_np
    return instfreq, None


def plot_sample(t: np.ndarray, freqs: np.ndarray, title="Mintaplot – inst. frekvenciák"):
    fig = go.Figure()
    for i in range(freqs.shape[1]):
        fig.add_trace(go.Scatter(x=t, y=freqs[:, i], mode='lines', name=f'Oszc {i+1}'))
    fig.update_layout(
        title=title,
        xaxis_title='Idő (s)',
        yaxis_title='Frekvencia (Hz)',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(x=0, y=1)
    )
    fig.show()



def main():
    t = np.linspace(0.0, MAX_TIME, N_OUT, dtype=np.float64)

    device = choose_device(DEVICE_MODE)
    print(f"Eszköz: {device}")

    run_dir = create_run_dir()
    print(f"Mentési mappa: {run_dir}")

    dtype = torch.float32
    Cij = build_coupling_matrix(OSCNUM, C_SCALE)
    Cij_t = torch.tensor(Cij, dtype=dtype, device=device)
    t_t   = torch.tensor(t,   dtype=dtype, device=device)

    theta0 = initial_phases(OSCNUM, seed=13)

    omega_list = build_frequency_grid()
    print(f"Futások száma: {len(omega_list)} (TEST_SMALL={TEST_SMALL})")

    summary_rows = []

    for omega in tqdm(omega_list, desc="Szimulációk futtatása"):
        freqs, _ = simulator_torch(
            theta0, omega, Cij_t, t_t,
            device=device, rtol=RTOL, atol=ATOL, return_theta=False
        )

        last_slice = slice(int(0.8 * len(t)), None)
        mean_last  = freqs[last_slice].mean(axis=0)

        summary_rows.append(list(omega) + list(mean_last))

        a, b = int(omega[0]), int(omega[1])
        np.savez_compressed(
            os.path.join(run_dir, f"sim_a{a}_b{b}.npz"),
            t=t, freqs=freqs, omega=omega
        )

    cols = [f"w{i+1}" for i in range(OSCNUM)] + [f"mean_last_f{i+1}" for i in range(OSCNUM)]
    pd.DataFrame(summary_rows, columns=cols).to_csv(
        os.path.join(run_dir, "summary_metrics.csv"),
        index=False
    )
    print(f"Összegzés elmentve: {os.path.join(run_dir, 'summary_metrics.csv')}")

    import json
    config = {
        "OSCNUM": OSCNUM, "C_SCALE": C_SCALE, "MAX_TIME": MAX_TIME, "N_OUT": N_OUT,
        "RTOL": RTOL, "ATOL": ATOL, "DEVICE_MODE": DEVICE_MODE, "TEST_SMALL": TEST_SMALL
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    first_npz = next((os.path.join(run_dir, fn) for fn in os.listdir(run_dir) if fn.endswith(".npz")), None)
    if first_npz:
        data = np.load(first_npz)
        try:
            plot_sample(data["t"], data["freqs"], title=f"Mintaplot – {os.path.basename(first_npz)}")
        except Exception:
            pass



if __name__ == "__main__":
    main()
