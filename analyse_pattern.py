# analyse_pattern.py
import os, re, sys, glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_PARENT = os.path.join(BASE_DIR, "out")
OSCNUM     = 6
EPSILON    = 0.35

def identify_pattern_by_me_plusplus(sync_pairs):
    res = []
    has_12 = (3,4) in sync_pairs or (4,3) in sync_pairs
    has_23 = (4,5) in sync_pairs or (5,4) in sync_pairs
    has_34 = (5,6) in sync_pairs or (6,5) in sync_pairs
    has_13 = (3,5) in sync_pairs or (5,3) in sync_pairs
    has_24 = (4,6) in sync_pairs or (6,4) in sync_pairs
    has_14 = (3,6) in sync_pairs or (6,3) in sync_pairs

    if has_12 and has_13 and has_23 and has_24 and has_34:
        res.append("Full Sync")
    if has_12:
        res.append("Pattern 1")
    if has_12 and has_23 and has_13:
        res.append("Pattern 2")
    if has_23 and has_34 and has_24:
        res.append("Pattern 3")
    if has_34:
        res.append("Pattern 4")
    if has_12 and has_34 and not has_23 and not has_13 and not has_24 and not has_14:
        res.append("Pattern 5")
    if has_23:
        res.append("Pattern 6")
    if has_23 and has_14 and not has_12 and not has_34 and not has_13 and not has_24:
        res.append("Or2 --- Or3  és  Or1 --- Or4")
    if has_13 and has_24 and not has_12 and not has_34 and not has_14 and not has_23:
        res.append("Or1 --- Or3  és  Or2 --- Or4")
    return res

def choose_run_dir():
    if not os.path.isdir(OUT_PARENT):
        print("Nincs 'out' mappa."); return None
    runs = [d for d in sorted(os.listdir(OUT_PARENT)) if os.path.isdir(os.path.join(OUT_PARENT, d))]
    if not runs:
        print("Nincs futás az out/ alatt."); return None
    print("Elérhető futások:", runs)
    try:
        name = input(f"Válassz futásnevet (Enter = {runs[-1]}): ").strip()
    except Exception:
        name = ""
    if not name: name = runs[-1]
    if name not in runs:
        print("Nincs ilyen futás:", name); return None
    return os.path.join(OUT_PARENT, name)

def theta_from_freqs(t: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    t = np.asarray(t); F = np.asarray(freqs)
    T, K = F.shape
    out = np.zeros((T, K), dtype=F.dtype)
    if T <= 1: return out
    dt  = np.diff(t)[:, None]        # [T-1,1]
    mid = 0.5 * (F[1:] + F[:-1])     # [T-1,K]
    out[1:] = np.cumsum(dt * mid, axis=0)
    return out  # sin(Δθ)-hez nem kell mod 2π

def scan_entries(run_dir: str):
    rx = re.compile(r"^sim_a(\d+)_b(\d+)\.npz$")
    entries = []
    for fn in os.listdir(run_dir):
        m = rx.match(fn)
        if m:
            a = int(m.group(1)); b = int(m.group(2))
            entries.append((a, b, os.path.join(run_dir, fn)))
    entries.sort()
    return entries

def gcd_step(vals):
    if len(vals) < 2: return None
    diffs = np.diff(np.sort(np.array(vals, dtype=int)))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0: return None
    g = int(np.gcd.reduce(diffs))
    return g if g > 0 else None

def ask_grid(a_vals, b_vals):
    a_min, a_max = min(a_vals), max(a_vals)
    b_min, b_max = min(b_vals), max(b_vals)
    a_step_auto = gcd_step(a_vals) or (a_max - a_min) or 1
    b_step_auto = gcd_step(b_vals) or (b_max - b_min) or 1

    print(f"Autó rács:  a: {a_min} → {a_max} step {a_step_auto}  ({len(a_vals)} egyedi érték)")
    print(f"            b: {b_min} → {b_max} step {b_step_auto}  ({len(b_vals)} egyedi érték)")
    use_auto = input("Használjuk ezt? (Enter = igen, n = kézi megadás) ").strip().lower()
    if use_auto != "n":
        a_grid = np.arange(a_min, a_max + a_step_auto, a_step_auto, dtype=int)
        b_grid = np.arange(b_min, b_max + b_step_auto, b_step_auto, dtype=int)
        return a_grid, b_grid, True

    def read_int(prompt, default):
        s = input(f"{prompt} (Enter = {default}): ").strip()
        try:
            return int(s) if s != "" else int(default)
        except:
            print("Érvénytelen, a defaultot használom."); return int(default)

    a_min_u = read_int("a_min", a_min)
    a_max_u = read_int("a_max", a_max)
    a_step_u = read_int("a_step", a_step_auto)
    b_min_u = read_int("b_min", b_min)
    b_max_u = read_int("b_max", b_max)
    b_step_u = read_int("b_step", b_step_auto)

    if a_step_u <= 0 or b_step_u <= 0:
        print("A step legyen pozitív. Vissza az autóra.")
        a_grid = np.arange(a_min, a_max + a_step_auto, a_step_auto, dtype=int)
        b_grid = np.arange(b_min, b_max + b_step_auto, b_step_auto, dtype=int)
        return a_grid, b_grid, True

    a_grid = np.arange(min(a_min_u, a_max_u), max(a_min_u, a_max_u) + a_step_u, a_step_u, dtype=int)
    b_grid = np.arange(min(b_min_u, b_max_u), max(b_min_u, b_max_u) + b_step_u, b_step_u, dtype=int)
    return a_grid, b_grid, False

def analyze_run(run_dir: str, epsilon: float = EPSILON, oscnum: int = OSCNUM):
    entries = scan_entries(run_dir)
    if not entries:
        print("Nincs *.npz ebben a futásban:", run_dir); return None

    patterns_by_pair = {}
    for a, b, path in entries:
        d = np.load(path)
        t = d["t"]; freqs = d["freqs"]
        theta = theta_from_freqs(t, freqs)

        sync_pairs = []
        for i in range(oscnum):
            for j in range(oscnum):
                diff = np.sin(theta[:, i] - theta[:, j])
                if np.var(diff) < epsilon:
                    sync_pairs.append((i+1, j+1))
        patterns_by_pair[(a, b)] = identify_pattern_by_me_plusplus(sync_pairs)

    a_vals = sorted({a for a, _ in patterns_by_pair})
    b_vals = sorted({b for _, b in patterns_by_pair})
    a_grid, b_grid, used_auto = ask_grid(a_vals, b_vals)

    pattern_names = [
        "Full Sync","Pattern 1","Pattern 2","Pattern 3","Pattern 4","Pattern 5","Pattern 6",
        "Or2 --- Or3  és  Or1 --- Or4","Or1 --- Or3  és  Or2 --- Or4"
    ]
    maps = {p: np.zeros((len(a_grid), len(b_grid)), dtype=int) for p in pattern_names}

    a_idx = {a:i for i,a in enumerate(a_grid)}
    b_idx = {b:i for i,b in enumerate(b_grid)}
    for (a,b), plist in patterns_by_pair.items():
        if a in a_idx and b in b_idx:
            ai, bi = a_idx[a], b_idx[b]
            for p in plist:
                if p in maps:
                    maps[p][ai, bi] = 1

    return maps, a_grid, b_grid, patterns_by_pair

def plot_maps(pattern_maps, a_grid, b_grid, run_dir: str):
    save_dir = os.path.join(run_dir, "pattern_maps_Magyar_felirat_2.0")
    os.makedirs(save_dir, exist_ok=True)

    extent = [int(a_grid.min()), int(a_grid.max()), int(b_grid.max()), int(b_grid.min())]
    for name, mat in pattern_maps.items():
        plt.figure(figsize=(6,5))
        plt.imshow(mat, extent=extent, cmap='Greys', interpolation='nearest')
        plt.gca().invert_yaxis()
        plt.title(f"Szinkrontérkép: {name}")
        plt.xlabel("F_a (Hz)")
        plt.ylabel("F_b (Hz)")
        plt.colorbar(label="Van (1) / Nincs (0)")

        def nice_ticks(vals, nticks=6):
            vals = np.array(vals)
            ticks = np.linspace(vals.min(), vals.max(), nticks).round(0).astype(int)
            return np.unique(ticks)
        plt.xticks(nice_ticks(a_grid))
        plt.yticks(nice_ticks(b_grid))
        plt.tight_layout()
        out_png = os.path.join(save_dir, f"{name.replace(' ', '_').replace('/', '-')}.png")
        plt.savefig(out_png, dpi=160)
        plt.close()
        print("Mentve:", out_png)

if __name__ == "__main__":
    run_dir = choose_run_dir()
    if not run_dir:
        sys.exit(0)

    res = analyze_run(run_dir, epsilon=EPSILON, oscnum=OSCNUM)
    if res is None:
        sys.exit(0)

    pattern_maps, a_grid, b_grid, patterns_by_pair = res

    plot_maps(pattern_maps, a_grid, b_grid, run_dir)

    rows = [{"a":a, "b":b, "patterns":"; ".join(patts)} for (a,b), patts in sorted(patterns_by_pair.items())]
    csv_path = os.path.join(run_dir, "pattern_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print("Összefoglaló CSV:", csv_path)
