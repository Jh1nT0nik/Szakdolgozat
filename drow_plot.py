# drow_plot.py
import os, re, glob, sys, webbrowser
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_PARENT = os.path.join(BASE_DIR, "out")

def choose_run_dir(cli_run: str | None = None):
    if not os.path.isdir(OUT_PARENT):
        print("Nincs 'out' mappa.")
        return None
    runs = [d for d in sorted(os.listdir(OUT_PARENT))
            if os.path.isdir(os.path.join(OUT_PARENT, d))]
    if not runs:
        print("Nincs futás az out/ alatt.")
        return None

    if cli_run and cli_run in runs:
        return os.path.join(OUT_PARENT, cli_run)

    print("Elérhető futások:", runs)
    try:
        name = input(f"Válassz futásnevet (Enter = {runs[-1]}): ").strip()
    except Exception:
        name = ""
    if not name:
        name = runs[-1]
    if name not in runs:
        print("Nincs ilyen futás:", name)
        return None
    return os.path.join(OUT_PARENT, name)

def list_pairs(run_dir: str):
    rx = re.compile(r"^sim_a(\d+)_b(\d+)\.npz$")
    pairs = []
    for fn in os.listdir(run_dir):
        m = rx.match(fn)
        if m:
            pairs.append((int(m.group(1)), int(m.group(2))))
    pairs.sort()
    return pairs

def ask_pair(pairs, cli_a: int | None = None, cli_b: int | None = None):
    if not pairs:
        return None
    last_a, last_b = pairs[-1]
    as_ = sorted({a for a, _ in pairs})
    bs_ = sorted({b for _, b in pairs})

    if cli_a is not None and cli_b is not None and (cli_a, cli_b) in set(pairs):
        return cli_a, cli_b

    print(f"Elérhető a-értékek: {as_}")
    print(f"Elérhető b-értékek: {bs_}")
    try:
        a_in = input(f"a érték (Enter = {last_a}): ").strip()
        b_in = input(f"b érték (Enter = {last_b}): ").strip()
    except Exception:
        a_in = b_in = ""

    a = last_a if a_in == "" else int(a_in)
    b = last_b if b_in == "" else int(b_in)

    if (a, b) not in set(pairs):
        print(f"Nincs ilyen kombináció ebben a futásban: a={a}, b={b}")
        same_a = [p for p in pairs if p[0] == a][:10]
        same_b = [p for p in pairs if p[1] == b][:10]
        if same_a:
            print("Létező párok ezzel az a-val:", same_a)
        if same_b:
            print("Létező párok ezzel a b-vel:", same_b)
        return None
    return a, b

def plot_from_npz(npz_path: str, save_dir: str, show_hz: bool = False):
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    import os, webbrowser, math

    d = np.load(npz_path)
    t = d["t"]
    F = d["freqs"]

    has_omega = "omega" in d.files
    omega = d["omega"] if has_omega else None  # [osc]

    unit = "Hz"
    if show_hz:
        F = F * 2 * math.pi   # rad/s
        if has_omega:
            omega = omega * 2 * math.pi
        unit = "rad/s"

    fig = go.Figure()
    for i in range(F.shape[1]):
        if has_omega:
            label_val = omega[i]
            label = f"f₀ = {label_val:.3g} {unit}"
        else:
            f0 = F[0, i]
            label = f"ω₀ = {f0:.3g} {unit}"

        fig.add_trace(go.Scatter(
            x=t,
            y=F[:, i],
            mode='lines',
            name=label
        ))

    trend_line = F.mean(axis=1)
    fig.add_trace(go.Scatter(
        x=t,
        y=trend_line,
        mode='lines',
        line=dict(dash='dash', color='black'),
        name='Trendvonal'
    ))

    fig.update_layout(
        title=f'Az oszcillátorok Frekvencia - Idő diagramja — {os.path.basename(npz_path)}',
        xaxis_title='Idő (s)',
        yaxis_title=f'Frekvencia ({unit})',
        hovermode='x unified',
        legend=dict(x=1, y=1),
        template='plotly_white'
    )

    html_name = f"plot_{os.path.splitext(os.path.basename(npz_path))[0]}.html"
    html_path = os.path.join(save_dir, html_name)
    pio.write_html(fig, file=html_path, full_html=True, include_plotlyjs="inline")
    print("HTML mentve:", html_path)
    webbrowser.open('file://' + os.path.realpath(html_path))

def parse_cli():
    run = a = b = None
    for arg in sys.argv[1:]:
        if arg.startswith("--run="): run = arg.split("=", 1)[1]
        elif arg.startswith("--a="): a = int(arg.split("=", 1)[1])
        elif arg.startswith("--b="): b = int(arg.split("=", 1)[1])
    return run, a, b

if __name__ == "__main__":
    cli_run, cli_a, cli_b = parse_cli()

    run_dir = choose_run_dir(cli_run)
    if not run_dir:
        raise SystemExit

    pairs = list_pairs(run_dir)
    if not pairs:
        print("Nincs .npz ebben a futásban:", run_dir)
        raise SystemExit

    sel = None
    while sel is None:
        sel = ask_pair(pairs, cli_a, cli_b)
        if sel is None:
            cli_a = cli_b = None

    a, b = sel
    npz_path = os.path.join(run_dir, f"sim_a{a}_b{b}.npz")
    plot_from_npz(npz_path, save_dir=run_dir)
