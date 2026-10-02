from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import unitary_group

import QSD_(36) as q36
import qsd_optimized as q20


# ------------------------------------------------ 1) tag gates with their block --
@contextmanager
def tagging(mod, leaf_size, leaf_name):
    orig_mux, orig_qsd = mod.mux_rotation, mod.qsd

    def mux(ops, axis, target, controls, angles, *args, **kwargs):
        ops.append(("MARK", f"{axis} mux q{target}<-{list(controls)}", None))
        return orig_mux(ops, axis, target, controls, angles, *args, **kwargs)

    def qsd(ops, U, wires):
        if len(wires) == leaf_size:
            ops.append(("MARK", f"{leaf_name} q{list(wires)}", None))
        return orig_qsd(ops, U, wires)

    mod.mux_rotation, mod.qsd = mux, qsd
    try:
        yield
    finally:
        mod.mux_rotation, mod.qsd = orig_mux, orig_qsd


def strip_marks(ops):
    return [op for op in ops if op[0] != "MARK"]


# ------------------------------------------------------- 2) ops -> DataFrame ----
def ops_to_df(ops, version, had=None):
    rows, block_id, block, step = [], -1, "(none)", 0
    for kind, a, b in ops:
        if kind == "MARK":
            block_id, block = block_id + 1, a
            continue
        row = dict(version=version, step=step, block_id=block_id, block=block,
                   block_type=block.split(" q", 1)[0], gate=kind,
                   control=None, target=None, angle=np.nan, matrix=None)
        if kind in ("CNOT", "CZ"):
            row.update(control=a, target=b)
        elif kind == "U":
            row.update(target=a, matrix=b)
            if had is not None and np.allclose(b, had):
                row["gate"] = "H"
        else:                                   # RX / RY / RZ
            row.update(target=a, angle=b)
        rows.append(row)
        step += 1
    df = pd.DataFrame(rows)
    df[["control", "target"]] = df[["control", "target"]].astype("Int64")
    df["cnot_cum"] = (df["gate"] == "CNOT").cumsum()
    return df


# ------------------------------------------------ 3) build both circuits --------
def build_circuits(U, n):
    out = {}

    with tagging(q36, leaf_size=1, leaf_name="1q gate"):
        ops = []
        q36.qsd(ops, U, list(range(n)))
    out["QSD l=1 (36 CNOT)"] = (ops, q36)

    with tagging(q20, leaf_size=2, leaf_name="2q block"):
        ops = q20.lower_cz(q20.synthesize(U))      # CZ -> H.CNOT.H so every entangler is a CNOT
    out["QSD optimized (20 CNOT)"] = (ops, q20)
    return out


def verify(ops, mod, U, n):
    C = mod.circuit_unitary(strip_marks(ops), n)
    ph = np.trace(U.conj().T @ C)
    return np.linalg.norm(C / (ph / abs(ph)) - U)       # error up to global phase


# ---------------------------------------------------------- 4) analysis ---------
def summarize(df):
    g = df.groupby("version")
    summary = pd.DataFrame({
        "total_gates": g.size(),
        "CNOT": g["gate"].apply(lambda s: (s == "CNOT").sum()),
        "1q_gates": g["gate"].apply(lambda s: s.isin(["U", "H", "RX", "RY", "RZ"]).sum()),
        "rotations": g["gate"].apply(lambda s: s.isin(["RX", "RY", "RZ"]).sum()),
    })
    summary["CNOT_saved_vs_l1"] = summary["CNOT"].max() - summary["CNOT"]
    return summary


def cnots_per_block_type(df):
    cn = df[df.gate == "CNOT"]
    t = cn.groupby(["version", "block_type"]).size().unstack(0).fillna(0).astype(int)
    t.loc["TOTAL"] = t.sum()
    return t


def cnots_per_block(df):
    cn = df[df.gate == "CNOT"]
    return (cn.groupby(["version", "block_id", "block"]).size()
              .rename("CNOTs").reset_index())


def cnots_per_pair(df):
    cn = df[df.gate == "CNOT"]
    return (cn.groupby(["version", "control", "target"]).size()
              .unstack(0).fillna(0).astype(int))


def plot_comparison(df, filename="qsd_comparison.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for version, d in df.groupby("version"):
        ax1.step(d["step"], d["cnot_cum"], where="post", label=version, lw=2)
    ax1.set_xlabel("gate index (time order)")
    ax1.set_ylabel("cumulative CNOT count")
    ax1.set_title("CNOTs along the circuit")
    ax1.grid(alpha=0.3)
    ax1.legend()

    t = cnots_per_block_type(df).drop(index="TOTAL")
    t.plot.bar(ax=ax2, rot=0)
    ax2.set_xlabel("block type")
    ax2.set_ylabel("CNOTs")
    ax2.set_title("CNOTs per block type")
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)
    return filename


# ---------------------------------------------------------------- main ----------
if __name__ == "__main__":
    pd.set_option("display.width", 140, "display.max_columns", 20, "display.max_rows", 200)

    n = 3
    U = unitary_group.rvs(2 ** n, random_state=7)
    circuits = build_circuits(U, n)

    frames = []
    for version, (ops, mod) in circuits.items():
        print(f"{version:<26} reconstruction error = {verify(ops, mod, U, n):.2e}")
        frames.append(ops_to_df(ops, version, had=q20.HAD))
    df = pd.concat(frames, ignore_index=True)

    print("\n=== summary ===")
    print(summarize(df))
    print("\n=== CNOTs per block type ===")
    print(cnots_per_block_type(df))
    print("\n=== CNOTs per block (optimized version) ===")
    cb = cnots_per_block(df)
    print(cb[cb.version.str.contains("optimized")].to_string(index=False))
    print("\n=== CNOTs per (control, target) pair ===")
    print(cnots_per_pair(df))
    print("\n=== gate counts by type ===")
    print(df.groupby(["version", "gate"]).size().unstack(0).fillna(0).astype(int))

    # exports (matrices cannot go into a CSV cell, so they are dropped there)
    df.drop(columns="matrix").to_csv("qsd_circuit_gates.csv", index=False)
    df.to_pickle("qsd_circuit_gates.pkl")
    print("\nfigure:", plot_comparison(df))

    # drawings of both circuits (marks removed)
    q36.draw_matplotlib(strip_marks(circuits["QSD l=1 (36 CNOT)"][0]), n, "qsd_36_circuit.png",
                        title="QSD l=1: 36 CNOTs")
    q20.draw_matplotlib(strip_marks(circuits["QSD optimized (20 CNOT)"][0]), n, "qsd_optimized_circuit.png",
                        title="QSD optimized: 20 CNOTs")
    print("saved: qsd_circuit_gates.csv / .pkl, qsd_comparison.png, qsd_36_circuit.png, qsd_optimized_circuit.png")
