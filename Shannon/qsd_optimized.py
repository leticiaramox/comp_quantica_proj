"""
Optimized Quantum Shannon Decomposition (QSD, l = 2)
Shende, Bullock, Markov - "Synthesis of Quantum-Logic Circuits", IEEE TCAD 25(6), 2006.

Ingredients (Table I, row "QSD (l=2, optimized)":  n=2 ->3, n=3 ->20, n=4 ->100 CNOTs)
  * Thm 10/12/13 : CSD + demultiplexing -> recursion on (n-1)-qubit operators
  * recursion stops at 2-qubit operators ("blocks")           c_2 = 3, not 6
  * Appendix A   : central multiplexed Ry built from controlled-Z gates; the terminal CZ is
                   itself a multiplexor (select = top wire) and is absorbed into the left
                   multiplexor                                     saves (4^(n-2)-1)/3 CNOTs
  * Appendix B   : Thm 14 - a 2-qubit block = (2-CNOT circuit) . (diagonal). The diagonal
                   commutes through the multiplexors (their controls are the low wires) and
                   is merged into the previous block                saves 4^(n-2)-1 CNOTs
  n = 3 :  4*3 + 3*4 - 1 - 3 = 20      (general: 23/48 4^n - 3/2 2^n + 4/3)

Conventions: wire 0 = most significant qubit (kron order). Circuit = list of ops in TIME order.
Requires numpy, scipy, matplotlib (only for drawing).
"""
import numpy as np
from scipy.linalg import cossin, schur, eigh
from scipy.stats import unitary_group

# ------------------------------------------------------------------ gates ---
I2 = np.eye(2, dtype=complex)
PX = np.array([[0, 1], [1, 0]], dtype=complex)
PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
PZ = np.array([[1, 0], [0, -1]], dtype=complex)
HAD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

def Rx(t): c, s = np.cos(t / 2), np.sin(t / 2); return np.array([[c, -1j * s], [-1j * s, c]])
def Ry(t): c, s = np.cos(t / 2), np.sin(t / 2); return np.array([[c, -s], [s, c]], dtype=complex)
def Rz(t): return np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)])

CNOT01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)  # ctrl = 1st factor
CNOT10 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)  # ctrl = 2nd factor

# op = (kind, a, b):  ("U", wire, 2x2) | ("RX"/"RY"/"RZ", wire, angle)
#                     ("CNOT", ctrl, tgt) | ("CZ", w1, w2) | ("BLOCK", (w0, w1), 4x4)  [placeholder]

# --------------------------------- multiplexed rotation (Thm 8, Gray code) ---
def mux_rotation(ops, axis, target, controls, angles, entangler="CNOT"):
    """Uniformly controlled Ry/Rz: angles[j] is applied to `target` when the control
    register (controls[0] = MSB) holds j. Uses 2^k entangling gates (CNOT, or CZ for Ry)."""
    k = len(controls)
    angles = np.asarray(angles, dtype=float)
    if k == 0:
        ops.append((axis, target, angles[0])); return
    gray = lambda i: i ^ (i >> 1)
    M = np.array([[(-1) ** bin(j & gray(i)).count("1") for j in range(2 ** k)] for i in range(2 ** k)])
    phi = M @ angles / 2 ** k
    for i in range(2 ** k):
        ops.append((axis, target, phi[i]))
        bit = (gray(i) ^ gray((i + 1) % 2 ** k)).bit_length() - 1
        ops.append((entangler, controls[k - 1 - bit], target))

# ------------------------------------------------ demultiplexing (Thm 12) ---
def demux(ops, U0, U1, wires):
    """U0 (+) U1 (select = wires[0])  =  (I x V)(D (+) D^dag)(I x W)."""
    T, V = schur(U0 @ U1.conj().T, output="complex")
    D = np.exp(0.5j * np.angle(np.diag(T)))
    W = np.diag(D) @ V.conj().T @ U1
    rest = wires[1:]
    qsd(ops, W, rest)
    mux_rotation(ops, "RZ", wires[0], rest, -2 * np.angle(D))
    qsd(ops, V, rest)

# ------------------------------------------------------------- QSD (Thm 13) --
def qsd(ops, U, wires):
    n = len(wires)
    if n == 2:                                   # stop at 2-qubit blocks (l = 2)
        ops.append(("BLOCK", tuple(wires), U)); return
    h = 2 ** (n - 1)
    (u1, u2), theta, (v1h, v2h) = cossin(U, p=h, q=h, separate=True)
    demux(ops, v1h, v2h, wires)                                   # right factor
    mux_rotation(ops, "RY", wires[0], wires[1:], 2 * theta, entangler="CZ")   # central Ry via CZ
    # Appendix A: (u1 (+) u2) . CZ = u1 (+) (u2 . Z_{wires[1]})  -> drop the terminal CZ
    assert ops.pop() == ("CZ", wires[1], wires[0])
    u2 = u2 @ np.kron(PZ, np.eye(h // 2))
    demux(ops, u1, u2, wires)                                     # left factor

# ================================================== two-qubit synthesis =====
MAGIC = np.array([[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]]) / np.sqrt(2)
_XYZ = [np.kron(P, P) for P in (PX, PY, PZ)]
_SIGNS = np.array([np.real(np.diag(MAGIC.conj().T @ PP @ MAGIC)) for PP in _XYZ]).T   # 4x3
_LIN = np.hstack([np.ones((4, 1)), _SIGNS])                                            # [1, x_k, y_k, z_k]

def N_can(a, b, c):
    """exp(i(a XX + b YY + c ZZ))"""
    return MAGIC @ np.diag(np.exp(1j * (_SIGNS @ np.array([a, b, c])))) @ MAGIC.conj().T

def kron_factor(K):
    """K = a (x) b  ->  (a, b)"""
    R = K.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    u, s, vh = np.linalg.svd(R)
    assert s[1] < 1e-8 * max(1, s[0]), "not a local operator"
    return (u[:, 0] * np.sqrt(s[0])).reshape(2, 2), (vh[0] * np.sqrt(s[0])).reshape(2, 2)

def kak(U):
    """U = e^{ig} A N(a,b,c) B, with A, B local (kron products). Returns A, B, (g,a,b,c)."""
    U = U / np.linalg.det(U) ** 0.25
    UB = MAGIC.conj().T @ U @ MAGIC
    S = UB.T @ UB                                   # symmetric unitary: Re, Im commute
    _, O = eigh(S.real + np.sqrt(2) * np.pi * 0.1 * S.imag)
    if np.linalg.det(O) < 0: O[:, 0] *= -1
    Lam = np.diag(O.T @ S @ O)
    D = np.sqrt(Lam)
    O1 = UB @ O @ np.diag(1 / D)
    if np.linalg.det(O1.real) < 0: D[0] *= -1; O1[:, 0] *= -1
    assert np.linalg.norm(O1.imag) < 1e-8
    O1 = O1.real
    A = MAGIC @ O1 @ MAGIC.conj().T
    B = MAGIC @ O.T @ MAGIC.conj().T
    g, a, b, c = np.linalg.solve(_LIN, np.angle(D))
    return A, B, (g, a, b, c)

_R = (I2 - 1j * (PX + PY + PZ)) / 2                # cyclic Clifford
if not np.allclose(_R @ PX @ _R.conj().T, PY):
    _R = _R.conj().T
assert np.allclose(_R @ PX @ _R.conj().T, PY) and np.allclose(_R @ PY @ _R.conj().T, PZ)

def _local_ops(K, w0, w1):
    a, b = kron_factor(K)
    return [("U", w0, a), ("U", w1, b)]

def synth_3cnot(M, w0, w1):
    """Any 2-qubit unitary with 3 CNOTs (Vatan-Williams form, derived via SWAP = C10 C01 C10)."""
    A, B, (g, a, b, c) = kak(M)
    p, q, r = np.pi / 2 - 2 * c, 2 * a - np.pi / 2, np.pi / 2 - 2 * b
    ops = _local_ops(B, w0, w1)
    ops += [("RZ", w0, -np.pi / 2), ("CNOT", w1, w0), ("RY", w1, r), ("CNOT", w0, w1),
            ("RZ", w0, p), ("RY", w1, q), ("CNOT", w1, w0), ("RZ", w1, np.pi / 2)]
    return ops + _local_ops(A, w0, w1)

def synth_2cnot(V, w0, w1):
    """2-qubit unitary with a vanishing canonical coordinate -> 2 CNOTs."""
    A, B, (g, *coords) = kak(V)
    half = np.pi / 2
    m = [round(x / half) for x in coords]
    res = [abs(x - mi * half) for x, mi in zip(coords, m)]
    j = int(np.argmin(res))
    assert res[j] < 1e-6, "block is not in the 2-CNOT class"
    P = [PX, PY, PZ][j]
    extra = (1j ** m[j]) * np.kron(np.linalg.matrix_power(P, m[j] % 2), np.linalg.matrix_power(P, m[j] % 2))
    coords[j] = 0.0
    k = {1: 0, 0: 1, 2: 2}[j]                        # rotate axes so the zero sits on YY
    for _ in range(k):
        coords = [coords[2], coords[0], coords[1]]
    Rk = np.linalg.matrix_power(_R, k)
    RR = np.kron(Rk, Rk)
    A2 = A @ RR.conj().T
    B2 = RR @ extra @ B
    x, _, z = coords
    ops = _local_ops(B2, w0, w1)
    ops += [("CNOT", w0, w1), ("RX", w0, -2 * x), ("RZ", w1, -2 * z), ("CNOT", w0, w1)]
    return ops + _local_ops(A2, w0, w1)

YY4 = np.kron(PY, PY)
def find_diag(M):
    """Return diagonal d0 (length 4) such that  M . diag(d0)  is implementable with 2 CNOTs
    (Appendix B / Thm 14). Uses Im tr[V (Y(x)Y) V^T (Y(x)Y)] = 0  <=>  a canonical coordinate is 0."""
    X = (M / np.linalg.det(M) ** 0.25).conj().T
    G = X @ YY4 @ X.T @ YY4
    a, b = G[0, 0] + G[3, 3], G[1, 1] + G[2, 2]
    s = np.arctan2(a.imag + b.imag, a.real - b.real)
    return np.array([np.exp(1j * s), np.exp(-1j * s), 1, 1])

def finalize_blocks(ops):
    """Replace 2-qubit BLOCK placeholders by circuits; block t = (2-CNOT circuit).(diag) and the
    diagonal is merged (through the multiplexors) into block t-1. The first block uses 3 CNOTs."""
    idx = [i for i, op in enumerate(ops) if op[0] == "BLOCK"]
    circuits, carry = [None] * len(idx), None
    for t in range(len(idx) - 1, -1, -1):
        (w0, w1), M = ops[idx[t]][1], ops[idx[t]][2]
        if carry is not None:
            M = np.diag(carry) @ M                    # diagonal of block t+1 arrives here
        if t > 0:
            d0 = find_diag(M)
            circuits[t] = synth_2cnot(M @ np.diag(d0), w0, w1)
            carry = np.conj(d0)                       # M = V . diag(conj d0): diagonal applied first
        else:
            circuits[t] = synth_3cnot(M, w0, w1)
    out, t = [], 0
    for op in ops:
        if op[0] == "BLOCK": out += circuits[t]; t += 1
        else: out.append(op)
    return out

def synthesize(U, optimize=True):
    n = int(np.log2(U.shape[0]))
    ops = []
    qsd(ops, U, list(range(n)))
    return finalize_blocks(ops)

def lower_cz(ops):
    """CZ(a,b) = (I x H) CNOT(a,b) (I x H): every CZ costs exactly one CNOT."""
    out = []
    for op in ops:
        if op[0] == "CZ": out += [("U", op[2], HAD), ("CNOT", op[1], op[2]), ("U", op[2], HAD)]
        else: out.append(op)
    return out

# ------------------------------------------------------ simulation / checks --
def op_matrix(op, n):
    kind, a, b = op
    dim = 2 ** n
    if kind in ("CNOT", "CZ"):
        M = np.zeros((dim, dim), dtype=complex)
        for x in range(dim):
            bits = [(x >> (n - 1 - w)) & 1 for w in range(n)]
            ph = 1
            if bits[a]:
                if kind == "CNOT": bits[b] ^= 1
                elif bits[b]: ph = -1
            M[sum(bit << (n - 1 - w) for w, bit in enumerate(bits)), x] = ph
        return M
    g = {"U": lambda: b, "RX": lambda: Rx(b), "RY": lambda: Ry(b), "RZ": lambda: Rz(b)}[kind]()
    mats = [I2] * n
    mats[a] = g
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

def circuit_unitary(ops, n):
    M = np.eye(2 ** n, dtype=complex)
    for op in ops:
        M = op_matrix(op, n) @ M
    return M

def count_cnots(ops):      return sum(op[0] == "CNOT" for op in ops)
def count_2q(ops):         return sum(op[0] in ("CNOT", "CZ") for op in ops)

def dist_up_to_phase(A, B):
    ph = np.trace(B.conj().T @ A)
    ph = ph / abs(ph)
    return np.linalg.norm(A / ph - B)

# ----------------------------------------------------------------- drawing ---
def draw_matplotlib(ops, n, filename="qsd_optimized_circuit.png", gates_per_row=22, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    rows = [ops[i:i + gates_per_row] for i in range(0, len(ops), gates_per_row)]
    fig, axes = plt.subplots(len(rows), 1, figsize=(gates_per_row * 0.55 + 1, (n * 0.75 + 0.6) * len(rows)))
    axes = np.atleast_1d(axes)
    colors = {"U": "#dbe4ff", "H": "#f3d9fa", "RX": "#ffd8d8", "RY": "#ffe3b3", "RZ": "#d3f0d8"}
    for r, (ax, row) in enumerate(zip(axes, rows)):
        ax.set_xlim(-1.2, gates_per_row + 0.2); ax.set_ylim(-(n - 0.4), 0.6)
        ax.set_aspect("equal"); ax.axis("off")
        for w in range(n):
            ax.plot([-0.4, gates_per_row], [-w, -w], color="black", lw=1, zorder=0)
            if r == 0: ax.text(-0.6, -w, f"q{w}", ha="right", va="center", fontsize=10)
        for x, (kind, a, b) in enumerate(row):
            if kind == "CNOT":
                ax.plot([x, x], [-a, -b], color="black", lw=1.2, zorder=1)
                ax.add_patch(Circle((x, -a), 0.09, color="black", zorder=3))
                ax.add_patch(Circle((x, -b), 0.22, fc="white", ec="black", lw=1.2, zorder=2))
                ax.plot([x - 0.22, x + 0.22], [-b, -b], color="black", lw=1.2, zorder=3)
                ax.plot([x, x], [-b - 0.22, -b + 0.22], color="black", lw=1.2, zorder=3)
            elif kind == "CZ":
                ax.plot([x, x], [-a, -b], color="black", lw=1.2, zorder=1)
                for w in (a, b): ax.add_patch(Circle((x, -w), 0.09, color="black", zorder=3))
            else:
                lab = {"U": "U", "RX": "Rx", "RY": "Ry", "RZ": "Rz"}[kind]
                if kind == "U" and np.allclose(b, HAD): lab = "H"
                ax.add_patch(Rectangle((x - 0.3, -a - 0.3), 0.6, 0.6, fc=colors[lab if lab == "H" else kind],
                                       ec="black", lw=1, zorder=2))
                ax.text(x, -a, lab, ha="center", va="center", fontsize=8, zorder=3)
    if title: fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(filename, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return filename

# -------------------------------------------------------------------- main ---
if __name__ == "__main__":
    n = 3
    U = unitary_group.rvs(2 ** n, random_state=7)
    ops = synthesize(U)
    print(f"gates: {len(ops)}   CNOT+CZ: {count_2q(ops)}   error: {dist_up_to_phase(circuit_unitary(ops, n), U):.2e}")
    ops = lower_cz(ops)                              # every CZ -> H . CNOT . H
    err = dist_up_to_phase(circuit_unitary(ops, n), U)
    print(f"after CZ -> CNOT lowering: CNOTs = {count_cnots(ops)}  (paper: 20)   error = {err:.2e}")
    assert count_cnots(ops) == 20 and err < 1e-9
    print("figure:", draw_matplotlib(ops, n, "qsd_optimized_circuit.png",
                                     title=f"Optimized QSD, {n} qubits, {count_cnots(ops)} CNOTs"))
