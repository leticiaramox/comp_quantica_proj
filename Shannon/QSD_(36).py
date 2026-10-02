import numpy as np
from scipy.linalg import cossin, schur
from scipy.stats import unitary_group

# ----------------------------------------------------------------- gates ---
def Ry(phi):
    c, s = np.cos(phi / 2), np.sin(phi / 2)
    return np.array([[c, s], [-s, c]], dtype=complex)

def Rz(phi):
    return np.diag([np.exp(-1j * phi / 2), np.exp(1j * phi / 2)])

# ------------------------------------------- multiplexed rotation (Thm 8) ---
def mux_rotation(ops, axis, target, controls, angles):
    k = len(controls)
    angles = np.asarray(angles, dtype=float)
    assert len(angles) == 2 ** k
    if k == 0:
        ops.append((axis, target, angles[0]))
        return
    gray = lambda i: i ^ (i >> 1)
    M = np.array([[(-1) ** bin(j & gray(i)).count("1") for j in range(2 ** k)]
                  for i in range(2 ** k)])
    phi = M @ angles / 2 ** k
    for i in range(2 ** k):
        ops.append((axis, target, phi[i]))
        diff = gray(i) ^ gray((i + 1) % 2 ** k)
        bit = diff.bit_length() - 1
        ops.append(("CNOT", controls[k - 1 - bit], target))

# --------------------------------------------- demultiplexing (Thm 12) -----
def demux(ops, U0, U1, wires):
    T, V = schur(U0 @ U1.conj().T, output="complex")
    D = np.exp(0.5j * np.angle(np.diag(T)))
    W = np.diag(D) @ V.conj().T @ U1
    rest = wires[1:]
    qsd(ops, W, rest)
    mux_rotation(ops, "RZ", wires[0], rest, -2 * np.angle(D))
    qsd(ops, V, rest)

# ------------------------------------------------------- QSD (Thm 13) ------
def qsd(ops, U, wires):
    n = len(wires)
    if n == 1:
        ops.append(("U", wires[0], U))
        return
    h = 2 ** (n - 1)
    (u1, u2), theta, (v1h, v2h) = cossin(U, p=h, q=h, separate=True)
    demux(ops, v1h, v2h, wires)
    mux_rotation(ops, "RY", wires[0], wires[1:], -2 * theta)
    demux(ops, u1, u2, wires)

# -------------------------------------------------- simulando --
def op_matrix(op, n):
    kind, a, b = op
    if kind == "CNOT":
        dim = 2 ** n
        M = np.zeros((dim, dim), dtype=complex)
        for x in range(dim):
            bits = [(x >> (n - 1 - w)) & 1 for w in range(n)]
            if bits[a]:
                bits[b] ^= 1
            y = sum(bit << (n - 1 - w) for w, bit in enumerate(bits))
            M[y, x] = 1
        return M
    g = {"U": lambda: b, "RY": lambda: Ry(b), "RZ": lambda: Rz(b)}[kind]()
    mats = [np.eye(2)] * n
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

def count_cnots(ops):
    return sum(op[0] == "CNOT" for op in ops)

def draw(ops, n, gates_per_row=25):
    rows = [ops[i:i + gates_per_row] for i in range(0, len(ops), gates_per_row)]
    rendered_blocks = []
    
    for r_idx, row in enumerate(rows):
        start_gate = r_idx * gates_per_row
        end_gate = min((r_idx + 1) * gates_per_row, len(ops))
        block_str = [f"--- Portas {start_gate} a {end_gate - 1} ---"]
        
        wire_lines = [f"q{w}: " for w in range(n)]
        
        for kind, a, b in row:
            for w in range(n):
                if kind == "CNOT":
                    if w == a:
                        wire_lines[w] += "─●─"
                    elif w == b:
                        wire_lines[w] += "─⊕─"
                    elif min(a, b) < w < max(a, b):
                        wire_lines[w] += "─│─"
                    else:
                        wire_lines[w] += "───"
                else:
                    if w == a:
                        wire_lines[w] += f"[{kind:^2}]"
                    else:
                        wire_lines[w] += "────"
                        
        block_str.extend(wire_lines)
        rendered_blocks.append("\n".join(block_str))
        
    return "\n\n".join(rendered_blocks)

# ------------------------------------------------------------------- main ---
if __name__ == "__main__":
    n = 3
    U = unitary_group.rvs(2 ** n, random_state=7)
    ops = []
    qsd(ops, U, list(range(n)))

    err = np.linalg.norm(circuit_unitary(ops, n) - U)
    print(f"qubits            : {n}")
    print(f"total gates       : {len(ops)}")
    print(f"nº de CNOT        : {count_cnots(ops)}   (paper, QSD l=1: 36)")
    print(f"||U_circ - U||_F  : {err:.2e}\n")
    assert count_cnots(ops) == 36 and err < 1e-9

    # Print full circuit diagram directly to terminal
    print("=" * 80)
    print("CIRCUITO GERADO")
    print("=" * 80)
    print(draw(ops, n, gates_per_row=22))
