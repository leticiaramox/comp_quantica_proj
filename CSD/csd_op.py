import numpy as np
import scipy.linalg as la
from scipy.linalg import cossin
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, random_unitary, Statevector
from qclib.state_preparation import LowRankInitialize


# =====================================================================
# 1. GRAY-CODE UNIFORMLY CONTROLLED Ry/Rz  (Sec. 3.1, Fig. 3)
# =====================================================================
def gerar_codigo_gray(n_bits):
    if n_bits == 0:
        return ['']
    return [bin(i ^ (i >> 1))[2:].zfill(n_bits) for i in range(2**n_bits)]


def identificar_bit_alterado(gray1, gray2):
    for idx, (b1, b2) in enumerate(zip(reversed(gray1), reversed(gray2))):
        if b1 != b2:
            return idx
    return 0


def aplicar_multiplexed_r(qc, tipo_porta, angles, target_qubit, control_qubits):
    k = len(control_qubits)
    angles = np.asarray(angles, dtype=float)
    if len(angles) != 2**k:
        raise ValueError("len(angles) must be 2**len(control_qubits)")
    gate = qc.ry if tipo_porta == 'y' else qc.rz
    if k == 0:
        gate(angles[0], target_qubit)
        return

    N = 2**k
    gray_seq = gerar_codigo_gray(k)
    M = np.zeros((N, N))
    for i in range(N):
        bits_i = [int(b) for b in bin(i)[2:].zfill(k)]
        for j in range(N):
            bits_g = [int(b) for b in gray_seq[j]]
            M[i, j] = (-1) ** sum(bi * bg for bi, bg in zip(bits_i, bits_g))
    alphas = (M.T @ angles) / N

    for j in range(N):
        gate(alphas[j], target_qubit)
        bit = identificar_bit_alterado(gray_seq[j], gray_seq[(j + 1) % N])
        qc.cx(control_qubits[bit], target_qubit)


def aplicar_diagonal(qc, fases, qubits_le):
    fases = np.asarray(fases, dtype=float)
    m = len(qubits_le)
    if m == 1:
        qc.rz(fases[1] - fases[0], qubits_le[0])
        qc.global_phase += (fases[0] + fases[1]) / 2
        return
    half = 2**(m - 1)
    p0, p1 = fases[:half], fases[half:]
    aplicar_multiplexed_r(qc, 'z', p1 - p0, qubits_le[-1], list(qubits_le[:-1]))
    aplicar_diagonal(qc, (p0 + p1) / 2, list(qubits_le[:-1]))


# =====================================================================
# 2. F~ : UNIFORMLY CONTROLLED ONE-QUBIT GATE UP TO A DIAGONAL (Sec. 3.2)
# =====================================================================
def factor_onefold_multiplexor(a, b, tol=1e-8):
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    X = a @ b.conj().T
    phi = np.angle(la.det(X))
    arg_x1 = np.angle((X * np.exp(-0.5j * phi))[0, 0])

    r = np.diag([np.exp(0.5j * (-np.pi / 2 - phi / 2 - arg_x1)),
                 np.exp(0.5j * (np.pi / 2 - phi / 2 + arg_x1))])
    d = np.diag([np.exp(1j * np.pi / 4), np.exp(-1j * np.pi / 4)])

    T, Z = la.schur(r @ X @ r, output='complex')
    i_plus = int(np.argmin(np.abs(np.diag(T) - 1j)))
    u = Z[:, [i_plus, 1 - i_plus]]
    v = d @ u.conj().T @ r.conj().T @ b

    err = max(la.norm(a - r.conj().T @ u @ d @ v), la.norm(b - r @ u @ d.conj().T @ v))
    if err > tol:
        raise ValueError(f"Multiplexor factorization failed: {err:.2e}")
    return r, u, d, v


def decompose_ucg_tilde(blocks, controls, target, tol=1e-8):
    blocks = [np.asarray(U, dtype=complex) for U in blocks]
    k = len(controls)
    if len(blocks) != 2**k:
        raise ValueError("The number of blocks must equal 2**len(controls).")
    if k == 0:
        return [("U", target, blocks[0])], np.ones(2, dtype=complex)

    half = len(blocks) // 2
    r_list, u_blocks, v_blocks = [], [], []
    for j in range(half):
        r, u, _, v = factor_onefold_multiplexor(blocks[j], blocks[j + half], tol=tol)
        r_list.append(r)
        u_blocks.append(u)
        v_blocks.append(v)

    ops_v, delta_v = decompose_ucg_tilde(v_blocks, controls[1:], target, tol=tol)
    u_adj = [u_blocks[j] @ np.diag(delta_v[2*j:2*j+2]) for j in range(half)]
    ops_u, delta_u = decompose_ucg_tilde(u_adj, controls[1:], target, tol=tol)

    delta_r = np.empty(2 * len(blocks), dtype=complex)
    for j, r in enumerate(r_list):
        delta_r[2*j:2*j+2] = np.diag(r.conj().T)
        delta_r[2*(j+half):2*(j+half)+2] = np.diag(r)
    return ops_v + [("D", controls[0], target)] + ops_u, delta_r * np.tile(delta_u, 2)


def emitir_ops(qc, operations, qubits):
    for item in operations:
        if item[0] == "U":
            qc.unitary(item[2], [qubits[item[1]]], label="U")
        else:
            _, c, t = item
            c, t = qubits[c], qubits[t]
            qc.h(t)
            qc.cx(c, t)
            qc.h(t)
            qc.sdg(c)
            qc.sdg(t)
            qc.global_phase += np.pi / 4


# =====================================================================
# 3. RECURSIVE CSD  ->  LIST OF UNIFORMLY CONTROLLED ONE-QUBIT LAYERS (Eq. 31)
# =====================================================================
def _canonico(n, target, cl, blocos):
    others = [q for q in range(n) if q != target]
    pos = {q: p for p, q in enumerate(others)}
    out = [None] * 2**(n - 1)
    for jc in range(2**(n - 1)):
        idx = 0
        for p, q in enumerate(cl):
            idx |= ((jc >> pos[q]) & 1) << p
        out[jc] = blocos[idx]
    return out


def camadas_csd(Us, qubits, ctrl, n):
    m = len(qubits)
    if m == 1:
        return [(qubits[0], _canonico(n, qubits[0], ctrl, list(Us)))]

    half = 2**(m - 1)
    top, lower = qubits[-1], qubits[:-1]
    cs = [cossin(U, p=half, q=half, separate=True) for U in Us]

    U_dir = [c[2][0] for c in cs] + [c[2][1] for c in cs]
    U_esq = [c[0][0] for c in cs] + [c[0][1] for c in cs]

    ry = [None] * (len(Us) * half)
    for l in range(half):
        for j in range(len(Us)):
            th = cs[j][1][l]
            ry[j + len(Us) * l] = np.array([[np.cos(th), -np.sin(th)],
                                            [np.sin(th), np.cos(th)]], dtype=complex)
    camada_ry = (top, _canonico(n, top, ctrl + lower, ry))

    return (camadas_csd(U_dir, lower, ctrl + [top], n)
            + [camada_ry]
            + camadas_csd(U_esq, lower, ctrl + [top], n))


# =====================================================================
# 4. LAST LAYER: multiplexor with r = I  ->  leftover (n-1)-qubit diagonal
# =====================================================================
def ultima_camada(qc, blocos, controls, t, qubits):
    half = len(blocos) // 2
    c0, rest = controls[0], controls[1:]
    u_blocks, v_blocks, thetas = [], [], np.zeros((half, 2))

    for j in range(half):
        a, b = blocos[j], blocos[j + half]
        T, V = la.schur(a @ b.conj().T, output='complex')
        ev = np.diag(T) / np.abs(np.diag(T))
        th = np.angle(ev) / 2
        d = np.exp(1j * th)
        v = d[:, None] * (V.conj().T @ b)
        assert la.norm(a - V @ np.diag(d) @ v) < 1e-8
        assert la.norm(b - V @ np.diag(d.conj()) @ v) < 1e-8
        u_blocks.append(V)
        v_blocks.append(v)
        thetas[j] = th

    ops_v, delta_v = decompose_ucg_tilde(v_blocks, rest, t)
    emitir_ops(qc, ops_v, qubits)

    ang = np.empty(2 * half)
    for j in range(half):
        for bit in (0, 1):
            ang[j + half * bit] = -2.0 * thetas[j, bit]
    ctrl_le = [qubits[q] for q in rest[::-1]] + [qubits[t]]
    aplicar_multiplexed_r(qc, 'z', ang, qubits[c0], ctrl_le)

    u_adj = [u_blocks[j] @ np.diag(delta_v[2*j:2*j+2]) for j in range(half)]
    ops_u, delta_u = decompose_ucg_tilde(u_adj, rest, t)
    emitir_ops(qc, ops_u, qubits)

    aplicar_diagonal(qc, np.angle(delta_u), [qubits[q] for q in (rest + [t])[::-1]])


# =====================================================================
# 5. FULL ALGORITHM
# =====================================================================
def decompor_unitaria(qc, U, qubits):
    U = np.asarray(U, dtype=complex)
    qubits = list(qubits)
    n = len(qubits)
    if U.shape != (2**n, 2**n):
        raise ValueError("U must be 2^n x 2^n, n = len(qubits).")
    if n == 1:
        qc.unitary(U, qubits, label="U")
        return

    camadas = camadas_csd([U], list(range(n)), [], n)
    pend = np.ones(2**n, dtype=complex)

    for k, (t, blocos) in enumerate(camadas):
        others = [q for q in range(n) if q != t]
        gidx = np.array([[(b << t) | sum(((j >> p) & 1) << q for p, q in enumerate(others))
                          for b in (0, 1)] for j in range(2**(n - 1))])

        blocos = [blocos[j] * pend[gidx[j]][None, :] for j in range(2**(n - 1))]
        controls = others[::-1]

        if k < len(camadas) - 1:
            ops, delta = decompose_ucg_tilde(blocos, controls, t)
            emitir_ops(qc, ops, qubits)
            pend = np.empty(2**n, dtype=complex)
            pend[gidx.reshape(-1)] = delta
        else:
            ultima_camada(qc, blocos, controls, t, qubits)


def cnots_esperados(n):
    return 4**n // 2 - 2**(n - 1) - 2


# =====================================================================
# 6. PEEPHOLE: CX · CX = I
# =====================================================================
def otimizar_cancelamento_cnot(qc):
    instrs = list(qc.data)
    manter = [True] * len(instrs)
    pilhas = {}
    for i, ins in enumerate(instrs):
        qs = tuple(ins.qubits)
        if ins.operation.name == 'cx':
            sc, st = pilhas.get(qs[0], []), pilhas.get(qs[1], [])
            if sc and st and sc[-1] == st[-1]:
                j = sc[-1]
                if instrs[j].operation.name == 'cx' and tuple(instrs[j].qubits) == qs:
                    manter[i] = manter[j] = False
                    sc.pop()
                    st.pop()
                    continue
        for q in qs:
            pilhas.setdefault(q, []).append(i)
    novo = qc.copy_empty_like()
    for ins, ok in zip(instrs, manter):
        if ok:
            novo.append(ins.operation, ins.qubits, ins.clbits)
    return novo


def erro_operador(qc, U):
    Uc = Operator(qc).data
    fase = np.angle(np.trace(U.conj().T @ Uc))
    return np.linalg.norm(Uc - U), np.linalg.norm(Uc - np.exp(1j * fase) * U)


# =====================================================================
# 7. RANDOM-UNITARY BENCHMARK  (original)
# =====================================================================
def benchmark_random(ns=(2, 3, 4, 5, 6)):
    print("=" * 70)
    print("Benchmark: random unitaries")
    print("=" * 70)
    for n in ns:
        U = random_unitary(2**n).data
        qc = QuantumCircuit(n)
        decompor_unitaria(qc, U, list(range(n)))
        qc2 = otimizar_cancelamento_cnot(qc)
        e, ef = erro_operador(qc2, U)
        print(f"n={n}: CNOTs = {qc.count_ops().get('cx', 0):4d} "
              f"(expected {cnots_esperados(n):4d}; after cancellation "
              f"{qc2.count_ops().get('cx', 0):4d}) "
              f"| ||C-U|| = {e:.1e} (up to global phase {ef:.1e})")


# =====================================================================
# 8. QCLIB: decompose unitaries extracted from qclib state-prep
# =====================================================================
def benchmark_qclib(n_trials=5, seed=42):
    """
    For each trial:
      1. Draw a random normalized state vector of length 2^n.
      2. Build a qclib LowRankInitialize circuit for that state.
      3. Extract the unitary U that qclib's circuit implements.
      4. Decompose U with our CSD algorithm.
      5. Check that our circuit reproduces U and also prepares |psi>.
    """
    print()
    print("=" * 70)
    print("Benchmark: unitaries extracted from qclib LowRankInitialize")
    print("=" * 70)
    rnd = np.random.RandomState(seed)

    for n in (2, 3, 4):
        print(f"\n--- n = {n} qubit(s) ---")
        for trial in range(n_trials):
            # 1. Random normalized state
            psi = rnd.rand(2**n) + 1j * rnd.rand(2**n)
            psi /= np.linalg.norm(psi)

            # 2. qclib state-prep circuit
            prep_circuit = LowRankInitialize(psi).definition

            # 3. Unitary that qclib implements
            U_qclib = Operator(prep_circuit).data

            # 4. Our CSD decomposition of that unitary
            qc = QuantumCircuit(n)
            decompor_unitaria(qc, U_qclib, list(range(n)))
            qc2 = otimizar_cancelamento_cnot(qc)

            # 5a. Operator error
            e, ef = erro_operador(qc2, U_qclib)

            # 5b. State fidelity: both circuits should prepare the same state
            #     starting from |0...0>.  qclib may use a different qubit order,
            #     so we compare via the unitary columns (column 0 = action on |0>).
            sv_qclib = U_qclib[:, 0]           # qclib maps |0> -> psi (up to phase/order)
            sv_ours  = Operator(qc2).data[:, 0]
            fidelity = abs(np.dot(sv_qclib.conj(), sv_ours))

            our_cx   = qc2.count_ops().get('cx', 0)
            qclib_cx = prep_circuit.count_ops().get('cx', 0)

            print(f"  trial {trial+1}: ||C-U|| = {e:.1e}  (phase-free {ef:.1e})  "
                  f"fidelity = {fidelity:.6f}  "
                  f"CNOTs: ours={our_cx}  qclib={qclib_cx}  expected={cnots_esperados(n)}")


if __name__ == "__main__":
    benchmark_random(ns=(2, 3, 4, 5, 6))
    benchmark_qclib(n_trials=5)

    # Visual circuit for n=3 (original)
    U8 = random_unitary(8).data
    circ = QuantumCircuit(3)
    decompor_unitaria(circ, U8, [0, 1, 2])
    print()
    print(otimizar_cancelamento_cnot(circ).draw(output='text', fold=200))
