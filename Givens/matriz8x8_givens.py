import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from scipy.stats import unitary_group

# 1. Algoritmo de Decomposição em 2 Níveis (Retorna submatrizes e seus índices)
def decompose_two_level_with_indices(U):
    N = U.shape[0]
    decompositions = []
    U_curr = U.copy().astype(complex)
    
    for j in range(N - 1):
        for i in range(N - 1, j, -1):
            a = U_curr[i - 1, j]
            b = U_curr[i, j]
            
            if np.abs(b) > 1e-10:
                r = np.sqrt(np.abs(a)**2 + np.abs(b)**2)
                c = a / r
                s = b / r
                
                # Submatriz 2x2 de rotação
                u_2x2 = np.array([[np.conj(c), np.conj(s)],
                                  [-s, c]])
                
                # Registra a submatriz e os índices i-1 e i onde ela atua
                decompositions.append({
                    'u_2x2': u_2x2.conj().T,
                    'state1': i - 1,
                    'state2': i
                })
                
                # Matriz 8x8 de 2 níveis para atualizar U_curr
                V_sub = np.eye(N, dtype=complex)
                V_sub[i-1, i-1] = np.conj(c)
                V_sub[i-1, i] = np.conj(s)
                V_sub[i, i-1] = -s
                V_sub[i, i] = c
                
                U_curr = V_sub @ U_curr
                
    # Adiciona elementos da diagonal restante, se houver
    return decompositions

# 2. Função de Construção da Etapa de Código de Gray 
def get_gray_code_path(state1_int, state2_int, num_qubits=3):
    b1 = format(state1_int, f'0{num_qubits}b')
    b2 = format(state2_int, f'0{num_qubits}b')
    current, target = list(map(int, b1)), list(map(int, b2))
    path = [list(current)]
    for k in range(num_qubits):
        if current[k] != target[k]:
            current[k] = target[k]
            path.append(list(current))
    return path

def apply_two_level_gate(qc, u_2x2, state1_int, state2_int):
    path = get_gray_code_path(state1_int, state2_int)
    
    # Permutação inicial
    for k in range(len(path) - 2):
        s_from, s_to = path[k], path[k+1]
        diff_idx = [i for i in range(3) if s_from[i] != s_to[i]][0]
        control_indices = [i for i in range(3) if i != diff_idx]
        
        for c in control_indices:
            if s_from[c] == 0: qc.x(c)
        qc.ccx(control_indices[0], control_indices[1], diff_idx)
        for c in control_indices:
            if s_from[c] == 0: qc.x(c)

    # Aplicação da porta U_til 2x2 controlada
    target_state, final_state = path[-2], path[-1]
    target_qubit = [i for i in range(3) if target_state[i] != final_state[i]][0]
    control_qubits = [i for i in range(3) if i != target_qubit]
    
    u_gate = UnitaryGate(u_2x2, label=r"Ũ").control(2)
    
    for c in control_qubits:
        if target_state[c] == 0: qc.x(c)
    qc.append(u_gate, control_qubits + [target_qubit])
    for c in control_qubits:
        if target_state[c] == 0: qc.x(c)

    # Uncomputation
    for k in reversed(range(len(path) - 2)):
        s_from, s_to = path[k], path[k+1]
        diff_idx = [i for i in range(3) if s_from[i] != s_to[i]][0]
        control_indices = [i for i in range(3) if i != diff_idx]
        
        for c in control_indices:
            if s_from[c] == 0: qc.x(c)
        qc.ccx(control_indices[0], control_indices[1], diff_idx)
        for c in control_indices:
            if s_from[c] == 0: qc.x(c)

# 3. Pipeline Principal de Integração
def build_full_unitary_circuit(U_8x8):
    # Fatora a matriz 8x8 nas etapas de 2 níveis
    decompositions = decompose_two_level_with_indices(U_8x8)
    
    qc = QuantumCircuit(3)
    
    # Aplica cada uma das matrizes de 2 níveis sequencialmente
    for step in decompositions:
        apply_two_level_gate(
            qc, 
            u_2x2=step['u_2x2'], 
            state1_int=step['state1'], 
            state2_int=step['state2']
        )
        qc.barrier() # Separa as etapas visualmente no circuito
        
    return qc, len(decompositions)

# --- Exemplo de Execução ---
U_8x8 = unitary_group.rvs(8) # Gera matriz 8x8 aleatória
circuit, num_gates = build_full_unitary_circuit(U_8x8)

print(f"Total de etapas de 2 níveis aplicadas: {num_gates}")
print(circuit.draw('text'))
