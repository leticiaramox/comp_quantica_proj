import numpy as np
import scipy.linalg as la
from scipy.linalg import cossin
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_unitary
from qiskit.synthesis import OneQubitEulerDecomposer

# =====================================================================
# 1. FUNÇÕES AUXILIARES DE CÓDIGO DE GRAY E ROTAÇÕES MULTIPLEXADAS
# =====================================================================
def gerar_codigo_gray(n_bits):
    if n_bits == 0:
        return ['']
    return [bin(i ^ (i >> 1))[2:].zfill(n_bits) for i in range(2**n_bits)]

def identificar_bit_alterado(gray1, gray2):
    """Retorna o índice do bit que mudou (da direita para a esquerda)."""
    for idx, (b1, b2) in enumerate(zip(reversed(gray1), reversed(gray2))):
        if b1 != b2:
            return idx
    return 0

def aplicar_multiplexed_r(qc, tipo_porta, angles, target_qubit, control_qubits):
    """
    Decompõe uma rotação multiplexada (Ry ou Rz) em uma sequência
    alternada de CNOTs e rotações de 1 qubit usando Código de Gray.
    """
    k = len(control_qubits)
    if k == 0:
        if tipo_porta == 'y':
            qc.ry(angles[0], target_qubit)
        else:
            qc.rz(angles[0], target_qubit)
        return

    N = 2**k
    gray_seq = gerar_codigo_gray(k)
    
    # Construção da Matriz de Transformação do Código de Gray (Walsh-Hadamard)
    M = np.zeros((N, N))
    for i in range(N):
        bits_i = [int(b) for b in bin(i)[2:].zfill(k)]
        for j in range(N):
            bits_g = [int(b) for b in gray_seq[j]]
            prod = sum(bi * bg for bi, bg in zip(bits_i, bits_g))
            M[i, j] = (-1) ** prod

    # Cálculo dos ângulos alfas das portas individuais
    alphas = (M.T @ angles) / N

    # Aplicação da sequência de portas CNOT e Rotações no circuito
    for j in range(N):
        if tipo_porta == 'y':
            qc.ry(alphas[j], target_qubit)
        else:
            qc.rz(alphas[j], target_qubit)
            
        # Determina qual qubit de controle altera o bit no Código de Gray
        proximo_j = (j + 1) % N
        bit_mudar = identificar_bit_alterado(gray_seq[j], gray_seq[proximo_j])
        qc.cx(control_qubits[bit_mudar], target_qubit)

# =====================================================================
# 2. DESMULTIPLEXAÇÃO DE BLOCOS DIAGONAIS (u1 ⊕ u2)
# =====================================================================
def demultiplex_block_diagonal(qc, u1, u2, qubits):
    """
    Fatoriza (u1 ⊕ u2) em matrizes não controladas e uma rotação multiplexada Rz.
    """
    target = qubits[-1]
    controls = qubits[:-1]
    
    # Eigen-decomposição de M = u1 @ u2^\dagger
    M = u1 @ u2.conj().T
    vals, V = la.eig(M)
    phases = np.angle(vals)
    thetas = phases / 2.0
    
    D = np.diag(np.exp(1j * thetas))
    W1 = V
    W2 = V.conj().T @ u2
    
    # 1. Aplica W2 nos qubits de controle (Recursivo)
    decompor_unitaria(qc, W2, controls)
    
    # 2. Aplica Multiplexed Rz no qubit alvo
    angles_rz = -2.0 * thetas
    aplicar_multiplexed_r(qc, 'z', angles_rz, target, controls)
    
    # 3. Aplica D e W1 nos qubits de controle (Recursivo)
    decompor_unitaria(qc, D, controls)
    decompor_unitaria(qc, W1, controls)

# =====================================================================
# 3. COMPILADOR RECURSIVO CSD (DECOMPOSIÇÃO DE SHANNON)
# =====================================================================
def decompor_unitaria(qc, U, qubits):
    """
    Decompõe recursivamente uma matriz unitária U em portas fundamentais.
    """
    dim = U.shape[0]
    
    # CASO BASE: 1 Qubit (Matriz 2x2) -> Decomposição Euler ZYZ (3 rotações)
    # CASO BASE: 1 Qubit (Matriz 2x2) decomposto do zero em Rz e Ry
    if dim == 2:
        # Normalização para SU(2) removendo fase global
        fase = np.sqrt(la.det(U))
        U_norm = U / fase if fase != 0 else U

        # Cálculo dos ângulos de Euler ZYZ
        theta = 2 * np.arccos(np.clip(np.abs(U_norm[0, 0]), 0, 1))
        
        angle1 = np.angle(U_norm[1, 1])
        angle2 = np.angle(U_norm[1, 0])
        
        phi = angle1 + angle2
        lam = angle1 - angle2

        qc.rz(lam, qubits[0])
        qc.ry(theta, qubits[0])
        qc.rz(phi, qubits[0])
        return

    # DECOMPOSIÇÃO CSD PARA MATRIZES MAIORES (Ex: 8x8 para 3 qubits)
    meio = dim // 2
    (u1, u2), thetas_csd, (v1h, v2h) = cossin(U, p=meio, q=meio, separate=True)
    
    target = qubits[-1]
    controls = qubits[:-1]

    # Passo 1: Decompor bloco V_CSD (Direita)
    demultiplex_block_diagonal(qc, v1h, v2h, qubits)

    # Passo 2: Rotação Multiplexada Ry central
    angles_ry = 2.0 * thetas_csd
    aplicar_multiplexed_r(qc, 'y', angles_ry, target, controls)

    # Passo 3: Decompor bloco U_CSD (Esquerda)
    demultiplex_block_diagonal(qc, u1, u2, qubits)

# =====================================================================
# 4. EXECUÇÃO, EXIBIÇÃO DO CIRCUITO E CONTAGEM DE PORTAS
# =====================================================================

# Criar matriz unitária aleatória de 8x8 (3 qubits)
matriz_8x8 = random_unitary(8).data

# Inicializar circuito de 3 qubits
circuito_customizado = QuantumCircuit(3)

# Compilar a matriz 8x8 gerando o circuito do zero
decompor_unitaria(circuito_customizado, matriz_8x8, qubits=[0, 1, 2])

# A) Desenhar o circuito gerado
print("=== DESENHO DO CIRCUITO DECOMPOSTO DO ZERO ===")
print(circuito_customizado.draw(output='text'))

# B) Contar e analisar as portas do circuito
contagem = circuito_customizado.count_ops()
total_portas = sum(contagem.values())

print("\n=== ESTATÍSTICAS E CONTAGEM DE PORTAS ===")
for porta, qtd in contagem.items():
    print(f"Porta '{porta}': {qtd}")

print(f"Total de portas no circuito: {total_portas}")
print(f"Quantidade total de CNOTs (cx): {contagem.get('cx', 0)}")
print(f"Profundidade do circuito: {circuito_customizado.depth()}")
