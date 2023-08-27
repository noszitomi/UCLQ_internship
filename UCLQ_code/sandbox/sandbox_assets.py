from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister, execute, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator as AerEstimator

def iniQC(numReg, numQ, numAnc):
    
    qc = QuantumCircuit()

    a = AncillaRegister(numAnc, 'Anc')
    qc.add_register(a)

    for i in range(numReg):
        qr = QuantumRegister(numQ, f'reg_{i}')
        qc.add_register(qr)

    return qc


def iniState(initial_state, qc, basis_gates, seed = 1, opt_level = 0):

    initial_state_gate = initial_state.to_gate()

    temp_circ = QuantumCircuit(initial_state.num_qubits)
    temp_circ.append(initial_state_gate, [*range(0, initial_state.num_qubits)])

    temp_circ_t = transpile(temp_circ, basis_gates = basis_gates, optimization_level = opt_level, seed_transpiler = seed)

    for register in qc.qregs[:0:-1]:
        qc = qc.compose(temp_circ_t, register)
    
    return(qc)


def derangement(qc, basis_gates, seed = 1, opt_level = 0):
    
    ancilla_qubit = qc.ancillas[0]
    registers = qc.qregs[1:]
    numQubits = registers[0].size

    temp_circ = qc.copy_empty_like()

    for i in reversed(range(1, len(registers))):
        for j in reversed(range(numQubits)):
            temp_circ.cswap(ancilla_qubit, registers[i][j], registers[i-1][j])
    
    temp_circ_t = transpile(temp_circ, basis_gates = basis_gates, optimization_level = opt_level, seed_transpiler = seed)

    qc = qc.compose(temp_circ_t)

    return(qc)


def observable(obs, qc, basis_gates, seed = 1, opt_level = 0):

    """Creates and applies the controlled version of a given observable to the
    main circuit.

    Parameters
    ----------
    obs : qiskit QuantumCircuit
        The (non-controlled) observable we want to apply
    qc : qiskit QuantumCircuit
        The main circuit we want to apply the observable to
    basis_gates : list
        The lis of basis gates to witch the observable is transpiled to
    opt_level : int
        The optimization level for the transpilation

    Returns
    -------
    A qiskit QuantumCircuit
    
    """

    c_observable_gate = (obs.to_gate()).control()

    temp_circ = QuantumCircuit(obs.num_qubits+1)
    temp_circ.append(c_observable_gate, [*range(0, obs.num_qubits+1)])

    temp_circ_t = transpile(temp_circ, basis_gates = basis_gates, optimization_level = opt_level, seed_transpiler = seed)

    qc = qc.compose(temp_circ_t, [qc.ancillas[0]] + qc.qregs[1][:])

    return(qc)


def circAssembler(copies, qubits, ancilla_qubits, initial_state, obs, basis_gates, der_op = True, opt_level = 0, seed = 1):

    """ Assembles a qiskit quantum circuit containing:
        - one ancilla register with 'numAnc' number of ancilla qubits
        -'copies' number of quantum registers each with 'qubits' number of qubits
        - [OPTIONAL] a derangement operator
        - [OPTIONAL] a controlled sigma gate (i.e. an observable)
    
    Parameters
    ----------
    numReg : int
        The number of quantum regsiters.
    numQ : int
        The number of qubits / each quantum register
    numAnc : int
        The number of ancilla qubits
    sigma:
        The circuit of the observable 
    basis_gates: 
        Basis gates to witch the observable is transpiled into
    der_op: boolean
        If true it includes a derangement operator.
    
    Returns
    -------
    A qiskit quantum circuit.

    """    

    temp_circ = QuantumCircuit(1)
    temp_circ.h(0)

    h_t = transpile(temp_circ, basis_gates = basis_gates, optimization_level = opt_level, seed_transpiler = seed)

    qc = iniQC(copies, qubits, ancilla_qubits)
    qc = iniState(initial_state = initial_state, qc = qc, basis_gates = basis_gates, seed = seed, opt_level = opt_level)
    qc.barrier()
    qc = qc.compose(h_t, 0)
    if der_op == True:
        qc = derangement(qc = qc, basis_gates = basis_gates, seed = seed, opt_level = opt_level)
        qc.barrier()
    if obs != False: 
        qc = observable(obs = obs, qc = qc, basis_gates = basis_gates, seed = seed, opt_level = opt_level)
    qc = qc.compose(h_t, 0)

    return(qc)


def prob0(qc, nb_shots, noise_model, seed = 1):

    """ Calculates the 'probability' of measuring the ancilla quibit in state 0.
    
    
    Parameters
    ----------
    qc : qiskit quantum circuit
    nb_shots : int
        The number shots for the simulation.
        If None the exact expectation value is calculated.
    noise_model : 
        The noise model to use for executing the circuit
    Returns
    -------
    A float expressing the probability 'prob0' from the paper.

    """

    obs = SparsePauliOp((qc.num_qubits-1)*"I"+"Z")

    Estimator = AerEstimator(
        backend_options= {"noise_model": noise_model},
        run_options={"seed": seed, "shots" : nb_shots},
        skip_transpilation = True,  # problematic, as the density matrix method does not support
                                    # native cswaps.
        approximation = True if nb_shots == None else False 
        )

    job = Estimator.run(qc, obs)
    result = job.result()
    exp_val = result.values[0]

    return(exp_val) 
    

def expValue(fullQC, idQC, nb_shots, noise_model, seed = 1):

    """ Calculates the expectation value of the circuit, assuming NO knowladge of the dominant eigenvector's eigenvalue.
    
    
    Parameters
    ----------
    fullQC : qiskit quantum circuit
        The 'full' quantum circuit, including the derangement operator and the controlled sigma gate.
    idQC : qiskit quantum circuit
        The quantum circuit for estimating the the dominant eginevctors eigenvalue.  (omitting the controlled sigma gate)
    nb_shots : int
        The number shots for the simulation.
        If None the exact expectation value is calculated.
    noise_model : 
        The noise model to use for executing the circuit
    Returns
    -------
    A float, expressing the expectation value of the controlled sigma gate. 

    """

    prob_0 = prob0(qc = fullQC, nb_shots = nb_shots, noise_model = noise_model, seed = seed)
    prob_0_prime = prob0(qc = idQC, nb_shots = nb_shots, noise_model = noise_model, seed = seed)
 
    return(prob_0/prob_0_prime)



#def circ_tester(numReg, numQ, numAnc, sigma, basis_gates, nb_shots, noise_model, decomp, seed = 1):
def circTester(copies, qubits, ancilla_qubits, initial_state, obs, basis_gates, nb_shots, noise_model, opt_level = 0, seed = 1):
    
    """ Tester for evaluating the quantum circuit, it shows:
        - the expectation value for the mitigated and unmitigated case
    
    Parameters
    ----------
    numReg : int
        The number of quantum regsiters.
    numQ : int
        The number of qubits / each quantum register
    numAnc : int
        The number of ancilla qubits
    obs:
        The circuit of the observable
    basis_gates: 
        Basis gates to witch the initial state, derangement operator and the observable is transpiled into
    nb_shots : int
        The number shots for the simulation.
        If None the exact expectation value is calculated.
    noise_model : 
        The noise model to use for executing the circuit 
    Returns
    -------
    The prob0, prob0' and expectation values for the mitigated and unmitigated cases.

    """

    full_mit = circAssembler(copies = copies, qubits = qubits, ancilla_qubits = ancilla_qubits, initial_state = initial_state, obs = obs,
                             basis_gates=basis_gates, der_op=True,opt_level = opt_level, seed = seed)
    id_mit = circAssembler(copies = copies, qubits = qubits, ancilla_qubits = ancilla_qubits, initial_state = initial_state, obs = False,
                           basis_gates=basis_gates, der_op=True, opt_level = opt_level, seed = seed)

    full_unmit = circAssembler(copies = copies, qubits = qubits, ancilla_qubits = ancilla_qubits, initial_state = initial_state,  obs = obs,
                               basis_gates = basis_gates, opt_level = opt_level,  der_op=False, seed = seed)
    id_unmit = circAssembler(copies = copies, qubits = qubits, ancilla_qubits = ancilla_qubits, initial_state = initial_state, obs = False,
                             basis_gates = basis_gates, opt_level=opt_level, der_op=False, seed = seed)

    return(expValue(fullQC = full_mit, idQC = id_mit, nb_shots = nb_shots, noise_model = noise_model, seed = seed), 
           expValue(fullQC = full_unmit, idQC = id_unmit, nb_shots = nb_shots, noise_model = noise_model, seed = seed))

def cnots(qc, regs):

    """ Applies a cx gate between the Ancilla qubit and each qubit of the selected quantum registers. 
    
    
    Parameters
    ----------
    qc : qiskit quantum circuit
        The circuit to which the cx gates are applied
    regs : int
        The register to which the cx gates are applied to. 
    Returns
    -------
    A qiskit quantum circuit.

    """

    ancilla_qubit = qc.ancillas[0]
    registers = qc.qregs[1:]
    numQubits = registers[0].size

    for reg in regs:
        for qubit in reversed(range(numQubits)):
            qc.cnot(ancilla_qubit, registers[reg][qubit])


def GHZ(qc):

    registers = qc.qregs[:]
    numQubits = registers[0].size

    for reg in registers:
        qc.h(reg[0])

        for qubit in range(numQubits-1):
            qc.cnot(reg[qubit], reg[qubit+1])