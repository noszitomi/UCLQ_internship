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



def derangement(qc):
    
    ancilla_qubit = qc.ancillas[0]
    registers = qc.qregs[1:]
    numQubits = registers[0].size

    for i in reversed(range(1, len(registers))):
        for j in reversed(range(numQubits)):
            qc.cswap(ancilla_qubit, registers[i][j], registers[i-1][j])


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
        skip_transpilation = False,  # problematic, as the density matrix method does not support
                                    # native cswaps.
        approximation = True if nb_shots == None else False 
        )

    job = Estimator.run(qc, obs)
    result = job.result()
    exp_val = result.values[0]

    return((1/2)+(1/2)*exp_val.real) 
    

def exp_value(fullQC, idQC, nb_shots, noise_model, seed = 1):

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


def observable(sigma, qc, basis_gates, seed=1, opt_level = 3):

    """Creates and applies the controlled version of a given observable to the
    main circuit.

    Parameters
    ----------
    sigma : qiskit QuantumCircuit
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

    numQ = len(qc.qregs[1][:]) + 1

    sigma = sigma.to_gate()
    c_sigma = sigma.control()

    temp_circ = QuantumCircuit(numQ)
    temp_circ.append(c_sigma, [*range(0, numQ)])

    temp_circ_t = transpile(temp_circ, basis_gates = basis_gates, optimization_level = opt_level, seed_transpiler = seed)

    qc = qc.compose(temp_circ_t, [*range(0, numQ)])

    return(qc)


def circ_assembler(numReg, numQ, numAnc, sigma, basis_gates, seed = 1, der_op = True):

    """ Assembles a qiskit quantum circuit containing:
        - one ancilla register with 'numAnc' number of ancilla qubits
        -'numReg' number of quanttum registers each with numQ qubits
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

    qc = iniQC(numReg, numQ, numAnc)
    GHZ(qc)
    qc.barrier()
    qc.h(0)
    if der_op == True:
        derangement(qc)
        qc.barrier()
    if sigma != False: 
        qc = observable(sigma = sigma, qc = qc, basis_gates = basis_gates, seed = seed)
    qc.h(0)

    return(qc)


def circ_tester(numReg, numQ, numAnc, sigma, basis_gates, nb_shots, noise_model, seed = 1, decomp = [None]):

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
    sigma:
        The circuit of the observable
    basis_gates: 
        Basis gates to witch the observable is transpiled into
    nb_shots : int
        The number shots for the simulation.
        If None the exact expectation value is calculated.
    noise_model : 
        The noise model to use for executing the circuit 
    decomp :
        A list of gates to 'shalow' decompose.
    Returns
    -------
    The prob0, prob0' and expectation values for the mitigated and unmitigated cases.

    """

    full_mit = circ_assembler(numReg=numReg, numQ = numQ, numAnc = numAnc, sigma = sigma, basis_gates=basis_gates, seed = seed, der_op=True).decompose(decomp)
    id_mit = circ_assembler(numReg = numReg, numQ = numQ, numAnc = numAnc, sigma = False, basis_gates=basis_gates, seed = seed, der_op=True).decompose(decomp)

    full_unmit = circ_assembler(numReg = numReg, numQ = numQ, numAnc = numAnc, sigma = sigma, basis_gates = basis_gates, seed = seed,  der_op=False).decompose(decomp)
    id_unmit = circ_assembler(numReg = numReg, numQ = numQ, numAnc = numAnc, sigma = False, basis_gates = basis_gates, seed = seed, der_op=False).decompose(decomp)

    return(exp_value(fullQC = full_mit, idQC = id_mit, nb_shots = nb_shots, noise_model = noise_model, seed = seed), 
           exp_value(fullQC = full_unmit, idQC = id_unmit, nb_shots = nb_shots, noise_model = noise_model, seed = seed))

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

    registers = qc.qregs[:0:-1]
    numQubits = registers[0].size

    for reg in registers:
        qc.h(reg[0])

        for qubit in range(numQubits-1):
            qc.cnot(reg[qubit], reg[qubit+1])
