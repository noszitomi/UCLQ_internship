from sandbox_assets import *
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from qiskit_aer_noise import (NoiseModel, depolarizing_error, thermal_relaxation_error)



class Grapher:

    def __init__(self, dictionary = None):

        if dictionary == None:

            self._copies = None
            self._initial_state = None 
            self._obs = None
            self._basis_gates = None
            self._nb_shots = None
            self._error_range = None
            self._depol = None
            self._thermal = None
            self._opt_level = None
            self._seed = None
        
        else:

            self._copies = dictionary['copies']
            self._initial_state = dictionary['initial_state']
            self._obs = dictionary['obs']
            self._basis_gates = dictionary['basis_gates']
            self._nb_shots = dictionary['nb_shots']
            self._error_range = dictionary['error_range']
            self._depol = dictionary['depol']
            self._thermal = dictionary['thermal']
            self._opt_level = dictionary['opt_level']
            self._seed = dictionary['seed']

        self._exact_expectation = None
        self._mit_unmit_values = None

    def fromDict(self, dictionary):

        self._copies = dictionary['copies']

        if self._initial_state is None:
            self._initial_state = dictionary['initial_state']
        else:
            print('Initial state is already set, please use addInitialState to overwrite')

        if self._obs == None: 
            self._obs = dictionary['obs']
        else:
            print('Observable is already set, please use addInitialState to overwrite')

        self._basis_gates = dictionary['basis_gates']
        self._nb_shots = dictionary['nb_shots']
        self._error_range = dictionary['error_range']
        self._depol = dictionary['depol']
        self._thermal = dictionary['thermal']
        self._opt_level = dictionary['opt_level']
        self._seed = dictionary['seed']

    def addInitialState(self, initial_state, show = False):
        self._initial_state = initial_state

        if show == True: display(initial_state.draw())

    def addObservable(self, observable, show = False):
        self._obs = observable

        if show == True: display(observable.draw())
    
    def exactExpectation(self, show = False):

        temp_circ_full = circAssembler(copies = self._copies, qubits = self._initial_state.num_qubits, ancilla_qubits = 1,
                                       initial_state = self._initial_state, obs = self._obs, basis_gates = self._basis_gates,
                                       der_op=True, opt_level = self._opt_level, seed = self._seed)
        
        temp_circ_id = circAssembler(copies = self._copies, qubits = self._initial_state.num_qubits, ancilla_qubits = 1,
                                       initial_state = self._initial_state, obs = False, basis_gates = self._basis_gates,
                                       der_op=True, opt_level = self._opt_level, seed = self._seed)
        
        self._exact_expectation = expValue(fullQC = temp_circ_full, idQC = temp_circ_id, nb_shots = None,
                                           noise_model = None, seed = self._seed)
        
        if show == True: print('The exact expectation value is: {}'.format(self._exact_expectation))
    
    def expecForGraph(self):

        qubits = self._initial_state.num_qubits

        mit_unmit_values = np.empty([2, self._error_range.size])

        for i, error in tqdm(enumerate(self._error_range)):

            noise_m = NoiseModel()
            
            if self._depol == True:
                
                dep_1q = depolarizing_error(10**error, 1)
                dep_2q = depolarizing_error(10**error, 2)

                noise_m.add_all_qubit_quantum_error(dep_1q, ['rz', 'id', 'sx', 'x'])
                noise_m.add_all_qubit_quantum_error(dep_2q, ['cx'])

            # DO: think of the scaling for thermal relaxation
            
            if self._thermal == True:

                t1, t2, time, excited_state_population = self._thermal_relaxation.values()

                therm_1q = thermal_relaxation_error(t1 = t1, t2 = t2, time = time, excited_state_population = excited_state_population)
                therm_2q = therm_1q.tensor(therm_1q)

                noise_m.add_all_qubit_quantum_error(therm_1q, ['rz', 'id', 'sx', 'x'])
                noise_m.add_all_qubit_quantum_error(therm_2q, ['cx'])

            mit_exp, unmit_exp = circTester(copies = self._copies, qubits = qubits, ancilla_qubits = 1,
                                                initial_state = self._initial_state, obs = self._obs,
                                                basis_gates = self._basis_gates, nb_shots = self._nb_shots,
                                                noise_model = noise_m, seed = self._seed)
        
            mit_unmit_values[0][i]  = mit_exp
            mit_unmit_values[1][i] = unmit_exp

        self._mit_unmit_values = mit_unmit_values

        return(mit_unmit_values)

    def graphExpecError(self, ylim = None, xlim = None, figsize = (6, 3)):

        if isinstance(self._mit_unmit_values, type(None)): self.expecForGraph()
        if self._exact_expectation == None: self.exactExpectation()

        fig, ax = plt.subplots(figsize = figsize)

        ax.plot(self._error_range, self._mit_unmit_values[0][:], label = 'Mitigated')
        ax.plot(self._error_range, self._mit_unmit_values[1][:], label = 'Unmitigated')

        if ylim != None: ax.set_ylim(ylim) 
        if xlim != None: ax.set_xlim(xlim)

        ax.axhline(self._exact_expectation, color = 'mediumaquamarine', label = 'Known expectation value')
        
        ax.legend()

        ax.set(
        title ='Fig. 1: Copies = {}, Shots = {}, ({} qubit(s)/register)'.format(self._copies, self._nb_shots, self._initial_state.num_qubits),
        xlabel = 'Error probability', 
        ylabel = 'Expectation value'
        )

        plt.show()  


    def graphAbsError(self, figsize = (6, 3)):
        
        if isinstance(self._mit_unmit_values, type(None)): self.expecForGraph()
        if self._exact_expectation == None: self.exactExpectation()

        fig, ax = plt.subplots(figsize = figsize)

        ax.loglog(10**self._error_range, np.abs(self._mit_unmit_values[0][:] - self._exact_expectation), label = 'Mitigated')
        ax.loglog(10**self._error_range, np.abs(self._mit_unmit_values[1][:] - self._exact_expectation), label = 'Unmitigated')
        
        ax.legend()

        ax.set(
        title ='Fig. 1: Copies = {}, Shots = {}, ({} qubit(s)/register)'.format(self._copies, self._nb_shots, self._initial_state.num_qubits),
        xlabel = 'Error probability', 
        ylabel = 'Error in expectation value'
        )

        plt.show()  
