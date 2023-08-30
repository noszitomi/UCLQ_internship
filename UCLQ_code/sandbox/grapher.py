from sandbox_assets import *
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from qiskit_aer.noise import (NoiseModel, depolarizing_error, thermal_relaxation_error)



class Grapher:

    def __init__(self,  copies, initial_state, obs, basis_gates, nb_shots, error_range, depol, thermal, opt_level, seed):

        self._copies = copies 
        self._initial_state = initial_state 
        self._obs = obs
        self._basis_gates = basis_gates
        self._nb_shots = nb_shots
        self._error_range = error_range
        self._depol = depol
        self._thermal = thermal
        self._opt_level = opt_level
        self._seed = seed


        self._exact_expectation = None
        self._mit_unmit_values = None
    
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
    
    def expecsForGraph(self):

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

                p_amp = 10**error
                p_damp = 10**error
                t_q1 = 3.271e-08
                t_q2 = 4.196e-07

                t1_q1 = t_q1/(np.log(1/(1-p_amp)))
                t2_q1 = (2*t1_q1*t_q1)/(t1_q1*np.log(1/(1-p_damp))+t_q1)

                t1_q2 = t_q2/(np.log(1/(1-p_amp)))
                t2_q2 = (2*t1_q2*t_q2)/(t1_q2*np.log(1/(1-p_damp))+t_q2)
                

                therm_1q = thermal_relaxation_error(t1 = t1_q1, t2 = t2_q1, time = t_q1)
                therm_2q = thermal_relaxation_error(t1 = t1_q2, t2 = t2_q2, time = t_q1).tensor(
                           thermal_relaxation_error(t1 = t1_q2, t2 = t2_q2, time = t_q2))

                noise_m.add_all_qubit_quantum_error(therm_1q, ['id', 'sx', 'x']) # for fake_backends we had t(rz) = 0
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

        if isinstance(self._mit_unmit_values, type(None)): self.expecsForGraph()
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
        
        if isinstance(self._mit_unmit_values, type(None)): self.expecsForGraph()
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
