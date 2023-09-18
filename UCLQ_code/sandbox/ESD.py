from typing import Any
from sandbox_assets import *
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit.quantum_info import Statevector, Operator
from qiskit.providers.fake_provider import FakeProviderForBackendV2
from tqdm import tqdm
from qiskit_aer.noise import (NoiseModel, depolarizing_error, thermal_relaxation_error, phase_damping_error,amplitude_damping_error )

from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Session, Options

import time
from itertools import permutations 
import matplotlib.pyplot as plt
import numpy as np


class esdCircuit:

    def __init__(self,
                copies = None,
                initial_state = None, noisy_ini = True,
                noisy_der = True,
                obs = None, noisy_obs = True,
                noisy_h = True,
                basis_gates = None,
                coupling_map = None,
                seed = None,
                opt_level = None,
                double_anc = False):
        
        self._copies = copies 
        self._initial_state = initial_state 
        self._obs = obs
        self._basis_gates = basis_gates
        self._opt_level = opt_level
        self._coupling_map = coupling_map
        self._seed = seed

        self._double_anc = double_anc

        self._noisy_ini_state = noisy_ini
        self._noisy_der = noisy_der
        self._noisy_obs = noisy_obs
        self._noisy_h = noisy_h

        self._mit_full_qc = None
        self._mit_id_qc = None
        self._unmit_full_qc = None
        self._unmit_id_qc = None

    def initial_state(self, initial_state = None):
        if initial_state:
            self._initial_state = initial_state
            # if (self.allParamsSet() and self.allCircsSet()) == True:
            #     self.updateCircs()

        elif (initial_state == None and self._initial_state == None) :
            print('The circuit does not have an initial state yet, please add one.')

        else:
            return(self._initial_state)
        
    def ini_state_gate_num(self):
        ini_state_T = transpile(self._initial_state, basis_gates = self._basis_gates,
                                optimization_level = self._opt_level, seed_transpiler = self._seed,  coupling_map = self._coupling_map)
        return(sum(ini_state_T.count_ops().values()))
    
    def couplingDiff(self):


        ini_state_coupling = transpile(self._initial_state, basis_gates = self._basis_gates,
                                optimization_level = self._opt_level, seed_transpiler = self._seed,  coupling_map = self._coupling_map)
        
        
        ini_state_simple = transpile(self._initial_state, basis_gates = self._basis_gates,
                                optimization_level = self._opt_level, seed_transpiler = self._seed)
        
        full_coupling = transpile(self.mitFullCirc(), basis_gates = self._basis_gates,
                                optimization_level = self._opt_level, seed_transpiler = self._seed, coupling_map = self._coupling_map)
        
        full_simple = transpile(self.mitFullCirc(), basis_gates = self._basis_gates,
                                optimization_level = self._opt_level, seed_transpiler = self._seed)

        
        table = [
            ["", "no coupling", "with coupling"],
            ["initial state", sum(ini_state_simple.count_ops().values()), sum(ini_state_coupling.count_ops().values())],
            ["full circuit", sum(full_simple.count_ops().values()) , sum(full_coupling.count_ops().values())]
                ]
        for row in table:
            print("{: <20} {: <15} {: <15}".format(*row))


    def copies(self, copies = None):
        if copies:
            self._copies = copies
            # if (self.allParamsSet() and self.allCircsSet()) == True:
            #     self.updateCircs()

        elif (copies == None and self._copies == None) :
            print('The circuit does not have an observable yet, please add one.')

        else:
            return(self._copies)

    def observable(self, observable = None):
        if observable:
            self._obs = observable
            # if (self.allParamsSet() and self.allCircsSet()) == True:
            #     self.updateCircs()

        elif (observable == None and self._obs == None) :
            print('The circuit does not have an observable yet, please add one.')

        else:
            return(self._obs)
    
    def basis_gates(self, basis_gates = None):
       
        if basis_gates:
            self._basis_gates = basis_gates
            # if (self.allParamsSet() and self.allCircsSet()) == True:
            #     self.updateCircs()

        elif (basis_gates == None and self._basis_gates == None) :
            print('The circuit does not have its basis gates specified, please add them.')
        
        else:
            return(self._basis_gates)
        

    def opt_level(self, opt_level = None):
        
        if opt_level:
            self._opt_level = opt_level
            # if (self.allParamsSet() and self.allCircsSet()) == True:
            #     self.updateCircs()

        elif (opt_level == None and self._opt_level == None) :
            print('The optimisation level has not been set yet, please set it.')
        
        else:
            return(self._opt_level)

    def seed(self, seed = None):
        
        if seed:
            self._seed = seed
            # if (self.allParamsSet() and self.allCircsSet()) == True:
            #     self.updateCircs()

        elif (seed == None and self._seed == None) :
            print('The seed has not been set yet, please set it.')
        
        else:
            return(self._seed)
        
    def circAssembler(self, noisy_ini_state = True, der_op = True, noisy_der = True, obs_op = True, noisy_obs = True, noisy_h = True):

        temp_h = QuantumCircuit(1)
        temp_h.h(0)

        h_t = transpile(temp_h, basis_gates = self._basis_gates, optimization_level = self._opt_level,
                        seed_transpiler = self._seed)
        
        # temp_cx = QuantumCircuit(2)
        # temp_cx.cx(0,1)

        # cx_t = transpile(temp_cx, basis_gates = self._basis_gates, optimization_level = self._opt_level,
        #                 seed_transpiler = self._seed)
        
        if not noisy_h: h_t = self.nonNoisy(h_t) 

        full_qc = self.iniQC()

        temp_ini = self.iniState(qc=full_qc)
        
        if not noisy_ini_state: temp_ini = self.nonNoisy(temp_ini)

        full_qc = full_qc.compose(temp_ini)
        
        full_qc.barrier()

        for i in full_qc.ancillas: full_qc = full_qc.compose(h_t, i)
        #full_qc = full_qc.compose(h_t, 0)

        #if self._double_anc: full_qc = full_qc.compose(cx_t, [0,1])

        if der_op:

            temp_der = self.derangement(qc = full_qc)
            
            if not noisy_der: temp_der = self.nonNoisy(temp_der)

            full_qc =full_qc.compose(temp_der)
            
            full_qc.barrier()

        #if self._double_anc: full_qc = full_qc.compose(cx_t, [0,1])

        if obs_op:

            obs_ini = self.observable(qc = full_qc)
            
            if not noisy_obs: obs_ini = self.nonNoisy(obs_ini)
            full_qc =full_qc.compose(obs_ini)

            full_qc.barrier()
        
        for i in full_qc.ancillas: full_qc = full_qc.compose(h_t, i)
        #full_qc = full_qc.compose(h_t, 0)

        if self._coupling_map:
            full_qc = transpile(full_qc, basis_gates = self._basis_gates, optimization_level = self._opt_level,
                        seed_transpiler = self._seed, coupling_map = self._coupling_map)

        return(full_qc)
    
    def mitFullCirc(self): 

        #if not isinstance(self._mit_id_qc, type(None)): return(self._mit_id_qc)

        self._mit_full_qc = self.circAssembler(noisy_ini_state = self._noisy_ini_state, der_op=True, noisy_der = self._noisy_der,
                                           obs_op = True, noisy_obs = self._noisy_obs, noisy_h = self._noisy_h)
        return(self._mit_full_qc) 
       
    def mitIDCirc(self):
        
        #if not isinstance(self._mit_id_qc, type(None)): return(self._mit_id_qc)

        self._mit_id_qc = self.circAssembler(noisy_ini_state=self._noisy_ini_state, der_op=True, noisy_der = self._noisy_der,
                                           obs_op = False, noisy_h = self._noisy_h)
        return(self._mit_id_qc)
    
    def unmitFullCirc(self):

        #if not isinstance(self._mit_id_qc, type(None)): return(self._mit_id_qc)

        self._unmit_full_qc = self.circAssembler(noisy_ini_state=self._noisy_ini_state, der_op=False,
                                            obs_op = True, noisy_obs=self._noisy_obs, noisy_h = self._noisy_h)
        return(self._unmit_full_qc) 
       
    def unmitIDCirc(self):

        #if not isinstance(self._mit_id_qc, type(None)): return(self._mit_id_qc)

        self._unmit_id_qc = self.circAssembler(noisy_ini_state=self._noisy_ini_state, der_op=False,
                                               obs_op = False, noisy_h = self._noisy_h)
        return(self._unmit_id_qc)

    def allParamsSet(self):
        if (self._copies and self._initial_state and self._obs and self._obs and
            self._basis_gates and self._opt_level and self._seed) != None:
            return(True)
        else:
            return(False)
    
    def allCircsSet(self):
        if (self._mit_full_qc and self._mit_id_qc and self._unmit_full_qc and self._unmit_id_qc) != None:
            return(True)
        else:
            return(False)
        
    def updateCircs(self):
        self.mitFullCirc()
        self.mitIDCirc()
        self.unmitFullCirc()
        self.unmitIDCirc()

    def checkNoisySettings(self, noisy_ini_state, noisy_der, noisy_obs, noisy_h):
        if (self._noisy_ini_state == noisy_ini_state and self._noisy_der == noisy_der
            and self._noisy_obs == noisy_obs and  self._noisy_h == noisy_h):
            return(True)
        else:
            return(False)

    def nonNoisy(self, qc):
        for gate in qc.data:
            gate[0].label = 'noisless_{}'.format(gate[0].name)
        return qc
    

    def iniQC(self):
    
        qc = QuantumCircuit()

        if self._double_anc:
            a = AncillaRegister(2, 'Anc')
        else:
             a = AncillaRegister(1, 'Anc')
        qc.add_register(a)

        for i in range(self._copies):
            qr = QuantumRegister(self._initial_state.num_qubits, f'reg_{i}')
            qc.add_register(qr)

        return qc


    def iniState(self, qc):

        initial_state_gate = self._initial_state.to_gate()

        temp_circ = QuantumCircuit(self._initial_state.num_qubits)
        temp_circ.append(initial_state_gate, [*range(0, self._initial_state.num_qubits)])

        temp_circ_t = transpile(temp_circ, basis_gates = self._basis_gates,
                                optimization_level = self._opt_level, seed_transpiler = self._seed)

        for register in qc.qregs[:0:-1]:
            qc = qc.compose(temp_circ_t, register)

        return(qc)
    

    def derangement(self, qc):

        ancilla_qubit =  qc.ancillas[1] if self._double_anc else qc.ancillas[0] 
        
        #ancilla_qubit = qc.ancillas[0]
        registers = qc.qregs[1:]
        numQubits = registers[0].size

        temp_circ = qc.copy_empty_like()

        for i in reversed(range(1, len(registers))):
            for j in reversed(range(numQubits)):
                temp_circ.cswap(ancilla_qubit, registers[i][j], registers[i-1][j])
        
        temp_circ_t = transpile(temp_circ, basis_gates = self._basis_gates,
                                optimization_level = self._opt_level, seed_transpiler = self._seed)

        return(temp_circ_t)

    def observable(self, qc):

        qc = qc.copy_empty_like()

        c_observable_gate = (self._obs.to_gate()).control()

        temp_circ = QuantumCircuit(self._obs.num_qubits+1)
        temp_circ.append(c_observable_gate, [*range(0, self._obs.num_qubits+1)])

        temp_circ_t = transpile(temp_circ, basis_gates = self._basis_gates,
                                optimization_level = self._opt_level, seed_transpiler = self._seed)

        qc = qc.compose(temp_circ_t, [qc.ancillas[0]] + qc.qregs[1][:])

        return(qc)


class esdResults:

    def __init__(self, nb_shots = None, noise_model = None, backend = 'aer', coupling_map = None, seed = None):
        
        self._backend = backend
        self._coupling_map = coupling_map
        self._nb_shots = nb_shots
        self._noise_model = noise_model
        self._seed = seed
        

    def prob0(self, qc):

        obs = SparsePauliOp((qc.num_qubits-qc.num_ancillas)*"I"+qc.num_ancillas*"Z")

        if self._backend == 'aer':
            estimator = AerEstimator(
            backend_options= {"noise_model": self._noise_model,
                              "coupling_map": self._coupling_map},
            run_options={"seed": self._seed, "shots" : self._nb_shots},
            skip_transpilation = True,  
            approximation = True if self._nb_shots == None else False
        )
            
            job = estimator.run(qc, obs)
            
        
        
        # look at initial layouts


        elif self._backend == 'quasm':  # qasm!!

            temp_backend =  service.get_backend('ibmq_qasm_simulator')

            options = Options()
            options.simulator = {'noise_model': self._noise_model,
                                 'seed_simulator': self._seed,
                                 'coupling_map': self._coupling_map}
            options.execution.shots = self._nb_shots
            options.optimization_level = 0
            options.resilience_level = 0

            # Think of approximation again 

            qc_temp = QuantumCircuit(qc.num_qubits)
            qc_temp = qc_temp.compose(qc)

            with Session(service=service, backend=temp_backend):
                estimator = Estimator(options = options)
                job = estimator.run(circuits=qc_temp, observables=obs, skip_transpilation=True)



        else:
            temp_backend = service.get_backend(self._backend)
 
            options = Options()
            options.optimization_level = 0
            options.resilience_level = 0


            qc_temp = QuantumCircuit(qc.num_qubits)
            qc_temp = qc_temp.compose(qc)

            with Session(service=service, backend=temp_backend):
                estimator = Estimator(options = options)
                job = estimator.run(circuits=qc_temp, observables=obs, skip_transpilation=True)


        result = job.result()
        exp_val = result.values[0]

        return(exp_val)
    

    def expVal(self, fullQC, idQC):

        prob_0 = self.prob0(fullQC)
        prob_0_prime = self.prob0(idQC)

        return(prob_0/prob_0_prime)

class esdNoiseModel:

    def __init__(self):

        self._model_source = None

        self._fake_backend_name = None

        self._depol = None
        self._d_amp = None
        self._deph = None
        self._two_qubit_scale = None
        self._google_scale = None

        self._max_error = None

        self._noise_model = None
        
    def specifyFakeDevice(self, fake_backend_name):
        self._fake_backend_name = fake_backend_name
        self._model_source = 'fake_device'
    
    def specifyCustomParams(self, depol = None, d_amp = None, deph = None, two_qubit_scale = 1, google_scale = False ):
        self._depol = depol
        self._d_amp = d_amp
        self._deph = deph
        self._two_qubit_scale = two_qubit_scale
        self._google_scale = google_scale
        self._model_source = 'custom_model'

    def getNoiseModel(self, scale_error):

        if self._model_source == 'fake_device':
            noise_m, max_error = self.noiseFromFakeDevice(scale_error=scale_error)
            return(noise_m, max_error)
        
        elif self._model_source == 'custom_model':
            noise_m, max_error = self.costumNoise(scale_error=scale_error)
            return(noise_m, max_error)

    def noiseFromFakeDevice(self, scale_error):

        # there must be a better way
        for fake_backend in FakeProviderForBackendV2().backends():
            if fake_backend.backend_name == self._fake_backend_name:
                backend = fake_backend

        max_error = 0

        for op_name, inst_prop_dic in backend.target.items():
            for qubits, inst_prop in inst_prop_dic.items():

                if backend.target[op_name][qubits] is None:
                    continue
                
                if backend.target[op_name][qubits].error: 
                    backend.target[op_name][qubits].error *= scale_error 
                    if op_name != 'measure':
                        max_error = max(max_error, backend.target[op_name][qubits].error)
                    
                backend.target[op_name][qubits].duration *= scale_error 

        noise_m = NoiseModel.from_backend(backend, gate_error = True, readout_error = False, thermal_relaxation = True)
        
        self._noise_model = noise_m
        return(self._noise_model, max_error)
    
    def costumNoise(self, scale_error):

        noise_m = NoiseModel()

        if self._depol:
            if not self._google_scale: err1_depol = depolarizing_error(self._depol*scale_error, 1)
            err2_depol = depolarizing_error(self._two_qubit_scale*self._depol*scale_error, 2)

            if not self._google_scale: noise_m.add_all_qubit_quantum_error(err1_depol, ['rz', 'id', 'sx', 'x'],  warnings=False)
            noise_m.add_all_qubit_quantum_error(err2_depol, ['cx'], warnings = False)
        
        if self._d_amp:
            if not self._google_scale: err_1d_amp = amplitude_damping_error(self._d_amp*scale_error)
            err_2d_amp = amplitude_damping_error(self._two_qubit_scale*self._d_amp*scale_error).tensor(
                                                 amplitude_damping_error(self._two_qubit_scale*self._d_amp*scale_error))
            
            if not self._google_scale: noise_m.add_all_qubit_quantum_error(err_1d_amp, ['rz', 'id', 'sx', 'x'],  warnings=False)
            noise_m.add_all_qubit_quantum_error(err_2d_amp, ['cx'], warnings = False)
        
        if self._deph:
            if not self._google_scale: err_1deph = phase_damping_error(self._deph*scale_error)
            err_2dephe = phase_damping_error(self._two_qubit_scale*self._deph*scale_error).tensor(
                                             phase_damping_error(self._two_qubit_scale*self._deph*scale_error))
            
            if not self._google_scale: noise_m.add_all_qubit_quantum_error(err_1deph, ['rz', 'id', 'sx', 'x'],  warnings=False)
            noise_m.add_all_qubit_quantum_error(err_2dephe, ['cx'], warnings = False)
        
        if self._google_scale:
            max_error = 2*sum(filter(None, [self._depol, self._d_amp, self._deph]))

        elif self._two_qubit_scale < 1:
            max_error = scale_error*max(filter(None, [self._depol, self._d_amp, self._deph]))

        else:
            max_error = scale_error*self._two_qubit_scale*max(filter(None, [self._depol, self._d_amp, self._deph]))
        
        self._noise_model = noise_m
        return(self._noise_model, max_error)
    


    
    def NO_ANC_noiseFromFakeDevice(self, scale_error):

        # there must be a better way
        for fake_backend in FakeProviderForBackendV2().backends():
            if fake_backend.backend_name == self._fake_backend_name:
                backend = fake_backend


class esdExperiment:

    def __init__(self,
                 esd_noise_model = esdNoiseModel,
                 esd_circuit = esdCircuit,
                 err_range = None,
                 seed = 1) -> None:
        
        self._err_range = err_range
        self._noise_model = esd_noise_model
        self._circuit = esd_circuit
        self._seed = seed

        self.circuits = []
        self.ini_num_gates = []
        self.noise_models = []
        self.max_errors = []
        self.exact_exps = []

        self._service = None
        self._backend = None
        self._nb_shots = None
        self.mit_unmit_values = None

        self._layer_range = None
        self._initial_state = None

    def getCircs(self):

        return([self._circuit.mitFullCirc(), self._circuit.mitIDCirc(),
                self._circuit.unmitFullCirc(), self._circuit.unmitIDCirc()])

    def standardRegisters(self, qc):

        temp_circ = QuantumCircuit(qc.num_qubits)
        temp_circ = temp_circ.compose(qc)

        return(temp_circ)
    
    def noiselessExpectation(self):

        psi = Statevector.from_instruction(self._circuit._initial_state)
        op = Operator.from_circuit(self._circuit._obs)

        expectation_value = psi.evolve(op).inner(psi).real

        return(expectation_value)

    def errRange(self):

        self.circuits += [self.getCircs()]
        self.ini_num_gates += [self._circuit.ini_state_gate_num()]
        self.exact_exps += [self.noiselessExpectation()]

        noise_models = []
        max_errors = []

        for error in tqdm(self._err_range):

            noise_model, max_error = self._noise_model.getNoiseModel(error)

            noise_models.append(noise_model)
            max_errors.append(max_error)

        self.noise_models += [noise_models]
        self.max_errors += [max_errors]
        

        # return(self.circuits, self.noise_models, self.max_errors)
    

    def layerRange(self, layer_range = None, initial_state = None, rzTheta = np.pi/4):

        if layer_range: self._layer_range = layer_range
        if initial_state: self._initial_state = initial_state

        og_initial_state = self._circuit._initial_state

        for layer in self._layer_range:
             
            if self._initial_state == 'GHZ':
                temp_initial_state = GHZreps(qubits = self._circuit._obs.num_qubits, layers = int(layer))
                temp_initial_state.rz(rzTheta, 0)
                
            elif initial_state == 'spin-ring':
                temp_initial_state = spinRing(qubits = self._circuit._obs.num_qubits, layers = int(layer))

            else:
                temp_initial_state = layers_layer(qc = og_initial_state, layers= int(layer) )

            self._circuit.initial_state(temp_initial_state)
            self.errRange()

        # return(self.circuits, self.noise_models, self.max_errors)
    

    
    def run(self, service, backend, coupling_map, nb_shots):
        
        self._nb_shots = nb_shots
        self._service = service
        self._backend = backend

        mit_unmit_values = [[], # for mitigated
                            []] # for unmitigated
       
        if backend == 'aer':

            for circuits, noise_models in zip(self.circuits, self.noise_models):
                for noise_model in tqdm(noise_models):
                    #print(noise_model)

                    estimator = AerEstimator(
                        backend_options= {"noise_model": noise_model,
                                          "coupling_map": coupling_map},
                        run_options={"seed": self._seed, "shots" : nb_shots},
                        skip_transpilation = True,  
                        approximation = True if nb_shots == None else False
                    )

                    observables = []
                    for circuit in circuits:
                        obs = SparsePauliOp.from_sparse_list([('Z', [circuit.data[-1].qubits[0].index], 1)], num_qubits = circuit.num_qubits)
                        observables.append(obs)
                    
                    #print(observables)

                    job = estimator.run(circuits, observables)

                    mitFull, mitID, unmitFull, unmitID = job.result().values

                    exp_mit = mitFull/mitID
                    exp_unmit = unmitFull/unmitID

                    mit_unmit_values[0].append(exp_mit)
                    mit_unmit_values[1].append(exp_unmit)

                time.sleep(1) # gives IOStrem.flush timed out without it. 
            
            self.mit_unmit_values = mit_unmit_values
            # return(self.mit_unmit_values)
            return()

        if backend == 'qasm':

            temp_circuits = [[self.standardRegisters(circuit) for circuit in sublist] for 
                             sublist in self.circuits]

            temp_backend = service.get_backend('ibmq_qasm_simulator')

            with Session(service = self._service, backend = temp_backend): 

                if self.noise_models:
                    for circuits, noise_models in zip(temp_circuits, self.noise_models):
                        for noise_model in noise_models:
                    
                            options = Options()

                            options.simulator = {'noise_model': noise_model,
                                                'seed_simulator': self._seed,
                                                'coupling_map': coupling_map}
                            
                            options.execution.shots = nb_shots
                            options.optimization_level = 0
                            options.resilience_level = 0
                            options.approximation = True if nb_shots == None else False

                            estimator = Estimator(options = options)

                            observables = []
                            for circuit in circuits:
                                obs = SparsePauliOp.from_sparse_list([('Z', [circuit.data[-1].qubits[0].index], 1)], num_qubits = circuit.num_qubits)
                                observables.append(obs)

                            job = estimator.run(circuits=circuits,
                                                observables=observables,
                                                shots = nb_shots,
                                                skip_transpilation=True)
                            
                            self.expValues(job = job, mit_unmit_values = mit_unmit_values)
                            
                else:
                    for circuits in temp_circuits:
                            options = Options()
                            options.simulator = {'noise_model': None,
                                                'seed_simulator': self._seed,
                                                'coupling_map': coupling_map}
                            
                            options.execution.shots = nb_shots
                            options.optimization_level = 0
                            options.resilience_level = 0
                            options.approximation = True if nb_shots == None else False

                            estimator = Estimator(options = options)

                            observables = []
                            for circuit in circuits:
                                obs = SparsePauliOp.from_sparse_list([('Z', [circuit.data[-1].qubits[0].index], 1)], num_qubits = circuit.num_qubits)
                                observables.append(obs)

                            job = estimator.run(circuits=circuits,
                                                observables=observables,
                                                shots = nb_shots,
                                                #skip_transpilation=True
                                                )
                            

                            self.expValues(job = job, mit_unmit_values = mit_unmit_values)
                            
            self.mit_unmit_values = mit_unmit_values
            return()
            # return(self.mit_unmit_values)
        
        else:
            
            temp_backend = self._service.get_backend(backend)

            with Session(service = self._service, backend = temp_backend): 

                circuits = [self.standardRegisters(circuit)
                            for sublist in self.circuits
                            for circuit in sublist]
                
                options = Options()

                options.execution.shots = nb_shots
                options.optimization_level = 0
                options.resilience_level = 0

                estimator = Estimator(options = options)

                observables = []
                for circuit in circuits:
                        obs = SparsePauliOp.from_sparse_list([('Z', [circuit.data[-1].qubits[0].index], 1)], num_qubits = circuit.num_qubits)
                        observables.append(obs)
                
                job = estimator.run(circuits=circuits,
                                    observables=observables,
                                    skip_transpilation=True)
                            
                self.expValues(job = job, mit_unmit_values = mit_unmit_values)

            self.mit_unmit_values = mit_unmit_values
            return()
            # return(self.mit_unmit_values)
                
    def expValues(self, job, mit_unmit_values):

        mitFull, mitID, unmitFull, unmitID = job.result().values

        exp_mit = mitFull/mitID
        exp_unmit = unmitFull/unmitID

        mit_unmit_values[0].append(exp_mit)
        mit_unmit_values[1].append(exp_unmit)

    
    def errorExpectationGraph(self, ylim = None, xlim = None, figsize = (10, 4), save = False):

        mit_expectations = self.mit_unmit_values[0][:]
        unmit_expectations = self.mit_unmit_values[1][:]

        if len(self.circuits) > 1:

            chunk_len = len(self.noise_models[0])

            mit_expectations = [mit_expectations[i:i + chunk_len] 
                                    for i in range(0, len(mit_expectations), chunk_len)]
            
            unmit_expectations = [unmit_expectations[i:i + chunk_len] 
                                    for i in range(0, len(unmit_expectations), chunk_len)]
        
        else:
            mit_expectations = [self.mit_unmit_values[0][:]]
            unmit_expectations = [self.mit_unmit_values[1][:]]
            
        colors = np.flip(plt.cm.viridis(np.linspace(0,0.7,len(self.circuits)+2)), 0)

        fig, ax = plt.subplots(figsize = figsize)

        for i, (mit, unmit) in enumerate(zip(mit_expectations, unmit_expectations)):

            
            ax.plot(self._err_range, mit, color = colors[i], label = '{} gates - mit'.format(self.ini_num_gates[i]))
            ax.plot(self._err_range, unmit, color = colors[i], linestyle = 'dotted', label = '{} gates - unmit'.format(self.ini_num_gates[i]) )

        ax.legend()
        ax.set_title('Shots: {}, backend: {} '.format(self._nb_shots, self._backend))

        ax.set_xlabel('Probability of error')
        ax.set_ylabel('Expectation value')

        if save:
            plt.savefig(save, format="pdf", bbox_inches="tight") 
        

    def errorAbsErrorGraph(self, ylim = None, xlim = None, figsize = (10, 4), save = False):

        mit_expectations = self.mit_unmit_values[0][:]
        unmit_expectations = self.mit_unmit_values[1][:]
        max_errors = self.max_errors

        if len(self.circuits) > 1:

            chunk_len = len(self.noise_models[0])

            mit_expectations = [mit_expectations[i:i + chunk_len] 
                                    for i in range(0, len(mit_expectations), chunk_len)]
            
            unmit_expectations = [unmit_expectations[i:i + chunk_len] 
                                    for i in range(0, len(unmit_expectations), chunk_len)]
            
        else:
            mit_expectations = [self.mit_unmit_values[0][:]]
            unmit_expectations = [self.mit_unmit_values[1][:]]
            
        colors = np.flip(plt.cm.viridis(np.linspace(0,0.7,len(self.circuits)+2)), 0)

        fig, ax = plt.subplots(figsize = figsize)

        for i, (mit, unmit) in enumerate(zip(mit_expectations, unmit_expectations)):

            ansatz_gate_num = self.ini_num_gates[i]
            mit = np.array(mit)
            unmit = np.array(unmit)
            exact = self.exact_exps[i]

            ax.loglog(ansatz_gate_num*np.array(max_errors[i]), np.abs(mit-exact), color = colors[i], label = '{} gates - mit'.format(ansatz_gate_num))
            ax.loglog(ansatz_gate_num*np.array(max_errors[i]), np.abs(unmit-exact), color = colors[i], linestyle = 'dotted', label = '{} gates - unmit'.format(ansatz_gate_num) )

        ax.legend()
        ax.set_title('Shots: {}, backend: {} '.format(self._nb_shots, self._backend))

        ax.set_xlabel('Ansatz circuit error rate')
        ax.set_ylabel('Absolute error in expectation value')

        if save:
            plt.savefig(save, format="pdf", bbox_inches="tight") 






    

class esdTests:

    def __init__(self,
                  esd_circuit = None,
                  esd_noise_model = None,
                  backend = 'aer',
                  noisy_anc = True,
                  nb_shots = None,
                  err_range = None,
                  layer_range = None,
                  qubits_range = None,
                  copies_range = None,
                  seed = 1):

        self._circuit = esd_circuit
        self._noise_model = esd_noise_model
        self._backend = backend 
        self._nb_shots = nb_shots
        self._seed = seed
        self._err_range = err_range
        self._layer_range = layer_range
        self._qubits_range = qubits_range
        self._copies_range = copies_range
        self._noisy_anc = noisy_anc

        self._noiseless_expectation = None

        self._err_range_mit_unmit_values = None

        self._depth_mit_unmit_list = None
        self._depth_exact_expec_list = None
        self._ansatz_gate_nums = None

        self._width_mit_unmit_list = None
        self._width_exact_expec_list = None

        self._copies_mit_unmit_list = None
        self._copies_exact_expec_list = None

    def noiselessExpectation(self, save_result = False):

        psi = Statevector.from_instruction(self._circuit._initial_state)
        op = Operator.from_circuit(self._circuit._obs)

        expectation_value = psi.evolve(op).inner(psi).real

        if save_result: self._noiseless_expectation = expectation_value

        return(expectation_value)


        # temp_result = esdResults(nb_shots = self._nb_shots, noise_model = None, seed = self._seed)

        # if save_result == True: self._noiseless_expectation = temp_result.expVal(fullQC = self._circuit.unmitFullCirc(), idQC= self._circuit.unmitIDCirc())

        # return(temp_result.expVal(fullQC = self._circuit.unmitFullCirc(), idQC= self._circuit.unmitIDCirc()))
    

    def errRange(self, save_result = False):

        coupling_map = self._circuit._coupling_map

        mit_unmit_values = np.empty([3, self._err_range.size])

        for i, error in tqdm(enumerate(self._err_range)):
                
            noise_m, max_error = self._noise_model.getNoiseModel(error)

            temp_result = esdResults(nb_shots = self._nb_shots, noise_model = noise_m,
                                     backend = self._backend, seed = self._seed,
                                     coupling_map= coupling_map)

            mit_unmit_values[0][i] = temp_result.expVal(fullQC = self._circuit.mitFullCirc(), idQC= self._circuit.mitIDCirc())
            mit_unmit_values[1][i] = temp_result.expVal(fullQC = self._circuit.unmitFullCirc(), idQC= self._circuit.unmitIDCirc())
            mit_unmit_values[2][i] = max_error

        if save_result: self._err_range_mit_unmit_values = mit_unmit_values

        return(mit_unmit_values)


    def inilayerRange(self, initial_state, rzTheta = np.pi/4):

        mit_unmit_list = []
        exact_expec_list = []
        ansatz_gate_nums = []
        #intial_state_depths = []

        for layer in tqdm(self._layer_range):
            if initial_state == 'GHZ':
                temp_initial_state = GHZreps(qubits = self._circuit._obs.num_qubits, layers = int(layer))
                temp_initial_state.rz(rzTheta, 0)
                
            elif initial_state == 'spin-ring':
                temp_initial_state = spinRing(qubits = self._circuit._obs.num_qubits, layers = int(layer))

            self._circuit.initial_state(temp_initial_state)
            ansatz_gate_nums.append(self._circuit.ini_state_gate_num())
            #intial_state_depths.append(temp_initial_state.depth())

            mit_unmit_list.append(self.errRange())
            exact_expec_list.append(self.noiselessExpectation())

            self._depth_mit_unmit_list = mit_unmit_list
            self._depth_exact_expec_list = exact_expec_list
            self._ansatz_gate_nums = ansatz_gate_nums
            #self._intial_state_depths = intial_state_depths

        return(self._depth_mit_unmit_list, self._depth_exact_expec_list, self._ansatz_gate_nums)
    

    def qubitRange(self, initial_state, err_range, rzTheta = np.pi/4):

        #------------------------
        #
        # Think this one through!
        #
        #------------------------

        # mit_unmits_list = []
        # exact_expec_list = []

        # if initial_state == 'GHZ':
        #     for qubits in self._qubit_range:
        #         temp_initial_state = GHZreps(qubits = qubits, layers = self._circuit)
        #         temp_initial_state.rz(rzTheta, 0)

        # for qubits in qubit_range:

        #     temp_obs = QuantumCircuit(qubits)
        #     temp_obs.compose(self._circuit.observable())
        #     temp_obs.id(range(self._circuit.observable().num_qubits, qubits))
        #     self._circuit.observable(temp_obs)

        #     mit_unmits_list.append(self.errRange(err_range = err_range))
        #     exact_expec_list.append(self.noiselessExpectation())


        #     self._width_mit_unmit_list = mit_unmits_list
        #     self._width_exact_expec_list = exact_expec_list

        # return(self._width_mit_unmit_list, self._width_exact_expec_list)
        pass
    

    def copiesRange(self):

        mit_unmits_list = []
        exact_expec_list = []
        ansatz_gate_nums = []

        for copies in self._copies_range:

            self._circuit.copies(copies)

            mit_unmits_list.append(self.errRange())
            exact_expec_list.append(self.noiselessExpectation())
            ansatz_gate_nums.append(self._circuit.ini_state_gate_num())


        self._copies_mit_unmit_list = mit_unmits_list
        self._copies_exact_expec_list = exact_expec_list
        self._ansatz_gate_nums = ansatz_gate_nums

        return(self._copies_mit_unmit_list, self._copies_exact_expec_list, self._ansatz_gate_nums)


class esdGraphs:

    def __init__(self, esd_Test):

        self._esd_Test = esd_Test

    def errorExpectation(self, ylim = None, xlim = None, figsize = (10, 4)):

        if isinstance(self._esd_Test._err_range_mit_unmit_values,  type(None)): self._esd_Test.errRange(save_result = True)
        if isinstance(self._esd_Test._noiseless_expectation, type(None)): self._esd_Test.noiselessExpectation(save_result = True)

        mitigated_expectations = self._esd_Test._err_range_mit_unmit_values[0][:]
        unmitigated_expectations = self._esd_Test._err_range_mit_unmit_values[1][:]
        errors = self._esd_Test._err_range
        #max_erros = self._esd_Test._err_range_mit_unmit_values[2][:]

        ansatz_gate_num = self._esd_Test._circuit.ini_state_gate_num()

        fig, ax = plt.subplots(figsize = figsize)

        ax.plot(errors, mitigated_expectations, label = 'Mitigated')
        ax.plot(errors, unmitigated_expectations, label = 'Unmitigated')

        if ylim: ax.set_ylim(ylim) 
        if xlim: ax.set_xlim(xlim)

        ax.axhline(self._esd_Test._noiseless_expectation, color = 'mediumaquamarine', label = 'Known expectation value')

        ax.legend()
        
        ax.set_title('Copies: {}, Shots: {}, backend: {} '.format(
            self._esd_Test._circuit.copies(), self._esd_Test._nb_shots,
            self._esd_Test._noise_model._fake_backend_name if self._esd_Test._noise_model._fake_backend_name else 'AerEstimator'))
        
        ax.set_xlabel('Ansatz circuit error rate (gates: {})'.format(ansatz_gate_num))
        ax.set_ylabel('Expectation value')

        
    
    def errorAbsError(self, ylim = None, xlim = None, figsize = (10, 4)):

        if isinstance(self._esd_Test._err_range_mit_unmit_values,  type(None)): self._esd_Test.errRange(save_result = True)
        if isinstance(self._esd_Test._noiseless_expectation, type(None)): self._esd_Test.noiselessExpectation(save_result = True)

        mitigated_expectations = self._esd_Test._err_range_mit_unmit_values[0][:]
        unmitigated_expectations = self._esd_Test._err_range_mit_unmit_values[1][:]
        #errors = self._esd_Test._err_range
        max_erros = self._esd_Test._err_range_mit_unmit_values[2][:]

        ansatz_gate_num = self._esd_Test._circuit.ini_state_gate_num()

        fig, ax = plt.subplots(figsize = figsize)

        ax.loglog(ansatz_gate_num*max_erros,  np.abs(mitigated_expectations - self._esd_Test._noiseless_expectation), label = 'Mitigated')
        ax.loglog(ansatz_gate_num*max_erros,  np.abs(unmitigated_expectations - self._esd_Test._noiseless_expectation), label = 'Unmitigated')

        if ylim: ax.set_ylim(ylim) 
        if xlim: ax.set_xlim(xlim)

        ax.legend()
        
        ax.set_title('Copies: {}, Shots: {}, backend: {} '.format(
            self._esd_Test._circuit.copies(), self._esd_Test._nb_shots,
            self._esd_Test._noise_model._fake_backend_name if self._esd_Test._noise_model._fake_backend_name else 'AerEstimator'))
        ax.set_xlabel('Ansatz circuit error rate (gates: {})'.format(ansatz_gate_num))

        ax.set_ylabel('Absolute error in expectation value')

        ax.legend(fontsize = 'x-small')

    
    # def errRangeGraph(self, ylim = None, xlim = None, figsize = (12, 8)):
        
    #     if isinstance(self._esd_Test._err_range_mit_unmit_values,  type(None)): self._esd_Test.errRange(save_result = True)
    #     if isinstance(self._esd_Test._noiseless_expectation, type(None)): self._esd_Test.noiselessExpectation(save_result = True)

    #     mitigated_expectations = self._esd_Test._err_range_mit_unmit_values[0][:]
    #     unmitigated_expectations = self._esd_Test._err_range_mit_unmit_values[1][:]
    #     errors = self._esd_Test._err_range
    #     max_erros = self._esd_Test._err_range_mit_unmit_values[2][:]

    #     ansatz_gate_num = self._esd_Test._circuit.ini_state_gate_num()

    #     fig, ((ax1, ax2)) = plt.subplots(2,1, figsize = figsize)

    #     ax1.plot(errors, mitigated_expectations, label = 'Mitigated')
    #     ax1.plot(errors, unmitigated_expectations, label = 'Unmitigated')

    #     if ylim != None: ax1.set_ylim(ylim) 
    #     if xlim != None: ax1.set_xlim(xlim)

    #     ax1.axhline(self._esd_Test._noiseless_expectation, color = 'mediumaquamarine', label = 'Known expectation value')

    #     ax1.legend()
        
    #     if self._esd_Test._noise_model._fake_backend_name != None: 
    #         ax1.set_title('Copies: {}, Shots: {}, backend: {} '.format(self._esd_Test._circuit.copies(), self._esd_Test._nb_shots, self._esd_Test._noise_model._fake_backend_name))
    #         ax1.set_xlabel('Ansatz circuit error rate (gates: {})'.format(ansatz_gate_num))

    #     else:
    #         ax1.set_title('Copies = {}, Shots = {}'.format(self._esd_Test._circuit.copies(), self._esd_Test._nb_shots))
    #         ax1.set_xlabel('Ansatz circuit error rate (gates: {})'.format(ansatz_gate_num))

    #     ax1.set_ylabel('Expectation value')

    #     ax2.loglog(ansatz_gate_num*max_erros,  np.abs(mitigated_expectations - self._esd_Test._noiseless_expectation), label = 'Mitigated')
    #     ax2.loglog(ansatz_gate_num*max_erros,  np.abs(unmitigated_expectations - self._esd_Test._noiseless_expectation), label = 'Unitigated')

    #     if ylim != None: ax2.set_ylim(ylim) 
    #     if xlim != None: ax2.set_xlim(xlim)

    #     ax2.legend()
        
    #     if self._esd_Test._noise_model._fake_backend_name != None: 
    #         ax2.set_title('Copies: {}, Shots: {}, backend: {} '.format(self._esd_Test._circuit.copies(), self._esd_Test._nb_shots, self._esd_Test._noise_model._fake_backend_name))
    #         ax2.set_xlabel('Ansatz circuit error rate (gates: {})'.format(ansatz_gate_num))

    #     else:
    #         ax2.set_title('Copies = {}, Shots = {}'.format(self._esd_Test._circuit.copies(), self._esd_Test._nb_shots))
    #         ax2.set_xlabel('Ansatz circuit error rate (gates: {})'.format(ansatz_gate_num))

    #     ax2.set_ylabel('Absolute error in expectation value')

    #     plt.tight_layout()

    
    def layerErrorExpectation(self, initial_state, ylim = None, xlim = None, figsize = (10, 4)):


        if isinstance(self._esd_Test._depth_mit_unmit_list,  type(None)): self._esd_Test.inilayerRange(initial_state = initial_state)

        depth_mit_unmit_list = self._esd_Test._depth_mit_unmit_list
        #depth_exact_expec_list = self._esd_Test._depth_exact_expec_list
        ansatz_gate_nums = self._esd_Test._ansatz_gate_nums
        errors = self._esd_Test._err_range

        colors = np.flip(plt.cm.viridis(np.linspace(0,1,len(depth_mit_unmit_list)+2)), 0)

        fig, ax = plt.subplots(figsize = figsize)

        for i in range(len(depth_mit_unmit_list)):

            mit_unmit = depth_mit_unmit_list[i]
            #exact_expec = depth_exact_expec_list[i]
            gate_num = ansatz_gate_nums[i]

            ax.plot(errors, mit_unmit[0][:], color = colors[i], label = '{} gates - mit'.format(gate_num))
            ax.plot(errors, mit_unmit[1][:], color = colors[i], linestyle = 'dotted', label = '{} gates - unmit'.format(gate_num))

            if ylim: ax.set_ylim(ylim) 
            if xlim: ax.set_xlim(xlim)


        ax.legend(fontsize = 'x-small')
        ax.set_title('Initial state: {} - expectation value'.format(initial_state))
        ax.set_ylabel('Expectation value')
        if self._esd_Test._noise_model._fake_backend_name:
            ax.set_xlabel('Scale of {}\'s error probability').format(self._esd_Test._noise_model._fake_backend_name)
        else:
            ax.set_xlabel('Error probability')


    def layerErrorAbsError(self, initial_state, ylim = None, xlim = None, figsize = (10, 4)):


        if isinstance(self._esd_Test._depth_mit_unmit_list,  type(None)): self._esd_Test.inilayerRange(initial_state = initial_state)

        depth_mit_unmit_list = self._esd_Test._depth_mit_unmit_list
        depth_exact_expec_list = self._esd_Test._depth_exact_expec_list
        ansatz_gate_nums = self._esd_Test._ansatz_gate_nums
        #errors = self._esd_Test._err_range

        colors = np.flip(plt.cm.viridis(np.linspace(0,1,len(depth_mit_unmit_list)+2)), 0)

        fig, ax = plt.subplots(figsize = figsize)



        for i in range(len(depth_mit_unmit_list)):

            mit_unmit = depth_mit_unmit_list[i]
            exact_expec = depth_exact_expec_list[i]
            gate_num = ansatz_gate_nums[i]

            ax.loglog(gate_num*mit_unmit[2][:],  np.abs(mit_unmit[0][:] - exact_expec), color = colors[i], label = '{} gates - mit'.format(gate_num))
            ax.loglog(gate_num*mit_unmit[2][:],  np.abs(mit_unmit[1][:] - exact_expec),  color = colors[i], linestyle = 'dotted', label = '{} gates - unmit'.format(gate_num))


            if ylim: ax.set_ylim(ylim) 
            if xlim: ax.set_xlim(xlim)


        ax.legend(fontsize = 'x-small')
        ax.set_title('Initial state: {} - absolute error in expectation value'.format(initial_state))
        ax.set_ylabel('Absolute error in expectation value')
        if self._esd_Test._noise_model._fake_backend_name:
            ax.set_xlabel('Ansatz circuit error rate (gates: {}, noise model: {})'.format(gate_num,  self._esd_Test._noise_model._fake_backend_name))
        else:
            ax.set_xlabel('Ansatz circuit error rate (gates: {})'.format(gate_num))

    def copiesErrorExpectation(self, ylim = None, xlim = None, figsize = (10, 4)):

        if isinstance(self._esd_Test._copies_mit_unmit_list,  type(None)): self._esd_Test.copiesRange()

        copies_mit_unmit_list = self._esd_Test._copies_mit_unmit_list
        #copies_exact_expec_list = self._esd_Test._copies_exact_expec_list
        #ansatz_gate_nums = self._esd_Test._ansatz_gate_nums
        errors = self._esd_Test._err_range

        colors = np.flip(plt.cm.viridis(np.linspace(0,1,len(copies_mit_unmit_list)+2)), 0)

        fig, ax = plt.subplots(figsize = figsize)



        for i in range(len(copies_mit_unmit_list)):

            mit_unmit = copies_mit_unmit_list[i]
            #gate_num = ansatz_gate_nums[i]

            ax.plot(errors, mit_unmit[0][:], color = colors[i], label = '{} copies - mit'.format(i))
            ax.plot(errors, mit_unmit[1][:], color = colors[i], linestyle = 'dotted', label = '{} copies - unmit'.format(i))

            if ylim: ax.set_ylim(ylim) 
            if xlim: ax.set_xlim(xlim)


        ax.legend(fontsize = 'x-small')
        ax.set_title('copies')
        ax.set_ylabel('Expectation value')
        if self._esd_Test._noise_model._fake_backend_name:
            ax.set_xlabel('Scale of {}\'s error probability').format(self._esd_Test._noise_model._fake_backend_name)
        else:
            ax.set_xlabel('Error probability')

    def copiesErrorAbsError(self, ylim = None, xlim = None, figsize = (10, 4)):

        if isinstance(self._esd_Test._copies_mit_unmit_list,  type(None)): self._esd_Test.copiesRange()

        copies_mit_unmit_list = self._esd_Test._copies_mit_unmit_list
        copies_exact_expec_list = self._esd_Test._copies_exact_expec_list
        ansatz_gate_nums = self._esd_Test._ansatz_gate_nums
        #errors = self._esd_Test._err_range

        colors = np.flip(plt.cm.viridis(np.linspace(0,1,len(copies_mit_unmit_list)+2)), 0)

        fig, ax = plt.subplots(figsize = figsize)



        for i in range(len(copies_mit_unmit_list)):

            mit_unmit = copies_mit_unmit_list[i]
            exact_expec = copies_exact_expec_list[i]
            gate_num = ansatz_gate_nums[i]

            ax.plot(gate_num*mit_unmit[2][:],  np.abs(mit_unmit[0][:] - exact_expec), color = colors[i], label = '{} copies - mit'.format(i))
            ax.plot(gate_num*mit_unmit[2][:],  np.abs(mit_unmit[1][:] - exact_expec), color = colors[i], linestyle = 'dotted', label = '{} copies - unmit'.format(i))

            if ylim: ax.set_ylim(ylim) 
            if xlim: ax.set_xlim(xlim)


        ax.legend(fontsize = 'x-small')
        ax.set_title('copies')
        ax.set_ylabel('Absolute error in expectation value')
        if self._esd_Test._noise_model._fake_backend_name:
            ax.set_xlabel('Ansatz circuit error rate (gates: {}, noise model: {})'.format(gate_num,  self._esd_Test._noise_model._fake_backend_name))
        else:
            ax.set_xlabel('Ansatz circuit error rate (gates: {})'.format(gate_num))

