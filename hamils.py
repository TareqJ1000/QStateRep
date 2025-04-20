# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 17:21:09 2025

@author: tjaou104

hamils.py

This code contains some useful functions to calculate the TFIM Hamiltonian for quantum systems of size N
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from functools import reduce

# Pauli operators

sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# Initial parameters 
h=1 # Transverse field stength
N=2 # Size of quantum system 

def compute_pauli_i(N, i, pauli_type='z'): # Note that i here works in base zero indexing 
    '''
    N - size of many body system
    i - which qubit are we acting on? 
    pauli_type - which pauli operator are we simulating?
    '''
    
    
    if (pauli_type=='z'):
        pauli_op = sigma_z
    elif (pauli_type=='x'):
        pauli_op = sigma_x
    else: 
        pauli_op = np.eye(2) # Or sigma 0
        
    
    prod_ops = [np.eye(2)]*N
    prod_ops[i] = pauli_op
    
    result = prod_ops[0]
    for op in prod_ops[1:]: # So we are starting from the ii = 2 position, not ii==0
        #print(pauli_i)
        result = np.kron(result, op)
        #input()
        
    return result


def init_hamil(N): # This instantiates the 2^N by 2^N Hamiltonian we are going to generate
    '''
    N - number of qubits in the system 
    '''
    # Construct the 2^N by 2^N dimensional Hamiltonian. The initial Hamiltonian is a tensor product of N zero matrices.  

    hamil = np.zeros((2,2), dtype=np.complex128)
    
    for ii in range(N-1): # Since we have already initialized the first element in the tensor product. 
        temp = np.zeros((2,2), dtype=np.complex128)
        hamil = np.kron(hamil, temp)
    
    return hamil


        
def compute_hamiltonian(N, h):
    '''
    N -- Hamiltonian 
    h -- transverse field strengh 
    '''
    # Construct the 2^N by 2^N dimensional Hamiltonian. The initial Hamiltonian is a tensor product of N zero matrices.  

    hamil = init_hamil(N)
    
    # Now we compute the sum using the formula 
    
    #print(np.shape(compute_pauli_i(N, 0, pauli_type='z')))
    #print(np.shape(compute_pauli_i(N, 1, pauli_type='z')))
    #input()
    
    for ii in range(N):
        if (ii+1==N):
            hamil += -compute_pauli_i(N, ii, pauli_type='z')@compute_pauli_i(N, 0, pauli_type='z') - h*compute_pauli_i(N, ii, pauli_type='x')
        else:
            hamil += -compute_pauli_i(N, ii, pauli_type='z')@compute_pauli_i(N, ii+1, pauli_type='z') - h*compute_pauli_i(N, ii, pauli_type='x')
        
    return hamil

#compute_pauli_i(2, 1, pauli_type='z')
            
#hamil = compute_hamiltonian(2,1)

#print(hamil)
#print(hamil)

def create_basis_array(N):
    '''
    The idea is that we create the 2^N combination of possible basis state configurations. A 0 means spin down, 1 means spin up
    N - Size of quantum system 
    '''
    
    #N = 3  # Change this to whatever length you need
    combinations = list(product([0, 1], repeat=N))

    for ii in range(len(combinations)):
        combinations[ii] = np.array(combinations[ii])
    
    return combinations

 
def gen_basis_state(N): 
    
    '''
    N - number of qubits in our system 
    '''
    base_kets = []

    # Initialize the complete basis state, which occupies the 2^N dimensional Hilbert space. 
    
    for ii in range(N):
        base_kets.append(np.zeros(2, dtype=np.complex128))
        
    # Now, we modify copies of the base ket by consulting the basis_array
    
    basis_array = create_basis_array(N)

    full_basis_array = []
    
    for combo in basis_array: 
        # Create a new copy of the base_kets 
        
        temp = np.copy(base_kets)
        #print(combo)
        #input()
    
        for ii in range(N):
            if (combo[ii]==0):
                temp[ii] = np.array([1,0], dtype=np.complex128)
            elif (combo[ii]==1):
                temp[ii] = np.array([0,1], dtype=np.complex128)
                
        # Finally, compute the kronecker product iteratively across all elements in the array
        
        kron_result =  reduce(np.kron, temp)
        
        #print(kron_result)
        #input()
        full_basis_array.append(kron_result)
        
    
    return full_basis_array



def local_hamil(sample_config, basis_states, hamil, psi_basis, psi_config):
    
    '''
    Computes the local energy as a function on the m^{th} basis state.  
    
    basis_states -- set of all basis states 
    hamil -- Hamiltonian  
    psi_basis --  set of probability amplitudes 
    '''
    
    bra_op = lambda ket: np.conjugate(np.transpose(ket))
    local_hamil = 0
    
    for ii in range(len(basis_states)):
        local_hamil += bra_op(sample_config)@hamil@basis_states[ii]*(psi_basis[ii]/psi_config)
        #print(local_hamil)
    
    return local_hamil

    


'''
    
    
def hamil_expect(basis_states, hamil, psi_basis):
    
    Computes the Hamiltonian expectatiom value
    
    basis_states -- set of all basis states 
    hamil -- Hamiltonian 
    psi_basis -- set of probability amplitudes 
    
    M = len(basis_states)
    hamil_expect = 0
    
    for m in range(M): 
        hamil_expect += local_hamil(basis_states, hamil, psi_basis)
    return (1/M)*(hamil_expect)


    #for ii in range(N):
    
#your_basis = gen_basis_state(3, create_basis_array(3))

'''




