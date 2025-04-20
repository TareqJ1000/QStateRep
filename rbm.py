# -*- coding: utf-8 -*-
'''
rbm.py 

Loads the class which implements a Restricted Boltzmann Machine (RBM).  
'''

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from functools import reduce
from hamils import gen_basis_state, compute_hamiltonian, local_hamil
import time

# This is a basic ass function to obtain the index in a list of vectors (i.e. a list of basis states) such that
# a particular vector (like one that we would sample) is equal 

def find_in_array(array_list, target):
    for i, arr in enumerate(array_list):
        if np.array_equal(arr, target):
            #print("Found at index:", i)
            return i
        
    print("No dice")

class ResBoltMan():
    def __init__(self, num_visi, num_hid, name, h=1.0): 
        
        '''
        num_visi -- number of visible neurons 
        num_hid -- number of hidden neurons 
        name -- name of RBM (useful for loading & saving)
        h -- a parameter used for the Hamiltonian 
        '''
        
        N = num_visi
        
        
        self.name = name
        self.num_visi = num_visi
        self.num_hid = num_hid
        self.hid = np.random.randint(0, 2, size=num_hid)
        
        #self.visi = np.random.randint(0, 2, size=num_visi)
        
        # Initialize the values for the trainable parameters. It is important to start with a large enough initialization or the gradient gets stuck. 
        self.weights = np.random.uniform(-3, 3, size=(self.num_visi, num_hid)) 
        self.bias_visi = np.random.uniform(0,3, size=self.num_visi)
        self.bias_hid = np.random.uniform(0, 3, size=num_hid)
        
        # Compute the Hamiltonian for an N-body system 
        self.hamil = compute_hamiltonian(N, h)
        
        # Compute the set of basis states associated w/ an N-body system 
        
        self.basis_reps = self.get_all_visi()
        self.basis_states = gen_basis_state(N)
        
        
        # Partition function. This is going to be the same for all samples. 
        self.part_func = self.partition_function()
        
        # Since the Partition function will be the same for all samples, so will its derivatives with respect to its traineable parameters 
        self.bias_visi_derivative = self.visi_bias_deri_part()
        self.weight_derivative = self.weight_deri_part()
        self.bias_hid_derivative = self.hid_bias_deri_part()
        
    
    def update_part_func(self): # We apply this everytime we perform an update on the trainaeble parameters. Do the same for the derivatives
    
        self.part_func = self.partition_function()
        self.bias_visi_derivative = self.visi_bias_deri_part()
        self.weight_derivative = self.weight_deri_part()
        self.bias_hid_derivative = self.hid_bias_deri_part()
        
        
    def gen_basis_state(self, visi):
        # This function converts a visible neuron configuration into the 2^{N} dimensional hilbert space 
        # visi -- visible neuron configuration 
        
        N = self.num_visi
        
        base_kets = []

        # Initialize the complete basis state, which occupies the 2^N dimensional Hilbert space. 
        
        for ii in range(N):
            base_kets.append(np.zeros(2, dtype=np.complex128))
        
        # Create a new copy of the base_kets 
        
        temp = np.copy(base_kets)
    
        for ii in range(N):
            if (visi[ii]==0):
                temp[ii] = np.array([1,0], dtype=np.complex128)
            elif (visi[ii]==1):
                temp[ii] = np.array([0,1], dtype=np.complex128)
                
        # Finally, compute the kronecker product iteratively across all elements in the array
        kron_result =  reduce(np.kron, temp)
        
        return kron_result
        
        
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def get_all_visi(self): # This computes every possible visible neuron configuration 
        all_visi = []
        combinations = list(product([0,1], repeat=self.num_visi))
        
        for ii in range(len(combinations)):
            combinations[ii] = np.array(combinations[ii])
        
        return combinations 
    
    
    def get_all_hid(self): # This computes every possible hidden neuron configuration
    
        all_hid = []
        combinations = list(product([0,1], repeat=self.num_hid))
        
        for ii in range(len(combinations)):
            combinations[ii] = np.array(combinations[ii])
            
        return combinations 

    
    def prob_function(self, visi): # This computes the unnormalized p(v). Here, we do the computation over all possible hidden neuron configurations 
        
        total_prob = 0 
        first_term = np.exp(np.transpose(visi)@self.bias_visi)
        second_term = 1
        
        
        '''
        for jj in range(self.num_hid):
            for ii in range(self.num_visi):
                temp += visi[ii]*self.weights[ii, jj]
            
            second_term = second_term*(1+np.exp(temp+self.bias_hid[jj]))
            # Reinitialize temp to 0 
            temp = 0
        '''
        
        #print(1 + np.exp(visi@self.weights + self.bias_hid))
        second_term = np.prod(1 + np.exp(visi@self.weights + self.bias_hid))
        
        return first_term*second_term 
    
    def partition_function(self): # Computes the partition function 
     
        total_part = 0
        
        # Get all combinations of visible neuron configurations
        
        visi_combos = self.get_all_visi()
        
        for visi in visi_combos:
            total_part += self.prob_function(visi)
            #input()
    
        #print(total_part)
        #print("BABBABABA")
        
        return total_part
    
    
    def visi_bias_deri_prob(self, visi): # This computes the derivative of the unnormalized probability with respect to the visible bias 
        first_term = np.exp(np.transpose(visi)@self.bias_visi)*visi
        second_term = np.prod(1 + np.exp(visi@self.weights + self.bias_hid))
        
        return first_term*second_term 
    
    
    def visi_bias_deri_part(self): # This computes the derivative of the partition function with respect to the visible bias
        total_part = 0
        
        # Get all combinations of visible neuron configurations
        
        visi_combos = self.get_all_visi()
        for visi in visi_combos: 
            total_part += self.visi_bias_deri_prob(visi)
            
        return total_part
    
    
    def hid_bias_deri_prob(self, visi): # This computes the derivative of the unnormalized probability with respect to the hidden bias
    
        first_term = np.exp(np.transpose(visi)@self.bias_visi)
        second_term = np.prod(1 + np.exp(visi@self.weights + self.bias_hid))
        activation_term = self.sigmoid(visi@self.weights + self.bias_hid)
        
        final_result = first_term*second_term*activation_term
        
        return final_result
    
    
    def hid_bias_deri_part(self): 
        total_part = 0
        
        # Get all combinations of visible neuron configurations
        
        visi_combos = self.get_all_visi()
        
        for visi in visi_combos: 
            total_part += self.hid_bias_deri_prob(visi)
            
        return total_part
        
    
    def weight_deri_prob(self, visi): # This computes the derivative of the probability function with respect to the matrix of 

    
        first_term = np.exp(np.transpose(visi)@self.bias_visi)
        second_term = np.prod(1 + np.exp(visi@self.weights + self.bias_hid))
        activation_term = self.sigmoid(visi@self.weights + self.bias_hid)
        
        final_result = first_term*second_term*np.outer(visi, activation_term)
        
        return final_result
    
    def weight_deri_part(self): # This computes the derivative of the partition function with respect to the matrix of weights
        total_part = 0
        
        # Get all combinations of visible neuron configurations
        
        visi_combos = self.get_all_visi()
        for visi in visi_combos: 
            total_part += self.weight_deri_prob(visi)
            
            
        return total_part
    
    def normalized_prob(self, visi):
        '''
        Computes the normalized probability 
        visi -- input visible neuron configuration
        '''
        return (1/self.part_func)*self.prob_function(visi)
    
    
    def visi_bias_deri_final(self, visi): # This computes the derivative of the NORMALIZED probability function wrt the visible bias.
     
        fac = 1/(2*np.sqrt(self.normalized_prob(visi)))
        return fac*((1/self.part_func)*(self.visi_bias_deri_prob(visi)) - ((1/self.part_func**2)*(self.prob_function(visi)*self.bias_visi_derivative)))
    
    def hid_bias_deri_final(self, visi): # This computes the derivative of the NORMALIZED probability function wrt the hidden bias. Here, the derivative is actually unchanged ... wait a second, this just evaluates to zero? 
        
        fac = 1/(2*np.sqrt(self.normalized_prob(visi)))
        return fac*((1/self.part_func*(self.hid_bias_deri_prob(visi)) - ((1/self.part_func)**2)*(self.prob_function(visi)*self.bias_hid_derivative)))
    
    def weight_deri_final(self, visi): # This computes the derivative of the NORMALIZED probability function. Gives us the entire weight matrix. 
        fac = 1/(2*np.sqrt(self.normalized_prob(visi)))
        return fac*((1/self.part_func)*(self.weight_deri_prob(visi)) - ((1/self.part_func**2)*(self.prob_function(visi)*self.weight_derivative)))
    
    
    
    def compute_prob_basis(self): 
        '''
        # Using an RBM, creates the set of probability amplitudes over all basis states

        Parameters
        ----------
        rbm : ResBoltMan
            Restricted Boltzmann Machine
        basis_states : array
            Array of every possible basis state

        Returns
        -------
        psi_basis: array
            Array of probability amplitudes for every possible basis state. This is calculated using the RBM
        '''
        
        psi_basis = np.zeros(len(self.basis_reps))
        
        #print(self.basis_states)
        #input()
        
        for ii in range(len(self.basis_reps)):
            #print(ii)
            psi_basis[ii] = np.sqrt(self.normalized_prob(self.basis_reps[ii]))
            
        return psi_basis
            
        
    def v_given_h(self, i):
        # This function computes the conditional probability that the visible neuron is equal to 1
        # b_v -> visible neuron bias
        # W -> weight vector of RBM 
        # h -> hidden neuron vector 
        # i -> i_th visible neuron
    
        sum_over_hidden = np.sum(np.array([self.weights[i,j]*self.hid[j] for j in range(self.num_hid)]))
        
        return self.sigmoid(self.bias_visi[i] + sum_over_hidden)
    
    def h_given_v(self, visi, j):
        # This function computes the conditional probability that the hidden neuron is equal to 1
        # b_h -> hidden neuron bias
        # W -> weight vector of RBM 
        # v -> input visible neuron vector 
        # j -> j_th visible neuron
        
        sum_over_visible = np.sum(np.array([self.weights[i,j]*visi[i] for i in range(self.num_visi)]))
        return self.sigmoid(self.bias_hid[j] + sum_over_visible) 
    
    def sample_new_v(self):
        # Samples a new visible vector given hidden neuron configuration 
        # v - visible neuron vector
        # h - hidden neuron vector 
        # W - weight vector
        # b_v - visible bias vector 
    
        # Let's instantiate a copy of the visible neuron vector 
        visi_neo = np.zeros(self.num_visi)
    
        # Let's create a series of uniformly distributed random values between 0 and 1 
        rand_visi = np.random.rand(self.num_visi)
    
        # The conditional probability defines the threshold. If it's less than or equal to this threshold, set the corresponding visible vector to 1. 
        
        for ii in range(self.num_visi):
            if (rand_visi[ii] < self.v_given_h(ii)):
                visi_neo[ii] = 1
            else:
                visi_neo[ii] = 0 

        return visi_neo 
    
        #self.visi = visi_neo
    
    def sample_new_h(self, visi): 
        
        # Samples a new visible vector given hidden neuron configuration 
        # v - visible neuron vector
        # h - hidden neuron vector 
        # W - weight vector
        # b_v - visible bias vector 
    
        # Let's instantiate a copy of the visible neuron vector 
        hid_neo = np.zeros(self.num_hid)
    
        # Let's create a series of uniformly distributed random values between 0 and 1 
        rand_visi = np.random.rand(self.num_hid)
    
        # The conditional probability defines the threshold. If it's less than or equal to this threshold, set the corresponding visible vector to 1. 
        for jj in range(self.num_hid):
            if (rand_visi[jj] < self.h_given_v(visi, jj)):
                hid_neo[jj] = 1
            else:
                hid_neo[jj] = 0 
        
        self.hid = hid_neo
        
        
    def gibb_sample(self, M):
        
        # Gibbs sampling algorithm
        # M - number of Gibbs sampling steps.
        
        # At the beginning, the hiiden neuron configuration should be random. 
        
        self.hid =  np.random.randint(0, 2, size=self.num_hid)
        
        # Store generated visible/hidden neuron configuration pairs
        visi_gibbs = []
        hid_gibbs = []
        
        for ii in range(M):
            # Sample from v 
            new_visi = self.sample_new_v()
            visi_gibbs.append(new_visi)
            # Sample from h 
            self.sample_new_h(new_visi)
            hid_gibbs.append(self.hid)
            
        return visi_gibbs, hid_gibbs
    
    
    
    def grad_update(self, M): # This performs the gradient update necessary to update the tuneable parameters of your matrix.
    
        # Given current RBM parameters, compute the probabilities of all possible basis states 
        
        psi_basis = self.compute_prob_basis()
        # print(psi_basis)
    
        # Using the RBM, perform Gibbs sampling M times. 
    
        visi_gibbs, hid_gibbs = self.gibb_sample(M)
        
        # Now perform the update
        
        visi_bias_update = np.zeros(self.num_visi, dtype=np.complex128)
        hid_bias_update = np.zeros(self.num_hid, dtype=np.complex128)
        weight_update = np.zeros(np.shape(self.weights), dtype=np.complex128)
        
    
        for visi in visi_gibbs: 
            # No actually compute the probability amplitude of the Gibbs sampled visi ... I don't think we need the basis state. 
            psi_visi = np.sqrt(self.normalized_prob(visi))
            # For the local hamiltonian, we need the STATE, not the representation! 
            visi_state = self.gen_basis_state(visi)
            # Update visible bias 
            visi_bias_update += (self.visi_bias_deri_final(visi))/(np.sqrt(self.normalized_prob(visi)))*local_hamil(visi_state, self.basis_states, self.hamil, psi_basis, psi_visi)
            # Update hidden bias 
            hid_bias_update += (self.hid_bias_deri_final(visi))/(np.sqrt(self.normalized_prob(visi)))*local_hamil(visi_state, self.basis_states, self.hamil, psi_basis, psi_visi)
            # Update weight matrix bias
            weight_update += (self.weight_deri_final(visi))/(np.sqrt(self.normalized_prob(visi)))*local_hamil(visi_state, self.basis_states, self.hamil, psi_basis, psi_visi)
        
        weight_update = (2/M)*np.real(weight_update)
        bias_hid_update = (2/M)*np.real(hid_bias_update)
        bias_visi_update = (2/M)*np.real(visi_bias_update)
        
        # Return the gradients 
        
        return weight_update, bias_hid_update, bias_visi_update
            
    
    def update_params(self, weight_update, bias_hid_update, bias_visi_update):
        
        '''
        This updates the trainable parameters of the network
        
        weight_update -- update on the weights
        bias_hid_update -- update on the hidden neuron biases
        bias_visi_update -- update on the visible neuron biases 
        '''

        self.weights -= weight_update
        self.bias_hid -= bias_hid_update
        self.bias_visi -= bias_visi_update
        
        # Don't forget to update the partition function, too
        
        self.update_part_func()
        
        
    def hamil_expect(self, M):
        
        psi_basis = self.compute_prob_basis()
        # Using the RBM, perform Gibbs sampling M times. 
        visi_gibbs, hid_gibbs = self.gibb_sample(M)
        hamil_expect = 0 
        
        for visi in visi_gibbs:
            psi_visi = np.sqrt(self.normalized_prob(visi))
            # For the local hamiltonian, we need the STATE representation, not the label! 
            visi_state = self.gen_basis_state(visi)
            # Compute the expectation value 
            hamil_expect += local_hamil(visi_state, self.basis_states, self.hamil, psi_basis, psi_visi)
            
        return (1/M)*hamil_expect
    
    # Saves the RBM's trainable parameters 
    
    def save_rbm(self, save_direc):
        
        np.savetxt(f'{save_direc}/weights_{self.name}', self.weights)
        np.savetxt(f'{save_direc}/bias_hid_{self.name}', self.bias_hid)
        np.savetxt(f'{save_direc}/bias_visi_{self.name}', self.bias_visi)
        
    # Load the RBM's trainable parameters, given the RBM's weights and stuff
    
    def load_rbm(self, rbm_name, save_direc):
        self.weights = np.loadtxt(f'{save_direc}/weights_{rbm_name}')
        self.bias_hid = np.loadtxt(f'{save_direc}/bias_hid_{rbm_name}')
        self.bias_visi = np.loadtxt(f'{save_direc}/bias_visi_{rbm_name}')
        
        
    


'''
    def grad_update(self, visi_list_0, hid_list_0, visi_list_1, hid_list_1, lr): # This computes the expectation values of quantities necessary to update the parameters
        
        for i in range(self.num_visi):
            for j in range(self.num_hid):
                self.weights[i,j] += lr*np.mean(visi_list_0[:, i]*hid_list_0[:, j] - visi_list_1[:, i]*hid_list_1[:, j])       
                                                
        for i in range(self.num_visi):
            self.bias_visi[i] += lr*np.mean(visi_list_0[:, i] - visi_list_1[:,i])
            
        for j in range(self.num_hid):
            self.bias_hid[j] += lr*np.mean(hid_list_0[:, j] - hid_list_1[:, j])
'''
    
    
        
    
    
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        

        