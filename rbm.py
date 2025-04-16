# -*- coding: utf-8 -*-
'''
rbm.py 

Loads the class which implements a Restricted Boltzmann Machine (RBM).  
'''

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# Let's define the sigmoid function. The Gibbs sampler thingy is predictated on computing conditional probabilities. 

class ResBoltMan():
    def __init__(self, num_visi, num_hid): 
        
        self.num_visi = num_visi
        self.num_hid = num_hid
        self.hid = np.random.randint(0, 2, size=num_hid)
        
        #self.visi = np.random.randint(0, 2, size=num_visi)
        self.weights = np.random.uniform(-0.05, 0.05, size=(num_visi, num_hid)) 
        self.bias_visi = np.random.uniform(0, 0.05, size=num_visi)
        self.bias_hid = np.random.uniform(0, 0.05, size=num_hid)
        
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

    
    def prob_function(self, visi): # This computes p(v). Here, we do the computation over all possible hidden neuron configurations 
        
        total_prob = 0 
        first_term = np.exp(np.transpose(visi)@self.bias_visi)
        second_term = 1
        temp = 0
        
        for jj in range(self.num_hid):
            for ii in range(self.num_visi):
                temp += np.exp(visi[ii]*self.weights[ii, jj] + self.bias_hid[jj])
            
            second_term = second_term * temp
            # Reinitialize temp to 0 
            temp = 0
        
        
        return first_term*second_term 
    
    def partition_function(self): # Computes the partition function 
     
        total_part = 0
        
        # Get all combinations of visible neuron configurations
        
        visi_combos = self.get_all_visi()
        
        for visi in visi_combos:
            total_part += self.prob_function(visi)
        
        return total_part
    
    def normalized_prob(self, visi):
        '''
        Computes the normalized probability 
        visi -- input visible neuron configuration
        '''
        return (1/self.partition_function())*self.prob_function(visi)
        
    def grad_update(self, visi_list_0, hid_list_0, visi_list_1, hid_list_1, lr): # This computes the expectation values of quantities necessary to update the parameters
        
        for i in range(self.num_visi):
            for j in range(self.num_hid):
                self.weights[i,j] += lr*np.mean(visi_list_0[:, i]*hid_list_0[:, j] - visi_list_1[:, i]*hid_list_1[:, j])       
                                                
        for i in range(self.num_visi):
            self.bias_visi[i] += lr*np.mean(visi_list_0[:, i] - visi_list_1[:,i])
            
        for j in range(self.num_hid):
            self.bias_hid[j] += lr*np.mean(hid_list_0[:, j] - hid_list_1[:, j])
            
        
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
        
        # Store generated visible/hidden neuron configuration pairs
        visi_gibbs = []
        hid_gibbs = []
        
        for ii in range(M):
            # Sample from v 
            new_visi = self.sample_from_v()
            visi_gibbs.append(new_visi)
            # Sample from h 
            self.sample_new_h(new_visi)
            hid_gibbs.append(self.hid)
            
        return visi_gibbs, hid_gibbs
    
    
        
    
    
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        

        