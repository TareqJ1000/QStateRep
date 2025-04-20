# -*- coding: utf-8 -*-
"""
train.py

Script to train RBMs to find ground state of Periodic TFIM Ising model.
"""

import numpy as np
import matplotlib.pyplot as plt 
from rbm import ResBoltMan
import argparse 

import yaml
from yaml import Loader


import os 


# This updates the trainable parameters using regular SGD 

def SGD(res_bolt, M, lr=0.001):
    
    '''
    res_bolt -- RBM network 
    M -- Number of times we sample the probability distribution approximated by the RBM using Gibbs sampling
    lr -- learning rate
    '''
    # Compute the gradients
    weight_grad, bias_hid_grad, bias_visi_grad = res_bolt.grad_update(M)
    
    # The update rule is simply the grads times the learning rate 
    weight_update, bias_hid_update, bias_visi_update = lr*weight_grad, lr*bias_hid_grad, lr*bias_visi_grad 
    
    # Update the RBM according to these rules. 
    res_bolt.update_params(weight_update, bias_hid_update, bias_visi_update)


# This updates the parameters using the Adam routine
 
def Adam(res_bolt, M, first_mom, second_mom, t,  lr=0.001, rho_1=0.1, rho_2=0.999):
    
    '''
    res_bolt -- RBM network 
    M -- Number of times we sample the probability distribution approximated by the RBM using Gibbs sampling 
    first_moment, second_moment -- first and second momentum gradients. These are usually initialized to zero, and the updated vectors must be passed for each iteration. 
    # these are lists containing the gradients of each parameters 
    t -- time step
    lr -- learning rate 
    rho_1, rho_2 -- first and second exponential decay rates
    '''
    
    first_moment = first_mom.copy()
    second_moment = second_mom.copy()

    # Numerical stability parameter 
    delta = 1e-8
    
    # Compute the gradients
    weight_grad, bias_hid_grad, bias_visi_grad = res_bolt.grad_update(M)
    
    # Update first and second moments
    first_moment[0] = rho_1*first_moment[0] + (1-rho_1)*(weight_grad)
    first_moment[1] = rho_1*first_moment[1] + (1-rho_1)*(bias_hid_grad)
    first_moment[2] = rho_1*first_moment[2] + (1-rho_1)*(bias_visi_grad)
    
    second_moment[0] = rho_2*second_moment[0] + (1-rho_2)*(weight_grad*weight_grad)
    second_moment[1] = rho_2*second_moment[1] + (1-rho_2)*(bias_hid_grad*bias_hid_grad)
    second_moment[2] = rho_2*second_moment[2] + (1-rho_2)*(bias_visi_grad*bias_visi_grad)
    
    # Apply 'correction' factor
    
    first_moment_corr = first_moment.copy()
    second_moment_corr = second_moment.copy()
    
    
    first_moment_corr[0], first_moment_corr[1], first_moment_corr[2] = first_moment[0]/(1-rho_1**(t+1)), first_moment[1]/(1-rho_1**(t+1)), first_moment[2]/(1-rho_1**(t+1))
    second_moment_corr[0], second_moment_corr[1], second_moment_corr[2] = second_moment[0]/(1-rho_2**(t+1)), second_moment[1]/(1-rho_2**(t+1)), second_moment[2]/(1-rho_2**(t+1))
    
    # Finally, compute update 
    
    weight_update = lr*(first_moment_corr[0])/(np.sqrt(second_moment_corr[0]) + delta)
    bias_hid_update = lr*(first_moment_corr[1])/(np.sqrt(second_moment_corr[1]) + delta)
    bias_visi_update = lr*(first_moment_corr[2])/(np.sqrt(second_moment_corr[2]) + delta)

    # Update the RBM according to these rules. 
    res_bolt.update_params(weight_update, bias_hid_update, bias_visi_update)
    
    # Return the uncorrected first and second moments 
    return first_moment, second_moment


    
###########################

# parse through slurm array (for use w/ bash script. You can set shift to be a random integer for the purposes of testing locally)

parser=argparse.ArgumentParser(description='test')
parser.add_argument('--ii', dest='ii', type=int,
    default=None, help='')
args = parser.parse_args()
shift = args.ii

# OMIT IN CLUSTER 

shift = 0

# OMIT IN CLUSTER

# Load up parameters needed for training 

stream = open(f"configs/train{shift}.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

# RBM parameters

N = cnfg['N'] # Size of quantum system 
h = cnfg['h'] # h parameter
num_hid = cnfg['num_hid'] # Number of hidden neurons in RBM
name = cnfg['name'] # Name of RBM

# Training parameters 

optim_type = cnfg['optim_type'] #'sgd' or 'adam'
num_epochs = cnfg['num_epochs'] # Number of training iterations
num_samples = cnfg['num_samples'] # Number of samples to take
lr = cnfg['lr'] # Learning rate 

# Directories to save trained model and plot 
save_direc = cnfg['save_direc']
plot_direc = cnfg['plot_direc']

# Create directories (if they don't exist)

os.makedirs(f'models/{save_direc}', exist_ok=True)
os.makedirs(f'plots/{plot_direc}', exist_ok=True)

# Now, initialize the network. Also, initialize the first and second moments (if applicable)

rbm_train = ResBoltMan(N, num_hid, name, h=h)
first_moment = [np.zeros((N,num_hid)), np.zeros(num_hid), np.zeros(N)]
second_moment = [np.zeros((N,num_hid)), np.zeros(num_hid), np.zeros(N)]

hamil_expects = []

# Now, start training the network 

for ii in range(num_epochs):
    if (optim_type=='sgd'):
        SGD(rbm_train, num_samples, lr=lr)
        
    elif (optim_type=='adam'):
        first_moment, second_moment = Adam(rbm_train, num_samples, first_moment, second_moment, ii, lr=lr)
    
    # Compute the expectation value of the Hamiltonian 
    
    hamil_expect = np.real(rbm_train.hamil_expect(num_samples))
    hamil_expects.append(hamil_expect)
    
    # Report progress on the network training 
    
    if (ii%5==0):
        print(f"Epoch: {ii}, Expectation value: {hamil_expect}. Saving network...")
        # Save the network 
        rbm_train.save_rbm(f'models/{save_direc}')
        
        plt.plot(np.arange(ii+1), hamil_expects)
        plt.savefig(f'plots/{plot_direc}/{ii}.png', bbox_inches='tight')
        plt.show()


# Save the network 
# rbm_train.save_rbm()

        
        
    








 














