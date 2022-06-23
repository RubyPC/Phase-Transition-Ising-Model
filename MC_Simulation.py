# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 14:05:58 2022

@author: 44797
"""

# =============================================================================
# Simulates macroscopic properties of 2D Ising model and saves data for the
# neural network.
# =============================================================================

import numpy as np
from numpy.random import rand
import math
import matplotlib.pyplot as plt
import pickle
from numba import jit

def State(N):
    # Generates a random spin configuration
    state = 2 * np.random.randint(2, size=(N,N))-1
    return state

@jit(nopython=True)
def Energy(config):
    
    # Computes the energy/hamiltonian of a given configuration with boundary
    # conditions
    
    total_energy = 0.0
    for i in range(len(config)):
        for j in range(len(config)):
            
            neighbours = config[(i+1)%N,j] + config[i,(j+1)%N] + config[(i-1)%N,j] + config[i,(j-1)%N]
            # Above is our nearest neighbour iteractions with the imposed boundary
            # conditions. %N allows for toroidal/periodic boundary conditions
            
            total_energy += -neighbours * config[i,j]
    return (total_energy/4)

def Magnetisation(config):
    # Computes the magnetisation (without the 1/N^2 factor)
    mag = np.sum(config)
    
    return mag

# Saves data for the neural network
def save1(object, filename):
    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(object, f)
        
# saves data in a textfile
def save2(object, filename):
    with open(filename, 'a+') as f:
        f.write(object)

# Implement the Metropolis algorithm 
@jit(nopython=True)    
def Metropolis(config, beta, J):
    
    # Monte Carlo moves using a Metropolis algorithm 
    for i in range(len(config)):
        for j in range(len(config)):
            x = np.random.randint(0,N)
            y = np.random.randint(0,N)
         
            neighbours = config[(x+1)%N,y] + config[x,(y+1)%N] + config[(x-1)%N,y] + config[x,(y-1)%N]
            # imposed boundary conditions same as above
            # calculate the change in energy, delta E
            energy_change = 2 * J * config[x,y] * neighbours
            
            r = rand()
            if (energy_change < 0 or r < min(1,np.exp(-energy_change*beta))):
                config[x,y] *= -1
                # if the change in energy is less than zero, flip the spin.
                # otherwise accept the move with the given probability 
                # (since beta=1/K_b) and flip the spin
                
            config[x,y] = config[x,y]
            
    return config
    # returns the configuration of spins
    
# declaration of parameters
    
N = int(input("Lattice size:")) # size of lattice
eqSteps = 1000                  # number of steps for equilibirum 
mcSteps = 1000                  # number of monte carlo steps 
nt = 5000                       # number of reduced temperature points (tau)
config = State(N)               # configuration 
J = 1  
    
# initialise variables
T = np.linspace(1., 3.5, nt) 

# empty arrays for energy, magnetisation and specific heat
E = np.zeros(nt)
M = np.zeros(nt)
C = np.zeros(nt)

# analytic result
m_analytic = np.zeros(nt)
c_analytic = np.zeros(nt) 

# Initialising spins and labels
spins, labels = np.zeros((0, N * N)), np.zeros((0, 1))
high, low = np.array([1, 0]), np.array([0, 1])

# critical temperature
T_c = 2/math.log(1 + math.sqrt(2))

# n1 and n2 will be used to calculate the ensemble average for the heat
# capacity of E and E^2 and the magnetisation
n1 = 1.0/(mcSteps*N*N)
n2 = 1.0/(mcSteps*mcSteps*N*N)

for t in range(nt):
    # looping over the reduced temperature points
    # initial energy and magnetisation
    E1 = Energy(config)
    M1 = Magnetisation(config)
    # initial energy squared
    E2 = E1**2
    beta = 1.0/T[t]
    
    # now equibrilate
    for i in range(eqSteps):
        # monte carlo moves
        Metropolis(config, beta, J)
    
    for i in range(mcSteps):
        # run through the monte carlo steps
        Metropolis(config, beta, J)
        # calculate the energy at each time step
        Ene = Energy(config)
        # calculate the magnetisation at each time step
        Mag = Magnetisation(config)
        
        # update the energy and magnetisation and square the energy in order
        # to calculate the specific heat
        E1 = E1 + Ene
        M1 = M1 + Mag
        E2 = E2 + (Ene*Ene)
        
    spins = np.vstack((spins, config.ravel()))
    
    
    # ensemble average of energy, magnetisation and specific heat
    E[t] = n1 * E1
    M[t] = n1 * M1
    C[t] = beta**2 * (n1*E2 - n2*E1*E1)
    
    if T[t] < T_c:
        labels = np.vstack((labels, M[t]))
    else:
        labels = np.vstack((labels, M[t]))

    # analytical result        
    if T[t] < T_c:
        c_analytic[t] = 0
        m_analytic[t] = pow(1 - pow(np.sinh(2*beta), -4), 1/8)
        
    const = math.log(1 + math.sqrt(2))
    if T[t] - T_c >= 0:
        c_analytic[t] = 0
    else:
        c_analytic[t] = (2.0/np.pi) * (const**2) * (-1.0*math.log(1 - T[t]/T_c) + math.log(1.0/const) - (1 + np.pi/4))


# energy and specific heat saved to a textfile
energy = E.tolist()
energy_data = ' '.join(str(i) for i in energy)
spec_heat = C.tolist()
spec_heat_data = ' '.join(str(j) for j in spec_heat)

# Save data for neural network
save1(0.5 * (spins + 1), 'train_spins'+str(N)), save1(labels, 'train_labels'+str(N)), save1(T, 'temperature'+str(N))
print("saved data!")


# plots
f = plt.figure(figsize=(12,5))

# plot the reduced temperature vs magnetisation
axes = f.add_subplot(1, 2, 1)    
plt.scatter(T, abs(M), s=50, marker='o', color='IndianRed')
plt.plot(T, m_analytic, label='Analytic Result')
plt.xlabel('Reduced Temperature')
plt.ylabel('Magnetisation')
plt.legend()
plt.axis('tight')
 
 
# plot the reduced temperature vs specific heat   
axes = f.add_subplot(1, 2, 2)   
plt.scatter(T, C, s=50, marker='o', color='IndianRed') 
plt.plot(T, c_analytic, label='Analytic Result')
plt.xlabel('Reduced Temperature')
plt.ylabel('Specific Heat')
plt.legend()
plt.axis('tight') 

plt.savefig('MagSpecHeat.png')