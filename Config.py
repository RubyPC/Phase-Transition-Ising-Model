# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 10:50:57 2021

@author: 44797
"""

# =============================================================================
# Configuration plot for small and large temperatures and for a temperature
# close to the critical temperature.
# =============================================================================

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib.colors as c


class Ising():
    
    # initialise a random configuration
    def State(self, N):
        # Generates a random spin configuration
        state = 2 * np.random.randint(2, size=(N,N))-1
        return state
    
    def Metropolis(self, config, beta, J):
    
        # Monte Carlo moves using a Metropolis algorithm 
        N = len(config)
        for i in range(N):
            for j in range(N):
                x = np.random.randint(0,N)
                y = np.random.randint(0,N)
         
                spin = config[x,y]
                neighbours = config[(x+1)%N,y] + config[x,(y+1)%N] + config[(x-1)%N,y] + config[x,(y-1)%N]
                # imposed boundary conditions same as above
                # calculate the change in energy, delta E
                energy_change = 2 * J * spin * neighbours
            
                r = rand()
                if energy_change < 0:
                    spin *= -1
                    # if the change in energy is less than zero, flip the spin
                
                elif r < min(1,np.exp(-energy_change * beta)):
                    spin *= -1
                    # otherwise accept the move with the given probability 
                    # (since beta=1/K_b) and flip the spin
                
                config[x,y] = spin
            
        return config
    # returns the configuration of spins
    
    # start by simulating the configuration for small temperature (~0.5)
    def Sim_SmallT(self):
        # set the parameters
        N = 100
        J = 1.
        t_small = .5
        config = self.State(N)
        # initialise the plot and plot the initial configuration
        f1 = plt.figure(figsize=(12,4), dpi=80)
        self.PlotSmall(f1, config, 0, N, 1)

        nt = 101 #number of time points
        
        # plot the configurations after 4 and 100 time stamps
        for i in range(nt):
            beta = 1./t_small
            self.Metropolis(config, beta, J)
            if i == 4: 
                self.PlotSmall(f1, config, i, N, 2)
            if i == 100:
                self.PlotSmall(f1, config, i, N, 3)
        
    # simulating the configurations for large T
    def Sim_LargeT(self):
        # setting parameters
        N = 100
        J = 1.
        t_large = 4.
        # initialise the configuration
        config = self.State(N)
        # initialise the plot and plot the initial configuration
        f2 = plt.figure(figsize=(12,4), dpi=80)
        self.PlotLarge(f2, config, 0, N, 1)
        
        nt = 101 #number of time points
        
        # plot the configuration for each time stamp
        for j in range(nt):
            beta = 1./t_large
            self.Metropolis(config, beta, J)
            if j == 4:
                self.PlotLarge(f2, config, j, N, 2)
            if j == 100:
                self.PlotLarge(f2, config, j, N, 3)
    
    # simulating the configrations for T close the to critical temperature
    # Tc ~ 2.667
    def Sim_Tc(self):
        # setting parameters
        N = 100
        J = 1.
        t_Tc = 2.5
        # initialise the configuration
        config = self.State(N)
        # initilaise a figure for the plot and plot the initial configuration
        f3 = plt.figure(figsize = (12,4), dpi=80)
        self.PlotTc(f3, config, 0, N, 1)
        
        nt = 101 # number of time points
        
        # plot the configuration for each time stamp
        for k in range(nt):
            beta = 1./t_Tc
            self.Metropolis(config, beta, J)
            if k == 4:
                self.PlotTc(f3, config, k, N, 2)
            if k == 100:
                self.PlotTc(f3, config, k, N, 3)
        
    # this plots the configurations for each value of t
    def PlotSmall(self, f1, config, i, N, n_):
        # plots the configuration as time goes on for small t
        x, y = np.meshgrid(range(N), range(N))
        # we want to plot 3 configurations in a row
        subplot = f1.add_subplot(1, 3, n_)
        # white represents 'spin up' and black represents 'spin down'
        cMap = c.ListedColormap(['w', 'k'])
        plt.setp(subplot.get_yticklabels(), visible=False)
        plt.setp(subplot.get_xticklabels(), visible=False)
        plt.pcolormesh(x, y, config, cmap=cMap)
        # show the title and the time stamp
        plt.title('Small T, Time = %d'%i)
        plt.axis('tight')
    
    def PlotLarge(self, f2, config, j, N, n_):
        # plots the configuration as time goes on for large t
        x2, y2 = np.meshgrid(range(N), range(N))
        subplot = f2.add_subplot(1, 3, n_)
        cMap = c.ListedColormap(['w', 'k'])
        plt.setp(subplot.get_yticklabels(), visible=False)
        plt.setp(subplot.get_xticklabels(), visible=False)
        plt.pcolormesh(x2, y2, config, cmap=cMap)
        plt.title('Large T, Time = %d'%j)
        plt.axis('tight')
    
    def PlotTc(self, f3, config, k, N, n_):
        # plots the configuration as time goes on for t close to the 
        # critical temperature, Tc
        x3, y3 = np.meshgrid(range(N), range(N))
        subplot = f3.add_subplot(1, 3, n_)
        cMap = c.ListedColormap(['w', 'k'])
        plt.setp(subplot.get_yticklabels(), visible=False)
        plt.setp(subplot.get_xticklabels(), visible=False)
        plt.pcolormesh(x3, y3, config, cmap=cMap)
        plt.title('T close to Tc, Time = %d'%k)
        plt.axis('tight')
        
    plt.show()
    
# run the simulation
   
# call the class
configuration = Ising()

configuration.Sim_SmallT() # simulation for small t

configuration.Sim_LargeT() # simulation for large t

configuration.Sim_Tc()     # simulation for t~T_c
        
                
    