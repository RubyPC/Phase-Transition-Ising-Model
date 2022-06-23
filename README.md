# Phase-Transition-Ising-Model
Simulating a phase transition in the 2D Ising model using a Monte Carlo method and utilising a machine learning architecture to predict the order parameter.

# The Model
The Ising Model is a particular example of a thermodynamic system, and it is the model we
use to study phase transitions. It is one of few exactly solvable models where we can compute its
thermodynamic quantities and interpret them. A phase transition does not occur in the 1D Ising model, but it does in 2D. The model is a discrete mathematical description of particles, where each particle is fixed to a lattice configuration of a finite number of sites, in which each site can be in one of two states.  A spin configuration, σ, of the system is given by σ = {σ_i|i ∈ Λ, σ_i ∈ {−1, 1}}.

The energy (Hamiltonian) of the model is given by:

![equation](https://latex.codecogs.com/svg.image?H(%5Csigma)=-J%5Csum_%7B%3Ci,j%3E%7D%7B%5Csigma_i%5Csigma_j%7D%20-%20h%5Csum_%7Bi%7D%20%7B%5Csigma_i%7D),

and for our purposes, we set the external electric field term, h=0, and the coupling constant, J=1.
We are interested in simulating the magnetisation:

![equation](https://latex.codecogs.com/svg.image?M(h,T)=%5Cfrac%7B1%7D%7BN%5E2%7D%5Csum_%7Bi%7D%20%7B%5Csigma_i%7D),

and specific heat:

![equation](https://latex.codecogs.com/svg.image?C(h,T)=%5Cfrac%7B1%7D%7Bk_B%7D%5Cfrac%7B%5Cpartial%5E2%5Cln(Z)%7D%7B%5Cpartial%20%5Cbeta%5E2%7D),

of the model in 2D.
The magnetisation of the Ising model is also known as the order parameter; it distinguishes the two phases of the model which are before and after a phase transition occurs (or between a low temperature phase and a high temperature phase).

# MCMC Simulation
To simulate the macroscopic properties above, you will need to run

> MC_Simulation.py  

which will ask to enter a 2D lattice size. The code plots a Monte Carlo simulation of the two macroscopic properties against the analytical result, and also stores the spin configurations each with their individual magnetisation value. The Monte Carlo approach to simulating the Ising model is that of the Metropolis-Hastings algorithm as follows:

1. A random configuration of spins of size N × N is intialised.
2. A spin is chosen at random and is flipped (meaning if its spin state was previously +1 it will now be −1 and vice versa).
3. The change in energy of the spin, ∆H, is calculated.
4. If the change in energy, ∆H ≤ 0, then the flip is accepted.
5. Otherwise, if ∆H > 0, the spin is flipped if and only if ![equation](https://latex.codecogs.com/svg.image?e%5E%7B-%5Cbeta%20H%7D) > r where r is a random number on the interval [0, 1] and ![equation](https://latex.codecogs.com/svg.image?%5Cbeta%20=1/k_BT).
6. This process is repeated until every spin in the configuration has been flipped.
7. An updated configuration is returned and the MC time is increased by 1.

Two plots will be generated of the magnetisation and specific heat as a function of the reduced temperature given by ![equation](https://latex.codecogs.com/svg.image?T_r=J/k_B%20) clearly showing that a phase transition occurs:

![image](https://user-images.githubusercontent.com/106536925/175266532-8477ad1b-00d5-453b-95e5-4bbe09c9b6e1.png)

for a lattice of size 10 x 10. For larger lattice sizes, the simulation of the specific heat moreso matches that of the analytical result.

Additionally, you can plot instantaneous configurations of the L=10 x 10 system as the system coarsens to its equilibrium state for different temperatures. For this, you will need to run

> Config.py
 
The interest is in a small temperature (~0.5), a large temperature (~3.0) and a temperature close to the critical temperature (~2.2).

![image](https://user-images.githubusercontent.com/106536925/175269071-6909b8e7-0dea-4331-a2ce-32a01d531d53.png)

Since ![equation](https://latex.codecogs.com/svg.image?T_r=J/k_B%20%3C%20T_c), this system should be ferromagnetic, which is clear from both of the updated
configurations as the spins are beginning to form domains, more evident for small temperatures. By inspection, it is also evident that the periodic boundary conditions are correctly implemented.

# Machine Learning
Following *Machine Learning Phases of Matter*[^1] published in Nature Physics, a Feed Forward Neural Network with a single hidden layer of 100 neurons can be used to predict the order parameter of the 2D Ising model. After running MC_Simulation.py, you should have the spin configurations with their individual magnetisation value as their label saved in a .pickle file. (Code not given) will load the spin configurations and their labels and split them for training and testing. The output of the neural network with errorbars is given by the output neuron, where the network was trained and tested on a range of lattice sizes (L=10,20,30,40,60). The output is plotted as a function of the reduced temperature.

![image](https://user-images.githubusercontent.com/106536925/175271788-c54763ad-ee48-4adc-9b5e-475ae8546b51.png)

The neural network is able to learn the order parameter of the phase transition with high accuracy. In other words, the neural network is able to distinguish the two phases of the model: before the phase transition and after the phase transition, correctly identifying the magnetisation as the relevant order parameter. 

[^1]:Juan Carrasquilla and Roger G Melko. Machine learning phases of matter. Nature Physics, 13(5):431–434, 2017.
