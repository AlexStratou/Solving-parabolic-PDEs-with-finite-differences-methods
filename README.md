# Solving-Parabolic-PDEs-with-finite-differences-methods
Solution of the heat/diffusion and time-dependent Schrodinger equations with Crank-Nicolson and Euler methods


### By: [Alexandros Stratoudakis](alexstrat4@gmail.com) 

#### Date: Spring 2024
_____________________________________________
The project aims to solve parabolic PDEs, namely the diffusion and Schrodinger equations in 1d and 2d. 

This file is instructions to the basic usage of the code. **Most classes and functions used, have a thorough doc-sheet as a comment where the parameters and returns are thoroughly described.** If someone is interested in such details, he hasn't but to open the files and read the comments on the definitions of functions and classes.

_______________________________________________________________
### Code Structure:
1. The bulk of the code is in **diffusion.py** and **schrodinger.py**. There, simulations are defined as python classes and solvers are called as methods of these classes.

2. There are four simulation classes: **Diffusion\_1d**, **Diffusion\_2d**, **Schrodinger\_1d** and **Schrodinger\_2d**. All, except 2D diffusion, use the Crank-Nicolson method to solve the equations using the method **.Crank_Nicolson**. For 2D diffusion, Euler's method is used instead with **.Euler** .

3.  The **utilities.py** script, contains useful function such as potentials, initial distributions and wave-packets.

4. All scripts that start with "run" (e.g. run\_schrodinger\_2D.py) are **non-essential**. They are examples of runfiles that use the code for some cases. With minor modifications to these, one can explore the full capabilities of the code.

____________
### How to run the code:
As mentioned before the example runfiles provided can be a good starting point for *any* application. However, I will outline the process in case someone wants to make his own.

1. It is **strongly advised** to open the parent (to the files) directory "as a project" with either Spyder (suggested) or PyCharm.

2. Import the class of the desired solver along with anything you need from the utilities script.

3. Instantiate the simulation, e.g. **sim = Schrodinger\_1d( L = 50., x0=0, T = 10, Nx = 2501, Nt = 10001 )**

4. Use the solver method, e.g. **u = sim.Crank\_Nicolson(lambda x: gaussian\_wave\_packet\_1d(x, lamda=8, center = 10, k0 = 10)  ,
                       Left\_BC = 0., Right\_BC = 0., BC\_type = 'dirichlet', normalize\_input = True,
                       V=0)**

5. Plot and/or save results e.g. with matplotlib.

The above example will simulate the propagation of a Gaussian wave-packet.

Note that in many cases, especially in 2D, the program can take significant time to run.
______________________________________________


https://github.com/AlexStratou/Solving-parabolic-PDEs-with-finite-differences-methods/assets/174814990/73cf3b3f-7c44-44d0-9904-6f2a0d3bc387


