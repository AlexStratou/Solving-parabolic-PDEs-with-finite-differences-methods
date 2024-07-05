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

### Examples
1. 1D alpha radiation

https://github.com/AlexStratou/Solving-parabolic-PDEs-with-finite-differences-methods/assets/174814990/73cf3b3f-7c44-44d0-9904-6f2a0d3bc387

2. 1D scattering by a step-potential


https://github.com/AlexStratou/Solving-parabolic-PDEs-with-finite-differences-methods/assets/174814990/67b623a5-a7f0-403d-834b-e52742ec7a18

3. Particle in a box time-evolution



https://github.com/AlexStratou/Solving-parabolic-PDEs-with-finite-differences-methods/assets/174814990/db9936d9-471a-4992-a1f3-d28d87ef1d28




https://github.com/AlexStratou/Solving-parabolic-PDEs-with-finite-differences-methods/assets/174814990/40ed24c5-6a5b-4459-b868-e709116855f6

4. 2D diffusion



https://github.com/AlexStratou/Solving-parabolic-PDEs-with-finite-differences-methods/assets/174814990/a9859500-c8f8-470d-9361-55b4fef2cfd5



5. 2D double-slit experiment



https://github.com/AlexStratou/Solving-parabolic-PDEs-with-finite-differences-methods/assets/174814990/d2dff9e7-16ed-4bc4-a8e5-a9462be6367c

### References

1. Trachanas, S. (2018). *An Introduction to Quantum Physics: A First Course for Physicists, Chemists, Materials Scientists, and Engineers*. Wiley-VCH.
2. Crank, J., & Nicolson, P. (1947). A practical method for numerical evaluation of solutions of partial differential equations of the heat-conduction type. *Mathematical Proceedings of the Cambridge Philosophical Society, 43*(1), 50-67. doi:10.1017/S0305004100023197
3. Euler, L. (1768). Institutiones calculi differentialis cum eius usu in analysi finitorum ac doctrina serierum. *IMPENSIS ACADEMIAE IMPERIALIS SCIENTIARUM*. Retrieved from https://archive.org/details/institutionescal00eule
4. Excellent repo by [artmenlope](https://github.com/artmenlope) (https://github.com/artmenlope/double-slit-2d-schrodinger) that partially inspired my 2D Schrodinger Crank-Nicolson scheme.

