# -*- coding: utf-8 -*-
"""
@author: Alexandros Stratoudakis
"""

import numpy as np

class Diffusion_1d:
    ''' This class defines and solves the 1d diffusion equation'''
    def __init__(self, L = 1.0, T = 1.0, sigma = 0.01, Nx = 101, Nt = 2001 ):
        """
        Initalization of the diffusio_1d class:       
            
        Parameters
        ----------
        L : float, optional
            Length of spatial domain. The default is 1.0.
        T : float, optional
            Time interval. The default is 1.0.
        sigma : float or complex, optional
            Diffusion coefficient. The default is 0.01.
        Nx : int , optional
            Number of spatial grid points. The default is 101.
        Nt : int optional
            Number of time steps,. The default is 2001.
            """

        self.L = L
        self.T = T
        self.sigma = sigma
        self.Nx = Nx
        self.Nt = Nt
        self.dx = L / (Nx - 1) # x-step
        self.dt = T / (Nt - 1) # t-step
        self.x = np.linspace(0, L, Nx)
        
    def Crank_Nicolson(self, IC, Left_BC = 0, Right_BC = 0, BC_type = 'dirichlet', epsilon = 1e-6):
        """
        

        Parameters
        ----------
       IC : function
            Initial condition.
        Left_BC : float, optional
            Left boundary condition. The default is 0.
        Right_BC : float, optional
            Right boundary condition. The default is 0.
        BC_type : str, optional
            Type of BC. The default is 'dirichlet'.
        epsilon : float, optional
            Convergence condition. Default is 1e-6
        Returns
        -------
        array
            Solution.

        """

        self.u = np.zeros((self.Nx, self.Nt)) # initialize solution
        self.u[:,0] = IC(self.x) # initial condition
        self.epsilon = epsilon
        self.beta = self.sigma * self.dt / (2 * self.dx**2) # CFL number: must be small
        print('beta =', self.beta,'dt =', self.dt,'dx =', self.dx)
        
        # AU^{n+1} = BU^n
        A = np.diag(1 + 2 * self.beta * np.ones(self.Nx)) + np.diag(-self.beta * np.ones(self.Nx - 1), k=-1)\
            + np.diag(-self.beta * np.ones(self.Nx - 1), k=1)
        B = np.diag(1 - 2 * self.beta * np.ones(self.Nx)) + np.diag(self.beta * np.ones(self.Nx - 1), k=-1)\
            + np.diag(self.beta * np.ones(self.Nx - 1), k=1)
            
        if BC_type == 'periodic':
            # Periodic BC implies that the two extreme anti-diagonal elements be non zero
            A[0,self.Nx -1] = A[self.Nx -1, 0] = -self.beta
            B[0,self.Nx -1] = B[self.Nx -1, 0] = self.beta
        
        print('computing Ainv...')
        Ainv = np.linalg.inv(A)
        print('Ainv computed!')
        G = np.dot(Ainv,B)        # G := A^{-1} * B
        print('solving...') 
        if BC_type == 'dirichlet':
            for n in range(0, self.Nt-1):
                #Dirichlet
                self.u[0,n] = Left_BC
                self.u[self.Nx-1,n] = Right_BC
                # -------------
                self.u[:,n+1] = np.dot(G , self.u[:, n])
                
                # Calculate sqrt(Δu^2) to check if we are on equlibrum
                if n%500 == 0:
                    self.du = self.u[:,n]-self.u[:,n-1]
                    w = np.sqrt(np.sum(self.du**2))
                    print('time_step = '+str(n),'√(Σᵢ(Δuᵢ²)) =' ,w)
                    if w < self.epsilon:
                        print('Equilibrium reached!')
                        break
                    
        
        elif BC_type == 'neumann':
            for n in range(0, self.Nt-1):
                #Neumann
                self.u[0,n] = self.u[1,n] - Left_BC * self.dx
                self.u[self.Nx-1,n] =self.u[self.Nx-2,n] +  Right_BC * self.dx
                # ------------
                self.u[:,n+1] = np.dot(G , self.u[:, n])
                
                # Calculate sqrt(Δu^2) to check if we are on equlibrum
                if n%500 == 0:
                    self.du = self.u[:,n]-self.u[:,n-1]
                    w = np.sqrt(np.sum(self.du**2))
                    print('time_step = '+str(n),'√(Σᵢ(Δuᵢ²)) =' ,w)
                    if w < self.epsilon:
                        print('Equilibrium reached!')
                        break
                
        elif BC_type == 'periodic':
            for n in range(0, self.Nt-1):
                # Periodic
                if (n%2)==0 : self.u[0,n] = self.u[self.Nx-1,n]
                else: self.u[self.Nx-1,n] = self.u[0,n] 
                # -------------
                self.u[:,n+1] = np.dot(G , self.u[:, n])
                
                # Calculate sqrt(Δu^2) to check if we are on equlibrum
                if n%500 == 0:
                    self.du = self.u[:,n]-self.u[:,n-1]
                    w = np.sqrt(np.sum(self.du**2))
                    print('time_step = '+str(n),'√(Σᵢ(Δuᵢ²)) =' ,w)
                    if w < self.epsilon:
                        print('Equilibrium reached!')
                        break
                
        elif BC_type == 'dirichlet_neumann':
             for n in range(0, self.Nt-1):
                 # left Dirichlet
                 self.u[0,n] = Left_BC
                 # right Neumann
                 self.u[self.Nx-1,n] =self.u[self.Nx-2,n] +  Right_BC * self.dx
                 # ----------
                 self.u[:,n+1] = np.dot(G , self.u[:, n])
                 
                 # Calculate sqrt(Δu^2) to check if we are on equlibrum
                 if n%500 == 0:
                     self.du = self.u[:,n]-self.u[:,n-1]
                     w = np.sqrt(np.sum(self.du**2))
                     print('time_step = '+str(n),'√(Σᵢ(Δuᵢ²)) =' ,w)
                     if w < self.epsilon:
                         print('Equilibrium reached!')
                         break
                 
        elif BC_type == 'neumann_dirichlet':
             for n in range(0, self.Nt-1):
                 # Left Neumann
                 self.u[0,n] = self.u[1,n] - Left_BC * self.dx
                 # Right dirichlet
                 self.u[self.Nx-1,n] = Right_BC
                 # ----------
                 self.u[:,n+1] = np.dot(G , self.u[:, n])
                 
                 # Calculate sqrt(Δu^2) to check if we are on equlibrum
                 if n%500 == 0:
                     self.du = self.u[:,n]-self.u[:,n-1]
                     w = np.sqrt(np.sum(self.du**2))
                     print('time_step = '+str(n),'√(Σᵢ(Δuᵢ²)) =' ,w)
                     if w < self.epsilon:
                         print('Equilibrium reached!')
                         break

        if n==self.Nt-2 : print('Maximum time-steps reached.')
        
        
        return self.u
        
        
        #mod it for time-dependent BC???
        
        
class Diffusion_2d:
    ''' This class defines and solves the 2d diffusion equation'''
    def __init__(self, L = 1.0, T = 1.0, sigma = 0.01, Nxy = 101, Nt = 2001 ):
        """
        Initalization of the diffusio_1d class:       
            
        Parameters
        ----------
        L : float, optional
            Length of spatial box domain. The default is 1.0.
        T : float, optional
            Time interval. The default is 1.0.
        sigma : float or complex, optional
            Diffusion coefficient. The default is 0.01.
        Nxy : int , optional
            Meshgrid size. The default is 101.
        Nt : int optional
            Number of time steps,. The default is 2001.
            """

        self.L = L
        self.T = T
        self.sigma = sigma
        self.Nxy = Nxy
        self.Nt = Nt
        self.dx = L / (Nxy - 1) # x-step
        self.dt = T / (Nt - 1) # t-step
        self.x = np.linspace(0, L, Nxy)
        self.y = np.linspace(0, L, Nxy)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
    def Euler(self, IC, Left_BC = 0., Right_BC = 0., Up_BC = 0., Down_BC = 0., BC_type = 'dirichlet', epsilon = 1e-6):
        """
        

        Parameters
        ----------
       IC : function
            Initial condition.
        Left_BC : float, optional
            Left boundary condition. The default is 0.
        Right_BC : float, optional
            Right boundary condition. The default is 0.
        Up_BC : float, optional
            Up boundary condition. The default is 0.
        Down_BC : float, optional
            Down boundary condition. The default is 0.
        BC_type : str, optional
            Type of BC. The default is 'dirichlet'.
        epsilon : float, optional
            Convergence condition. Default is 1e-6
        Returns
        -------
        array
            Solution.

        """

        self.u = np.zeros( (self.Nxy ,self.Nxy, self.Nt) ) # initialize solution
        self.u[:,:,0] = IC(self.X, self.Y) # initial condition
        self.epsilon = epsilon
        self.beta = self.sigma * self.dt / ( self.dx**2) # CFL number: must be small <0.25
        print('beta = ', self.beta,'dt =', self.dt,'dx = dy =', self.dx)
        
            
        print('solving...') 
        if BC_type == 'dirichlet':
            for n in range(0, self.Nt-1):
                #Dirichlet
                self.u[0,:,n] = Left_BC
                self.u[self.Nxy-1, :,n] = Right_BC
                self.u[:, self.Nxy-1,n] = Up_BC
                self.u[:, 0,n] = Down_BC
                # -------------
                for i in range(1 , self.Nxy-1):
                    for j in range(1, self.Nxy-1):
                        self.u[i,j,n+1] = (1 - 4 * self.beta) * self.u[i,j,n] + self.beta * \
                            (self.u[i+1,j,n] + self.u[i-1, j, n] + self.u[i,j+1,n] + self.u[i,j-1,n])
                
                # Calculate sqrt(Δu^2) to check if we are on equlibrum
                if n%100 == 0:
                    self.du = self.u[:,:,n]-self.u[:,:,n-1]
                    w = np.sqrt(np.sum(self.du**2))
                    print('time_step = '+str(n),'√(Σᵢⱼ(Δuᵢⱼ²)) =' , w)
                    if w < self.epsilon:
                        print('Equilibrium reached!')
                        break
                    
        
        elif BC_type == 'neumann':
            for n in range(0, self.Nt-1):
                #Neumann
                self.u[0,:,n] = self.u[1,:,n] - Left_BC * self.dx
                self.u[self.Nxy-1,:,n] =self.u[self.Nxy-2,:,n] +  Right_BC * self.dx
                self.u[:,0,n] = self.u[:,1,n] - Down_BC * self.dx
                self.u[:,self.Nxy-1,n] =self.u[:,self.Nxy-2,n] +  Up_BC * self.dx
                # ------------
                for i in range(1 , self.Nxy-1):
                    for j in range(1, self.Nxy-1):
                        
                        
                        self.u[i,j,n+1] = (1 - 4 * self.beta) * self.u[i,j,n] + self.beta * \
                            (self.u[i+1,j,n] + self.u[i-1, j, n] + self.u[i,j+1,n] + self.u[i,j-1,n])
                
                # Calculate sqrt(Δu^2) to check if we are on equlibrum
                if n%100 == 0:
                    self.du = self.u[:,:,n]-self.u[:,:,n-1]
                    w = np.sqrt(np.sum(self.du**2))
                    print('time_step = '+str(n),'√(Σᵢⱼ(Δuᵢⱼ²)) =' , w)
                    if w < self.epsilon:
                        print('Equilibrium reached!')
                        break
                
                        
                
        if n==self.Nt-2 : print('Maximum time-steps reached.')
        
        
        return self.u    