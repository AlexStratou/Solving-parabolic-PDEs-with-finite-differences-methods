# -*- coding: utf-8 -*-
"""

@author: Alexandros Str4atoudakis
"""
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


class Schrodinger_1d:
    ''' This class defines and solves the 1d diffusion equation'''
    def __init__(self, L = 1.0, x0=0., T = 1.0, Nx = 101, Nt = 2001 ):
        """
        Initalization of the diffusio_1d class:       
            
        Parameters
        ----------------------------------------------------------------------
        L : float, optional
            Length of spatial domain. The default is 1.0.
        x0 : float, optional
             Starting point of x-axis.
        T : float, optional
            Time interval. The default is 1.0.
        
        Nx : int , optional
            Number of spatial grid points. The default is 101.
        Nt : int optional
            Number of time steps,. The default is 2001.
            """
        self.x0 = x0
        self.x1 = x0 + L
        self.L = L
        self.T = T
        self.sigma = 1j/2 #Schroedinger diffusion coefficient with hbar=m=1
        self.Nx = Nx
        self.Nt = Nt
        self.dx = L / (Nx - 1) # x-step
        self.dt = T / (Nt - 1) # t-step
        self.x = np.linspace(self.x0, self.x1, Nx)
        
    def Crank_Nicolson(self, IC, Left_BC = 0, Right_BC = 0, BC_type = 'dirichlet', normalize_input = True, V = lambda x: 0):
        """
        

        Parameters
        ----------
       IC : function
            Initial condition.
        Left_BC : complex, optional
            Left boundary condition. The default is 0.
        Right_BC : complex, optional
            Right boundary condition. The default is 0.
        BC_type : str, optional
            Type of BC. The default is 'dirichlet'.
        normalize_input : bool, optional
            Wheather to normalise the IC. The default is True.
        V: function, optional
            Potential function. The default is lambda x: 0
        Returns
        -------
        array
            Solution.

        """

        self.u = np.zeros((self.Nx, self.Nt), dtype = complex) # initialize solution
        self.u[:,0] = IC(self.x) # initial condition
        if normalize_input == True:
            norm_sq = self.dx * np.sum( abs( self.u[:,0])**2 )
            self.u[:,0] = self.u[:,0]/np.sqrt(norm_sq)
        self.V = V(self.x)   #potential to column vector
        self.v = self.V * self.dt /2 # v=V*dt/2
        self.beta = self.sigma * self.dt / (2 * self.dx**2) # CFL number: must be small
        print('|beta| =', abs(self.beta),'dt =', self.dt,'dx =', self.dx)
        
        # AU^{n+1} = BU^n
        A = np.diag(1 + 2 * self.beta * np.ones(self.Nx, dtype = complex) +1j * self.v * np.ones(self.Nx, dtype=complex)) +\
            np.diag(-self.beta * np.ones(self.Nx - 1, dtype = complex), k=-1)\
            + np.diag(-self.beta * np.ones(self.Nx - 1, dtype = complex), k=1)
        B = np.diag(1 - 2 * self.beta * np.ones(self.Nx, dtype = complex) - 1j * self.v * np.ones(self.Nx, dtype=complex)) + \
            np.diag(self.beta * np.ones(self.Nx - 1,dtype = complex), k=-1)\
            + np.diag(self.beta * np.ones(self.Nx - 1, dtype = complex), k=1)
            
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
                
                # Calculate total probability as a check
                if n%100 == 0:
                   
                   P = self.dx * np.sum(abs(self.u[:,n])**2) 
                   print('Time-step =',n, 'Prob =', P)
               
        
       
                
        elif BC_type == 'periodic':
            for n in range(0, self.Nt-1):
                # Periodic
                if (n%2)==0 : self.u[0,n] = self.u[self.Nx-1,n]
                else: self.u[self.Nx-1,n] = self.u[0,n] 
                # -------------
                self.u[:,n+1] = np.dot(G , self.u[:, n])
                
                
                if n%500 == 0:
                    P = self.dx * np.sum(abs(self.u[:,n])**2) 
                    print('Time-step =',n, 'Prob =', P)
                  
    

        if n==self.Nt-2 : print('Maximum time-steps reached.')
        
        
        return self.u
        






class Schrodinger_2d:
    ''' This class defines and solves the 2d diffusion equation'''
    def __init__(self, L = 1.0, xy0 =(0.,0.), T = 1.0, Nxy = 101, Nt = 2001 ):
        """
        Initalization of the diffusio_1d class:       
            
        Parameters
        ----------
        L : float, optional
            Length of spatial box domain. The default is 1.0.
        xy0 : tuple like (x0,y0), optional
            Origin of coordinates
        T : float, optional
            Time interval. The default is 1.0.
        Nxy : int , optional
            Meshgrid size. The default is 101.
        Nt : int optional
            Number of time steps,. The default is 2001.
            """

        self.L = L
        self.x0 = xy0[0]
        self.y0  = xy0[1]
        self.x1 = self.x0 + self.L
        self.y1 = self.y0 + self.L
        self.T = T
        self.sigma = 1j/2 #schroedinger diffusion coefficient
        self.Nxy = Nxy
        self.dim = Nxy**2
        self.Nt = Nt
        self.dx = L / (Nxy - 1) # x-step
        self.dt = T / (Nt - 1) # t-step
        self.x = np.linspace(self.x0, self.x1, Nxy)
        self.y = np.linspace(self.y0, self.y1, Nxy)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
    
        
    
    def Crank_Nicolson(self, IC, normalize_input = True, V = lambda x: 0):
        """
        

        Parameters
        ----------
        IC : function
            The initial condition.
        normalize_input : bool, optional
            Wheather or not to normalize the initial condition. The default is True.
        V : function, optional
            Potential function. The default is lambda x: 0.

        Returns
        -------
        array
            Wavefunction.

        """
        self.V = np.zeros((self.Nxy,self.Nxy), dtype = np.complex64)
        self.V[:,:] = V(self.x, self.y) #potential
        Vref=200  # A potential to be applied on the bountary to enforce reflective BC. Must be large.
        self.V[:,0] = self.dt * Vref
        self.V[:,self.Nxy-1] = self.dt * Vref
        self.V[0,:] = self.dt * Vref
        self.V[self.Nxy-1,:] = self.dt * Vref
        
        #redefine mesh for the CN scheme
        self.x, self.dx = np.linspace(self.x0, self.x1, self.Nxy-2,retstep=True) 
        self.y ,self.dy= np.linspace(self.y0, self.y1, self.Nxy-2,retstep=True) 
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        self.Ncol = (self.Nxy-2)**2
        
        self.u = np.zeros( (self.Nxy-2 ,self.Nxy-2, self.Nt) , dtype = np.complex64) # initialize solution u = u(x,y,t)
        self.u[:,:,0] = IC(self.X, self.Y) # initial condition
        self.u[0,:,0] = self.u[-1,:,0] = self.u[:,0,0] = self.u[:,-1,0] = 0. #zero at boundaries
        
        if normalize_input == True:
            norm_sq = self.dx**2 * np.sum( abs( self.u[:,:,0])**2 )
            self.u[:,:,0] = self.u[:,:,0]/np.sqrt(norm_sq)
        self.beta = -self.dt/(2j*self.dx**2)
        print('|beta| = ', abs(self.beta),'dt =', self.dt,'dx = dy =', self.dx)
        
        
        A = np.zeros((self.Ncol,self.Ncol), np.complex64)
        B = np.zeros((self.Ncol,self.Ncol), np.complex64)
       
        # Calculate A and B CN matrices.
        print("Calculating Crank Nicolson Matrices...")
        for k in range(self.Ncol):     
            
            i = 1 + k//(self.Nxy-2)
            j = 1 + k%(self.Nxy-2)
            
            # Main central diagonal.
            A[k,k] = 1 + 2*self.beta + 2*self.beta + 1j*self.dt/2*self.V[i,j]
            B[k,k] = 1 - 2*self.beta - 2*self.beta - 1j*self.dt/2*self.V[i,j]
            
            if i != 1: # Lower secondary diagonal.
                A[k,(i-2)*(self.Nxy-2)+j-1] = -self.beta 
                B[k,(i-2)*(self.Nxy-2)+j-1] = self.beta
                
            if i != self.Nxy-2: # Upper secondary diagonal.
                A[k,i*(self.Nxy-2)+j-1] = -self.beta
                B[k,i*(self.Nxy-2)+j-1] = self.beta
            
            if j != 1: # Lower main diagonal.
                A[k,k-1] = -self.beta 
                B[k,k-1] = self.beta 

            if j != self.Nxy-2: # Upper main diagonal.
                A[k,k+1] = -self.beta
                B[k,k+1] = self.beta
        print('Done!')
        
        
        A_sparce = csc_matrix(A)
       
       # solve A*u_vect[n+1]=B*u_vect[n]
        for n in range(1,self.Nt):
            
               
            
            u_vect = self.u[:,:,n-1].reshape((self.Ncol)) # make u[i,j] a column vector
            b = np.matmul(B,u_vect) # We calculate the RHS array.
            u_vect = spsolve(A_sparce,b) #solve the system
            self.u[:,:,n] = u_vect.reshape((self.Nxy-2,self.Nxy-2)) #reshape u back to 2d
            P = self.dx**2  * np.sum(abs(self.u[:,:,n])**2)  #calculate probability
            print('Time-step:',n,'/',self.Nt, 'Prob =',P)
            
        return self.u
    
