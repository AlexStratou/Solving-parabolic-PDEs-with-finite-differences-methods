# -*- coding: utf-8 -*-
"""
Utility functions
@author: Alexandros Stratoudakis
"""
import numpy as np

def gaussian_1d_norm(x, center = 0.5 , std_dev = 0.1):
    """
    Normalised (in -infty to infty) gaussian function

    Parameters
    ----------
    x : 1d array-like or float
        Where to compute the function 
    center : float
        Center of the distribution.
    std_dev : float
        Standard deviation.
     """
    return 1 / (std_dev * np.sqrt(2*np.pi)) * np.exp (-(x-center)**2 / (2 * std_dev**2) )


def gaussian_1d(x, center = 0.5, std_dev = 0.1, peak = 1.):
    """
    Gaussian function.

    Parameters
    ----------
    x : 1d array-like or float
        Where to compute the function.
    center : float, optional
        Center of the distribution. The default is 0.5.
    std_dev : float, optional
        Standard deviation. The default is 0.1.
    peak : float, optional
        Maximum of the distribution. The default is 1..

    Returns
    -------
    float or array
        value/es of the function.
    """
    return peak * np.exp (-(x-center)**2 / (2 * std_dev**2) )

def square_1d(x, center = 0.5, width = 0.1, peak = 1.):
    """
    A rectangle function.

    Parameters
    ----------
    x : array-like or float
        Where to comute function.
    center : float, optional
        Center of the distribution. The default is 0.5.
    width : float, optional
        Width of the rectangle. The default is 0.1.
    peak : float, optional
        Maximum of the function. The default is 1..

    Returns
    -------
    float
        Value of function.

    """
    try: len(x)
    except:
        if (x < center - width/2) or (x > center + width/2):
            return 0.
        else: return peak
    
    ret=[]
    for x0 in x:
        if (x0 < center - width/2) or (x0 > center + width/2):
            ret.append(0.)
        else: ret.append(peak)
    return ret

def gaussian_2d_norm(x, y, center = (0.5,0.5) , std_dev = 0.1):
    """
    Normalised (in -infty to infty) gaussian function

    Parameters
    ----------
    x ,y : 1d array-like or floats
        Where to compute the function 
    center : tuple
        Center of the distribution.
    std_dev : float
        Standard deviation.
     """
    return (1 / (std_dev * np.sqrt(2*np.pi)))**2 * np.exp ((-(x-center[0])**2  - (y-center[1])**2 ) / (2 * std_dev**2)**2 )

def gaussian_2d(x, y, center = (0.5,0.5) , std_dev = 0.1, peak = 1.):
    """
    

    Parameters
    ----------
    x ,y : 1d array-like or floats
        Where to compute the function 
    center : tuple
        Center of the distribution.
    std_dev : float
        Standard deviation.
     """
    return peak * np.exp ((-(x-center[0])**2  - (y-center[1])**2 ) / (2 * std_dev**2)**2 )



def starfish_2d(x, y):
    """
    An initial condition that looks like a starfish (for diffusion)
    Source: https://github.com/kimy-de/crank-nicolson-2d
    """
    R0 = .25
    eps = 5 * 0.01 / (2 * np.sqrt(2) * np.arctanh(0.9))
    
    theta = np.arctan2(y - 0.5, x - 0.5)
    ret = np.tanh(
            (R0 + 0.1 * np.cos(6 * theta) - (np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2))) / (
                                np.sqrt(2.0) * eps))

    return ret

def donut_2d(x,y,center = (0.5,0.5), in_sd = 0.2 , out_sd = 0.4, peak = 1.):
    """
    Two gaussians with opposite sing and different standars deviations that
    resemble a donut.
    
    Parameters
    ----------
    x , y : floats or array-like
    center : float, optional
        Where the center is. The default is 0.5.
    in_sd : float, optional
        Inner gaussian std_dev. The default is 0.2.
    out_sd : float, optional
        outter gaussian std_dev. The default is 0.4.
    peak : float, optional
        Peak of the gaussians. The default is 1..

    Returns
    -------
    float or array
        value/s of the function at given x,y.

    """
    return -gaussian_2d(x, y, center = center, std_dev = in_sd, peak = peak) + \
        gaussian_2d(x, y, center = center, std_dev= out_sd, peak =peak)


def gaussian_wave_packet_1d(x, lamda = 0.1, center = 0.0, k0 = 1. ):
    """
    A wave packet with gaussian envelope.
    
    Parameters
    ----------
    x : float or array-like.
        Where to compute the function.
    lamda : float, optional
        Dispersion measure. The envelope goes like exp((x-x0)^2/4λ). The default is 0.1.
    center : float, optional
        Spatial center of the wavepacket. The default is 0.0.
    k0 : float, optional
        Momentum center of the wavepacket. The default is 1..

    Returns
    -------
    float or np.array
        Wave-packet.

    """
    return np.exp(-(x-center)**2 / 4*lamda) * np.exp(1j * k0 * x)


def potential_barrier_1d(x, center = 0, width = 1, V0 = 1):
    """
    A 1d potential barrier
    Parameters
    ----------
    x : float or array-like
        where to compute the value of the potential.
    center : float, optional
        Center of barrier. The default is 0.
    width : float, optional
        Width of barrier. The default is 1.
    V0 : float, optional
        Height of the barrier. The default is 1.

    Returns
    -------
    float or array-like
        value of the potential at given point/s.

    """
    return V0 * (np.heaviside(x - center + width/2, 0.5) - np.heaviside(x - center - width/2,0.5)) 

def alpha_radiation_potential(x, width = 4.6, Vmin=-20, Z=92,  V_wall = 2000):
    """
    

    Parameters
    ----------
    x : float or array-like
        Where to compute the potential.
    width : float, optional
        Width of the strong force potential. The default is 4.6.
    Vmin : float, optional
        Depth of strong force potential. The default is -20.
    Z : float, optional
        The atomic number of the nucleus. The default is 92.
    V_wall : float, optional
        Nucleus potential. This prevents the particle entering the nucleus. The default is 2000.

    Returns
    -------
    float or np.array
        The potential at given value/s.

    """
    
    return (V_wall+Vmin) * np.heaviside(-x,0) + np.heaviside(x-width, 0)*( (2 * Z/x) - Vmin ) +Vmin


def gaussian_wave_packet_2d(x, y, lamda = (1.,1.), center = (0.,0.), k0=(1.,1.)):
    """
    A wave packet with gaussian envelope.
    
    Parameters
    ----------
    x : float or array-like.
        Where to compute the function.
    lamda : tuple : (λx,λy), optional
        Dispersion measures. The envelope goes like exp((x-x0)^2/4λ). The default is (1,1).
    center : tuple like (x0,y0), optional
        Spatial center of the wavepacket. The default is (0.,0.)
    k0 : tuple like (k0x, k0y), optional
        Momentum center of the wavepacket. The default is 1.
   
    Returns
    -------
    float or np.array
        Wave-packet.

    """
    return gaussian_wave_packet_1d(x, lamda = lamda[0], center = center[0], k0 = k0[0] ) * \
        gaussian_wave_packet_1d(y, lamda = lamda[1], center = center[1], k0 = k0[1] )
        

def double_slit_potential(x,y, x0=0.0, y0=0.0 ,dist = 5.0, thickness = 1.0, width=1.0, Vmax=100):
    """
    The potential for the double slit experiment. The wall is parallel to the y-axis.

    Parameters
    ----------
    x : array
        x-coordinates.
    y : array
        y-coordinates.
    x0 : float, optional
        Where to place the wall. The default is 0.0.
    y0 : float, optional
        Where to place the slits. The default is 0.0.
    dist : float, optional
        Distance of the slits. The default is 5.0.
    thickness : float, optional
        Thickness of the wall. The default is 1.0.
    width : float, optional
        Width of the slits. The default is 1.0.
    Vmax : float, optional
        Wall's potential. The default is 100.

    Returns
    -------
    V : TYPE
        DESCRIPTION.

    """
    #length = y[-1]-y[0]
    V=np.zeros((len(x),len(y)))
    for i,xi in enumerate(x):
        for j,yi in enumerate(y):
            if xi> x0-thickness/2 and xi < x0+thickness/2 and ((yi>y0+dist/2 +width/2 or yi<y0-dist/2-width/2) or (yi<y0+dist/2-width/2 and yi>y0-dist/2+width/2)):
                
                V[i,j]=Vmax
            else: V[i,j]=0
    
    return V