# -*- coding: utf-8 -*-
"""
Sript that calls the xode to solve the 1d diffusion equation
@author: Alexandros
"""

from utilities import gaussian_1d_norm, square_1d
from diffusion import Diffusion_1d
import matplotlib.pyplot as plt
import numpy as np

sim = Diffusion_1d( L = 1.0, T = 1.0, sigma = 0.1, Nx = 101, Nt = 1001 )

u = sim.Crank_Nicolson(lambda x: np.sin(np.pi * x / sim.L), Left_BC = 0., Right_BC = 0., BC_type = 'dirichlet' ,epsilon = 1e-5)
x = sim.x





import matplotlib.animation as animation

# Set up the figure and axes
fig, ax = plt.subplots()
ax.set_xlabel('x')
ax.set_ylabel('u(x,t)')
ax.grid()
# Set up the lines and points that will be animated
lines = ax.plot(x, u[:,0],'b')

# Set up the formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)

# Set up the function that will be called on each frame
def update(num):
    # Update the data being plotted
    lines[0].set_data(x, u[:,num])
    ax.set_title('t = '+str(round(num*sim.dt,2)))
    return lines

# Create the animation object
ani = animation.FuncAnimation(fig, update, frames=range(sim.Nt-1), interval = 0.000001, repeat=True)