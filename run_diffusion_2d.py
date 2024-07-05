# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:37:05 2024

@author: Alexandros Stratoudakis
"""

from utilities import gaussian_2d_norm, gaussian_2d,donut_2d, starfish_2d
from diffusion import Diffusion_2d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
sim = Diffusion_2d(L = 0.8, T = 2.5, sigma = 0.01, Nxy = 201, Nt = 8001)

u = sim.Euler( starfish_2d  , Left_BC = 0., Right_BC = 0,
              Up_BC = 0., Down_BC = 0, BC_type = 'neumann', epsilon = 0.001)
save=True
def plotheatmap(u, n):
    plt.clf()

    plt.title('t ='+str(round(sim.dt * n, 3)))
    plt.xlabel("x")
    plt.ylabel("y")

    # This is to plot u_k (u at time-step k)
    plt.pcolormesh(sim.x,sim.y,u[:,:,n], cmap=plt.cm.jet, vmin=0, vmax = 1)
    plt.colorbar()

    return plt

def animate(k):
    plotheatmap(u, k)

anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=sim.Nt-1, repeat=False)

# saving to m4 using ffmpeg writer 
writervideo = animation.FFMpegWriter(fps=320) 
if save==True:
    anim.save('diff2D_insulation_dirichlet.mp4', writer=writervideo) 