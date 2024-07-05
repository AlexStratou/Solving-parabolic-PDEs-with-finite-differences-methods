# -*- coding: utf-8 -*-
"""

@author: Alexandros Stratoudakis
"""

from utilities import  gaussian_wave_packet_2d, double_slit_potential
from schrodinger import Schrodinger_2d
import matplotlib.pyplot as plt
import time
import numpy as np
import matplotlib.animation as animation
t_start = time.time()
sim = Schrodinger_2d(L = 10.0, xy0=(-5,-5), T = 1.0, Nxy = 203, Nt = 1001)

#------------------parameters-----------------------
save_animation = True
save_result = True
load_data = False
#-------wave-packet-----------
ky0= 10.
kx0= 0.
E =round( (kx0**2+ky0**2)/2, 2)
#---double_slit-----------
x0=0.0
y0=10
dist = 20
thickness = 0.2
width=2.0
Vmax=400.
#---------------------------------------------------
#simulation
if load_data==False:
    u = sim.Crank_Nicolson(lambda x,y: gaussian_wave_packet_2d(x, y, lamda = (5,5), center = (0, -3.5), k0=(kx0,ky0))  ,  normalize_input = True,
                           V=lambda x,y: double_slit_potential(x , y, x0=x0,y0=y0 ,dist = dist, thickness = thickness, width=width, Vmax=Vmax))
elif load_data==True:
    sim.x, sim.dx = np.linspace(sim.x0, sim.x1, sim.Nxy-2, retstep=True)
    sim.y = np.linspace(sim.y0, sim.y1, sim.Nxy-2)
    folder = 'animations/2d_schrodinger/data/' 
    name = 'single_slit_E'+str(E)+'_w'+str(width)+'_th'+str(thickness)+'_dx'+str(sim.dx)+'_dt'+str(sim.dt)+'.npy'
    u = np.load(folder+name)


def plotheatmap(u, n):
    plt.clf()

    plt.title('Gaussian wave-packet through single slit, E = '+str(E)+' \n t ='+str(round(sim.dt * n, 3)))
    plt.xlabel("x")
    plt.ylabel("y")

    # This is to plot u_k (u at time-step k)
    plt.pcolormesh(sim.x, sim.y, (np.abs(u[:,:,n])**2).T, cmap=plt.cm.jet,
                   vmin=0, 
                   #vmax = max_heatmap_value
                   )
    plt.vlines(x0-thickness/2, sim.y[0], y0-dist/2-width/2, colors='white', zorder=2)
    plt.vlines(x0-thickness/2, y0-dist/2+width/2, y0+dist/2-width/2, colors='white', zorder=2)
    plt.vlines(x0-thickness/2, y0+dist/2+width/2, sim.y[-1], colors='white', zorder=2)
    plt.vlines(x0+thickness/2, sim.y[0], y0-dist/2-width/2, colors='white', zorder=2)
    plt.vlines(x0+thickness/2, y0-dist/2+width/2, y0+dist/2-width/2, colors='white', zorder=2)
    plt.vlines(x0+thickness/2, y0+dist/2+width/2, sim.y[-1], colors='white', zorder=2)
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.colorbar()

    return plt

def animate(k):
    plotheatmap(u, k)

anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=sim.Nt-1, repeat=True)

if save_animation ==True:
    # saving to m4 using ffmpeg writer 
    folder = 'animations/2d_schrodinger/'    
    name = 'single_slit_E'+str(E)+'_w'+str(width)+'_th'+str(thickness)+'_dx'+str(sim.dx)+'_dt'+str(sim.dt)+'.mp4'
    writervideo = animation.FFMpegWriter(fps=180) 
    anim.save(folder+name, writer=writervideo) 
if save_result==True:
    folder = 'animations/2d_schrodinger/data/' 
    name = 'single_slit_E'+str(E)+'_w'+str(width)+'_th'+str(thickness)+'_dx'+str(sim.dx)+'_dt'+str(sim.dt)+'.npy'
    np.save(folder+name, u)
print('Runtime =',(time.time() - t_start)/60,'mins')