# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:59:40 2024

@author: Alexandros Stratoudakis
"""

from utilities import  gaussian_wave_packet_2d, double_slit_potential
from schrodinger import Schrodinger_2d
import matplotlib.pyplot as plt
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import matplotlib.animation as animation
import matplotlib as mpl
t_start = time.time()
sim = Schrodinger_2d(L = 10.0, xy0=(-5,-5), T = 0.5, Nxy = 203, Nt = 1001)

#------------------parameters-----------------------
save_animation = True
save_result = True    # save solution as .npy
load_data = False     # load a .npy file
#-------wave-packet-----------
ky0= 10.
kx0= 0.
E =round( (kx0**2+ky0**2)/2, 2)
#---double_slit-----------
x0=0.0
y0=0.
dist = 1.6
thickness = 0.2
width = 0.2
Vmax=400.
screen = int(4/5 * sim.Nxy) # x_screen = screen * sim.dx
#---------------------------------------------------
#simulation
if load_data==False:
    u = sim.Crank_Nicolson(lambda x,y: gaussian_wave_packet_2d(x, y, lamda = (2,2), center = (0, -3.5), k0=(kx0,ky0))  ,  normalize_input = True,
                           V=lambda x,y: double_slit_potential(x , y, x0=x0,y0=y0 ,dist = dist, thickness = thickness, width=width, Vmax=Vmax))
elif load_data==True:
    sim.x, sim.dx = np.linspace(sim.x0, sim.x1, sim.Nxy-2, retstep=True)
    sim.y = np.linspace(sim.y0, sim.y1, sim.Nxy-2)
    folder = 'animations/2d_schrodinger/data/' 
    name = 'double_slit_E'+str(E)+'_w'+str(width)+'_th'+str(thickness)+'_d'+str(dist)+'_dx'+str(sim.dx)+'_dt'+str(sim.dt)+'.npy'
    u = np.load(folder+name)



##Heatmap Plot
mpl.rc('font',**{'family':'serif','serif':['Times New Roman']})
mpl.rcParams['font.size']=18
mpl.rcParams. update({ "text.usetex": True, "font.family": "Computer Modern Roman" })



fig, (ax1,ax2)= plt.subplots(1,2, figsize = (20,9))
fig.tight_layout(pad=5)

line1 = ax1.pcolormesh(sim.x, sim.y, (np.abs(u[:,:,0])**2).T, cmap=plt.cm.jet,
               vmin=0, 
               #vmax = max_heatmap_value
               )
cb = fig.colorbar(line1, ax=ax1, fraction=0.046, pad=0.04)    
rep=0
def plotheatmap(u, n):
    ax1.clear()
    if len(fig.axes)>=3:
        fig.delaxes(fig.axes[2])
    ax1.cla()
    fig.suptitle('Gaussian wave-packet through double slit \n E = '+str(E)+', w = '+str(width)+', d = '+str(dist)+' \n'+r'$t =$'+' '+str(round(sim.dt * n, 3)), fontsize=26)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_aspect('equal')
    ax2.cla()
    ax2.set_box_aspect(1)
    line2 = ax2.plot(sim.x,(np.abs(u[screen,:,n])**2).T )
    ax2.set_title(r'Screen at $x=$'+' '+str(round(sim.x[screen], 2)))
    # This is to plot u_k (u at time-step k)
    line1 = ax1.pcolormesh(sim.x, sim.y, (np.abs(u[:,:,n])**2).T, cmap=plt.cm.jet,
                   vmin=0, 
                   #vmax = max_heatmap_value
                   )
    ax1.vlines(sim.x[screen], sim.y[0],sim.y[-1],'red',linestyles='dashed', zorder=2)
    ax1.vlines(x0-thickness/2, sim.y[0], y0-dist/2-width/2, colors='white', zorder=2)
    ax1.vlines(x0-thickness/2, y0-dist/2+width/2, y0+dist/2-width/2, colors='white', zorder=2)
    ax1.vlines(x0-thickness/2, y0+dist/2+width/2, sim.y[-1], colors='white', zorder=2)
    ax1.vlines(x0+thickness/2, sim.y[0], y0-dist/2-width/2, colors='white', zorder=2)
    ax1.vlines(x0+thickness/2, y0-dist/2+width/2, y0+dist/2-width/2, colors='white', zorder=2)
    ax1.vlines(x0+thickness/2, y0+dist/2+width/2, sim.y[-1], colors='white', zorder=2)
    ax1.set_title(r'$|\Psi (\vec r, t)|^2$')
    ax1.set_xlim(-5,5)
    ax1.set_ylim(-5,5)
    ax2.set_ylim(0,np.max((np.abs(u[int(sim.Nxy/2):,:,n].T)**2)))
    ax2.grid()
    ax2.set_xlabel(r'y')
    ax2.set_ylabel(r'$|\Psi (\vec r, t)|^2$')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='4%', pad=0.05)
    fig.colorbar(line1, cax=cax, orientation='vertical')
    return [line1, line2]

def animate(k):
    plotheatmap(u, k)

anim = animation.FuncAnimation(fig, animate, interval=1, frames=sim.Nt-1, repeat=True)

if save_animation ==True:
    # saving to m4 using ffmpeg writer 
    folder = 'animations/2d_schrodinger/'    
    name = 'double_slit_E'+str(E)+'_w'+str(width)+'_th'+str(thickness)+'_d'+str(dist)+'_dx'+str(sim.dx)+'_dt'+str(sim.dt)+'.mp4'
    writervideo = animation.FFMpegWriter(fps=120) 
    anim.save(folder+name, writer=writervideo) 
if save_result==True:
    folder = 'animations/2d_schrodinger/data/' 
    name = 'double_slit_E'+str(E)+'_w'+str(width)+'_th'+str(thickness)+'_d'+str(dist)+'_dx'+str(sim.dx)+'_dt'+str(sim.dt)+'.npy'
    np.save(folder+name, u)
print('Runtime =',(time.time() - t_start)/60,'mins')