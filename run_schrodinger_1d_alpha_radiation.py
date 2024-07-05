# -*- coding: utf-8 -*-
"""
Sript that calls the xode to solve the 1d schrodinger equation
@author: Alexandros Stratoudakis
"""

from utilities import gaussian_1d_norm, square_1d, gaussian_wave_packet_1d, potential_barrier_1d, alpha_radiation_potential
from schrodinger import Schrodinger_1d
import matplotlib.pyplot as plt
import numpy as np

sim = Schrodinger_1d( L = 50., x0=-1, T = 10, Nx = 2501, Nt = 10001 )


# wave-packet/potential parameters
k0=13.484
E = k0**2 / 2
V_wall = 2000
V_min = -50
#simulation initiation
u = sim.Crank_Nicolson(lambda x: gaussian_wave_packet_1d(x, lamda=8, center = 1, k0 = k0)  ,
                       Left_BC = 0., Right_BC = 0., BC_type = 'dirichlet', normalize_input = True,
                       V=lambda x: alpha_radiation_potential(x -1e-9,width = 2, Vmin = V_min, V_wall=V_wall))
x = sim.x #get x-points

import matplotlib.animation as animation

# Set up the figure and axes
plot_potential = True
save_animation = False
V = sim.V
Vmax = max(V[np.where(x>0)])
fig, (ax1, ax2, ax3) = plt.subplots(3,1,sharex= True)
fig.suptitle('Tunneling of Gaussian Wave-packet: E = '+str(round(E,2))+' , $V_{min}=$'+str(V_min)+', $V_{max} = $'+str(round(Vmax,2)))
ax1.set_xlabel('x')
ax1.set_ylabel('|ψ(x,t)|²')
ax2.set_ylabel('Re[ψ(x,t)]')
ax3.set_ylabel('Im[ψ(x,t)]')
ax1.grid()
ax2.grid()
ax3.grid()
ax1.set_ylim(-0.1,2)
ax1.set_xlim(-0.5,10)
ax2.set_ylim(-1.5,1.5)
ax3.set_ylim(-1.5,1.5)
# Set up the lines and points that will be animated
if plot_potential ==True:
    ax11=ax1.twinx()
    ax22=ax2.twinx()
    ax33=ax3.twinx()
    ax11.set_ylim(V_min - 1, 1.5*Vmax)
    ax22.set_ylim(V_min - 1, 1.5*Vmax)
    ax33.set_ylim(V_min - 1, 1.5*Vmax)
    ax11.set_ylabel("V(x)")
    ax22.set_ylabel("V(x)")
    ax33.set_ylabel("V(x)")
    ax11.plot(x,V, alpha = 1)
    ax22.plot(x,V)
    ax33.plot(x,V)
lines1 = ax1.plot(x, abs(u[:,0])**2,'b')
lines2 = ax2.plot(x, u[:,0].real ,'r')
lines3 = ax3.plot(x, u[:,0].imag ,'g')


# Set up the formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)

# Set up the function that will be called on each frame
def update(num):
    # Update the data being plotted
    lines1[0].set_data(x, abs(u[:,num]))
    lines2[0].set_data(x, u[:,num].real)
    lines3[0].set_data(x, u[:,num].imag)
    ax1.set_title('t = '+str(round(num*sim.dt,2)))
    return lines1,lines2, lines3

# Create the animation object
ani = animation.FuncAnimation(fig, update, frames=range(sim.Nt-1), interval = 1, repeat=True)

#  saving to m4 using ffmpeg writer 
if save_animation ==True:
    writervideo = animation.FFMpegWriter(fps=320) 
    ani.save('gaussian_wavepacket_1d_alpha_E='+str(round(E,2))+'Vmin='+str(V_min)+'Vmax='+str(round(Vmax,2))+'.mp4', writer=writervideo) 
