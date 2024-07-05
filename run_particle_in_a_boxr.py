
from utilities import gaussian_1d_norm, square_1d, gaussian_wave_packet_1d
from schrodinger import Schrodinger_1d
import matplotlib.pyplot as plt
import numpy as np

sim = Schrodinger_1d( L = 1.0, x0= 0, T = 10, Nx = 1001, Nt = 8001 )

u = sim.Crank_Nicolson(lambda x:np.sin(np.pi*x) +  np.sin(2 * np.pi * x) +np.sin(3 * np.pi * x) ,Left_BC = 0., Right_BC = 0., BC_type = 'dirichlet', normalize_input = True)
x = sim.x




import matplotlib.animation as animation


# Set up the figure and axes
fig, (ax1, ax2, ax3) = plt.subplots(3,1,sharex= True)
fig.suptitle(r'Time evolution of $\psi= \frac{\psi_1+\psi_2+\psi_3}{\sqrt{3}}$ ')
ax1.set_xlabel('x')
ax1.set_ylabel('|ψ(x,t)|²')
ax2.set_ylabel('Re[Ψ(x,t)]')
ax3.set_ylabel('Im[Ψ(x,t)]')
ax1.grid()
ax2.grid()
ax3.grid()
ax3.set_ylim(-2,2)
ax2.set_ylim(-2,2)
#ax1.set_xlim(0,50)
# Set up the lines and points that will be animated
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
    return lines1,lines2,lines3

# Create the animation object
ani = animation.FuncAnimation(fig, update, frames=range(sim.Nt-1), interval = 1, repeat=True)

#  saving to m4 using ffmpeg writer 
writervideo = animation.FFMpegWriter(fps=320) 
ani.save('particle_in_a_box_psi1+psi2+psi3.mp4', writer=writervideo) 