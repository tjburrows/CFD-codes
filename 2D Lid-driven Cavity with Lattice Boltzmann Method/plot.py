import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
Re = 100;
N = 100;
M = 300;

data = {};
for scalar in ['x','y','u','v','rho']:
    file = open("Re=%d_N=%d_M=%d_%s.bin" % (Re, N, M, scalar), "rb")
    array = np.fromfile(file, dtype='float', count=-1, sep="")
    data[scalar] = np.reshape(array, (M,N))

x = data['x']
y = data['y']
u = data['u']
v = data['v']
rho = data['rho']
vmag = np.sqrt(u ** 2 + v ** 2)

fig0, ax0 = plt.subplots(dpi=250)
strm  = ax0.streamplot(x[0,:], y[:,0], u, v, density = 4, color = vmag, arrowstyle = '-', minlength = 0.1, linewidth = 1, transform = None)
clb = fig0.colorbar(strm.lines)
clb.ax.set_title('|u|')
plt.axis("scaled")
plt.title("Streamlines, N=%d, M=%d, Re=%d" % (N, M, Re))
