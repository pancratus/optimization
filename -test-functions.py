
# coding: utf-8

# # Test functions for single-objective optimization

# In[1]:

import matplotlib.pyplot as plt
from numpy import *
from pylab import *
get_ipython().magic('matplotlib notebook')

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import inferno as colormap
from matplotlib.colors import LogNorm, Normalize
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

'''
3D plot
'''
def plot3D(func, x_range, log_norm=False, interaction=False, s=0):
    fig = plt.figure()
    fig.clf()
    ax = Axes3D(fig, azim=-128.0, elev=43.0)
    
    if s==0:
        s = (amax(x_range)-amin(x_range))/100
    x1 = arange(x_range[0][0], x_range[0][1] + s, s)
    x2 = arange(x_range[1][0], x_range[1][1] + s, s)
    x = meshgrid(x1, x2)
    f = func(x)
    
    if log_norm:
        norm = LogNorm()
    else:
        norm = Normalize()
    ax.plot_surface(x[0], x[1], f, rstride=1, cstride=1, norm=norm,
                    cmap=colormap, linewidth=0, edgecolor='none', alpha=1)

    ax.set_xlim([min(x1), max(x1)])
    ax.set_ylim([min(x2), max(x2)])
    ax.set_zlim([amin(f), amax(f)])

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.show()


# ## Rosenbrock function
rosen = lambda x: sum([100*(x[i+1]-x[i]**2)**2 + (x[i]-1)**2 for i in range(shape(x)[0]-1)], axis=0)
rosen_range = [(-2,2), (-1,3)]

# ## Sphere function
sphere = lambda x: sum([v**2 for v in x], axis=0)
sphere_range = [(-2,2), (-2,2)]

# ## McCormick function
mccormick = lambda x: sin(x[0]+x[1]) + (x[0]-x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1
mccormick_range = [(-2,4), (-4,4)]

# ## Styblinski–Tang function
tang = lambda x: sum([v**4 -16*v**2 + 5*v for v in x], axis=0)/2
tang_range = [(-5,5), (-5,5)]

# ## Three-hump camel function
three_hump = lambda x: 2*x[0]**2 -1.05*x[0]**4 + x[0]**6/6 + x[0]*x[1] + x[1]**2
three_hump_range = [(-5,5), (-5,5)]

# ## Lévi function N.13
sqn = lambda x: sin(x*pi)**2
levi13 = lambda x: sqn(3*x[0]) + (1+sqn(3*x[1]))*(x[0]-1)**2 + (1+sqn(2*x[1]))*(x[1]-1)**2
levi13_range = [(-10,10), (-10,10)]

# ## Beale's function
beale = lambda x: (1.5-x[0]+x[0]*x[1])**2 + (2.25-x[0]+x[0]*x[1]**2)**2 + (2.625-x[0]+x[0]*x[1]**3)**2
beale_range = [(-4.5,4.5), (-4.5,4.5)]

# ## Beale's function
goldstein = lambda x: vectorize(int)(1+(x[0]+x[1]+1)**2*(19-14*x[0]+3*x[0]**2-14*x[1]+6*x[0]*x[1]+3*x[1]**2))*\
vectorize(int)(30+(2*x[0]-3*x[1])**2*(18-32*x[0]+12*x[0]**2+48*x[1]-36*x[0]*x[1]+27*x[1]**2))
goldstein_range = [(-2,2), (-2,2)]

# ## Bukin function N.6
bukin = lambda x: 100*sqrt(abs(x[1]-0.01*x[0]**2))+0.01*abs(x[0]+10)
bukin_range = [(-15,-5), (-3,3)]

# ## Matyas function
matyas = lambda x: 0.26*(x[0]**2+x[1]**2)-0.48*x[0]*x[1]
matyas_range = [(-10,10), (-10,10)]

# ## Booth's function
booth = lambda x: (x[0]+2*x[1]+7)**2 + (2*x[0]+x[1]+5)**2
booth_range = [(-10,10), (-10,10)]

# ## Eggholder function
eggholder = lambda x: 1000-x[1]*sin(absolute(x[0]/2+x[1])**0.5)-x[0]*sin(absolute(x[0]-x[1])**0.5)
eggholder_range = [(-512,512), (-465,559)]

# ## Easom function
easom = lambda x: -cos(x[0])*cos(x[1])*exp(-(x[0]-pi)**2-(x[1]-pi)**2)
easom_range = [(-5,10), (-5,10)]

# ## Cross-in-tray function
cross = lambda x: -0.0001*(abs(sin(x[0])*sin(x[1])*exp(abs(100-sqrt(x[0]**2+x[1]**2)/pi)))+1)**0.1
cross_range = [(-10,10), (-10,10)]

# ## Cross-in-tray function N.2
rcross = lambda x: 0.0001*(abs(sin(x[0])*sin(x[1])*exp(abs(100-sqrt(x[0]**2+x[1]**2)/pi)))+1)**0.1
rcross_range = [(-10,10), (-10,10)]

# ## Ackley's function
ackley = lambda x: -20*exp(-0.2*sqrt(x[0]**2+x[1]**2))-exp(cos(2*pi*x[0])/2+cos(2*pi*x[1])/2) + exp(1) + 20
ackley_range = [(-5,5), (-5,5)]

# ## Rastrigin function
rastrigin = lambda x,A=10: A*len(x)+sum([v**2-A*cos(2*pi*v) for v in x],axis=0)
rastrigin_range = [(-5.12,5.12), (-5.12,5.12)]
