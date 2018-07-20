#!/usr/bin/env python

import numpy
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

a = 1.
k = 1.
m=1.
w = numpy.sqrt(k/m)

def U(q):
    return 0.5* k*q**2

def qpt(t, q_0, p_0):
    A = q_0
    B = p_0/w
    return A*numpy.cos(w*t)+B*numpy.sin(w*t),-w*A*numpy.sin(w*t)+w*B*numpy.cos(w*t)

n = 10000
q = [2.4]
q2 = [1]
p = [0.1]
p2 =[0.5]
p_prop=[]
p2_prop=[]
t= 4.24

files = []
numpy.random.seed(1)

for i in range(n):
    p_ = numpy.random.normal(0,numpy.sqrt(m))
    qp = qpt(t, q[-1], p_)
    q.append(qp[0])
    p.append(qp[1])
    p_prop.append(p_)
    p2_ = numpy.random.normal(0,numpy.sqrt(m))
    qp = qpt(t, q2[-1], p2_)
    q2.append(qp[0])
    p2.append(qp[1])
    p2_prop.append(p2_)

def sideplot(index):
    plt.scatter(q[index],q2[index])
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.xlabel('q_x')
    plt.ylabel('q_y')
    plt.title('Step {}'.format(index))
    plt.savefig("{}_0.pdf".format(index))
    plt.arrow(q[index],q2[index],p_prop[index]/1.75,p2_prop[index]/1.75,head_width=.2)
    plt.savefig("{}_1.pdf".format(index))
    dum = numpy.arange(0,t,0.1)
    qpath = qpt(dum,q[index],p_prop[index])
    q2path = qpt(dum,q2[index],p2_prop[index])
    plt.scatter(qpath[0],q2path[0],s=2)
    plt.scatter(q[1],q2[1])
    plt.savefig("{}_2.pdf".format(index))

sideplot(0)
sideplot(1)
sideplot(2)

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro', animated=True,ms=2)

def init():
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_xlabel('q_x')
    ax.set_ylabel('q_y')
    return ln,

nframes=500

def update(frame):
    if frame < nframes-1:
        ln.set_data(q[frame], q2[frame])
    else:
        ln.set_data(q, q2)
    return ln,

ani = FuncAnimation(fig, update, frames=nframes,
                    init_func=init, blit=True)

# Set up formatting for the movie files
Writer = matplotlib.animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

ani.save("temp.mp4",writer=writer)

from chainconsumer import ChainConsumer
plt.clf()
c = ChainConsumer()

data = numpy.array([q,q2]).T
c.add_chain(data, parameters=["$q_x$", "$q_y$"])
c.configure(plot_hists=False)
c.plotter.plot(filename="example.png", figsize="column")


# q=numpy.array(q)

# plt.hist(q,density=True,bins=50)
# x = numpy.arange(-3.5,3.5,0.001)
# plt.plot(x,numpy.exp(-U(x))/numpy.sqrt(2*numpy.pi))
# plt.xlabel('q')
# plt.savefig("temp.pdf")