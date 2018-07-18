#!/usr/bin/env python

import numpy
import matplotlib
matplotlib.use("Agg")
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

n = 500
q = [0.5]
p = [0.1]
t= 65.24

files = []

for i in range(n):
    p_ = numpy.random.normal(0,numpy.sqrt(m))
    qp = qpt(t, q[-1], p_)
    q.append(qp[0])
    p.append(qp[1])

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro', animated=True,ms=2)

def init():
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_xlabel('q')
    ax.set_ylabel('p')
    return ln,

def update(frame):
    ln.set_data(q[0:frame], p[0:frame])
    return ln,

ani = FuncAnimation(fig, update, frames=n,
                    init_func=init, blit=True)

# Set up formatting for the movie files
Writer = matplotlib.animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

ani.save("temp.mp4",writer=writer)

plt.clf()
q=numpy.array(q)

plt.hist(q,density=True,bins=50)
x = numpy.arange(-3.5,3.5,0.001)
plt.plot(x,numpy.exp(-U(x))/numpy.sqrt(2*numpy.pi))
plt.xlabel('q')
plt.savefig("temp.pdf")