#!/usr/bin/env python

import numpy
import matplotlib.pyplot as plt

a = 1.
k = 1.
m=1.
w = numpy.sqrt(k/m)

def U(q):
    return 0.5* k*q**2

def qpt(t, q_0, p_0):
    A = q_0
    B = p_0/w
    return A*numpy.cos(w*t)+B*numpy.sin(w*t),0

n = 10000
q = [0.5]
p = [0.1]
t= 65.24

for i in xrange(n):
    p_ = numpy.random.normal(0,numpy.sqrt(m))
    qp = qpt(t, q[-1], p_)
    q.append(qp[0])
    p.append(qp[1])

q=numpy.array(q)

plt.hist(q,normed=True,bins=100)
x = numpy.arange(-3.5,3.5,0.001)
plt.plot(x,numpy.exp(-U(x))/numpy.sqrt(2*numpy.pi))
plt.show()