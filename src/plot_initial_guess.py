#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

d = np.genfromtxt("initial_guess")
plt.plot(d[:,0], d[:,1], "-ob", lw=1.5)

d2 = np.genfromtxt("target_geometry")
plt.plot(d2[:,0],d2[:,1],"-r", lw=1.5);

plt.axis("square")

plt.show()
