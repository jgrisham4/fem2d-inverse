#!/usr/bin/env python

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Inputs
r_o = 10.0
r_i = 5.0
T_i = 373.0
T_o = 283.0

# Checking to make sure data file exists
if not os.path.isfile("slice.dat"):
    print("\nslice.dat doesn't exist.")
    print("Exiting.\n")
    sys.exit()

# Exact solution
r = np.linspace(r_i, r_o, 100)
T_e = (T_i - T_o)/(r_i - r_o)*r + (T_o*r_i - T_i*r_o)/(r_i - r_o)

# Importing and plotting data
data = np.genfromtxt("slice.dat")
fig1 = plt.figure()
plt.plot(r, T_e, "-k", lw=1.5, label="Exact")
plt.plot(data[:, 0], data[:, 1], "--r", lw=1.5, label="Numerical")
plt.xlabel("r")
plt.ylabel("T, deg K")
plt.grid()
plt.legend()
plt.show()
