#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pylab as plb
params = {"legend.fontsize": 13}
plb.rcParams.update(params)

# Importing data
d = np.genfromtxt("initial_guess")
d2 = np.genfromtxt("target_geometry")

# Plotting
w = 4.5
h = w
my_ms = 4
fig = plt.figure(figsize=(w, h))
plt.plot(d2[:,0]/5.0, d2[:,1]/5.0,"-ok", lw=1.5, ms=my_ms, label="target");
plt.plot(d[:,0]/5.0, d[:,1]/5.0, "-ok", lw=1.5, color="gray", ms=my_ms, label="initial")
plt.plot(np.array([d[-1, 0], d[0, 0]])/5.0, np.array([d[-1, 1], d[0, 1]])/5.0, "-ok", lw=1.5, color="gray", ms=my_ms)
plt.plot(np.array([d2[-1, 0], d2[0, 0]])/5.0, np.array([d2[-1, 1], d2[0, 1]])/5.0, "-ok", lw=1.5, ms=my_ms)

# Adding labels
plt.xlabel(r"$x/\bar{r}_i$", fontsize=14)
plt.ylabel(r"$y/\bar{r}_i$", fontsize=14)
plt.axis("square")
plt.legend(bbox_to_anchor=(0.0,1.0), loc=2, borderaxespad=0.0)
plt.tight_layout()

# Saving figure
save_path = "/home/james/Documents/mypapers/journal/sacvm-asme-ht/images/"
fig.savefig("{}initial_guess.pdf".format(save_path))
