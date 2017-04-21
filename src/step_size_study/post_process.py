#!/usr/bin/env python

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

class opt_data:
    """
    This class represents optimization data.  It has a method for importing
    data from a file, which is called inside the constructor.
    """

    def __init__(self, file_path, dr):
        self.dr = dr
        self.file_path = file_path
        self.ndvs = None
        self.niter = None
        self.f = None
        self.x = None
        self.dfdx = None
        self.has_nans = False
        self.data_imported = self.import_data()

    def import_data(self):

        # Making sure file exists
        if not os.path.isfile(self.file_path):
            print("\nWarning: Can't open {}.\n".format(self.file_path))
            return False
        else:
            print("Importing data from {}".format(self.file_path))

        # Importing data
        with open(self.file_path) as df:
            lines = df.readlines()[1:]

        # Getting data from header
        self.ndvs = int(lines[0].strip().split()[-1])
        self.niter = int(lines[1].strip().split()[-1])

        # Reading optimization data
        self.f = []
        self.x = []
        self.dfdx = []
        for i in range(0, self.niter-1):
            sline = lines[i+2].strip().split()
            self.f.append(float(sline[0]))
            if math.isnan(self.f[-1]):
                self.has_nans = True
                break
            self.x.append(list(map(lambda item : float(item), sline[1:self.ndvs+1])))
            self.dfdx.append(list(map(lambda item : float(item), sline[self.ndvs+1:])))

        # Appending the last line of data
        if not self.has_nans:
            sline = lines[-1].strip().split()
            self.f.append(float(sline[0]))
            self.x.append(list(map(lambda item : float(item), sline[1:])))

        return True

# Inputs of directories to use
dirs = sorted(list(filter(lambda item : os.path.isdir(item) and item.startswith("dr"), os.listdir(os.getcwd()))))

# Importing data
sam_fname = "opt_sam.dat"
sacvm_fname = "opt_sacvm.dat"
opt_sam = []
opt_sacvm = []
for d in dirs:

    # Figuring out step size from directory name
    exponent = float(d[-2:])
    dr = 1.0*10**(-exponent)

    # Importing data
    opt_sam.append(opt_data("{}/{}".format(d, sam_fname), dr))
    opt_sacvm.append(opt_data("{}/{}".format(d, sacvm_fname), dr))

# Creating 3D plot for SAM data
fig, ax = plt.subplots(3, 3, sharex="col", sharey="row", figsize=(9.5, 7.5))
zsam = list(map(lambda item : item.f, opt_sam))
zcvm = list(map(lambda item : item.f, opt_sacvm))
alphabet = [r"$({})$".format(l) for l in ["a", "b", "c", "d", "e", "f", "g", "h", "i"]]
for j in range(0, 3):
    for i in range(0, 3):
        xcvm = np.arange(1, opt_sacvm[j*3+i].niter+1)
        xsam = np.arange(1, opt_sam[j*3+i].niter+1)
        ax[j,i].plot(xcvm, zcvm[j*3+i], "-", color="gray", lw=1.5)
        if len(xsam) == len(zsam[i]) and opt_sam[j*3+i].data_imported:
            ax[j,i].plot(xsam, zsam[j*3+i], "-b", lw=1.5)
        elif opt_sam[j*3+i].data_imported:
            xsam = np.arange(1, len(zsam[j*3+i])+1)
            ax[j,i].plot(xsam[:-2], zsam[j*3+i][:-2], "-b", lw=1.5)
        ax[j,i].set_ylim([5e-5, 1e0])
        ax[j,i].set_yscale("log")
        ax[j,i].set_title(r"$\Delta x = 1\times10^{"+"{:1.0f}".format(np.log10(opt_sam[j*3+i].dr))+"}$")
        ax[j,i].text(57, 2.5e-1, alphabet[j*3+i], fontsize=13)

for i in range(0, 3):
    ax[i, 0].set_ylabel(r"$f(\mathbf{x})$", fontsize=13)
    ax[-1, i].set_xlabel("Iteration")
    ax[-1, i].set_xlim([0, 70])

save_path = "/home/james/Documents/mypapers/journal/sacvm-asme-ht/images"
fig.savefig("{}/results_sam.pdf".format(save_path))
