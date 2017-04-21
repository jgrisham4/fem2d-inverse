#!/usr/bin/env python

"""
The purpose of this script is to find the solution to the 
inverse problem using different methods to compute sensitivity
and different step sizes.

Author: James Grisham
Date  : 11/18/2016
"""

import os
import sys
from subprocess import call
import numpy as np
import shutil

# Function for running a job
def run_job(dr_value, exe_path, exe_sam, exe_sacvm, flux_data):
    
    # Making directory for the particular dr value
    dir_name = "dr{:02.0f}".format(np.log10(dr_value))
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)
    os.mkdir(dir_name)
    os.chdir(dir_name)

    # Linking necessary executables
    os.symlink("{}{}".format(exe_path, exe_sam), exe_sam)
    os.symlink("{}{}".format(exe_path, exe_sacvm), exe_sacvm)
    os.symlink("{}{}".format(exe_path, flux_data), flux_data)

    # Writing dr to file
    with open("step_size.inp", "w") as ssf:
        ssf.write("{:1.6e}".format(dr_value))

    # Calling executables
    call("./{} &> {}.out".format(exe_sam, exe_sam), shell=True)
    call("./{} &> {}.out".format(exe_sacvm, exe_sacvm), shell=True)

    # Changing directories up one level
    os.chdir("..")

    return True

# Inputs
#dr = np.logspace(-12, -6, 4)  # dr
#dr = np.logspace(-17, -13, 5)  # dr
dr = [1.0e-11]
exe_dir      = "/share/codes/fem2d-inverse/src/"
target_sacvm = "inverse_problem_sacvm"
target_sam   = "inverse_problem_sam"
target_flux  = "qn_target.dat"


# Running jobs for sacvm
for drv in dr:
    run_job(drv, exe_dir, target_sacvm, target_sam, target_flux)


    
