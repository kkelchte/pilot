#!/usr/bin/python
""" 
Script for launching condor jobs but then on opal.
The network in tensorflow is trained on an offline dataset
after which it is online evaluated.

All unknown arguments are passed to tensorflow.
Author: Klaas Kelchtermans
Dependencies: github.com/kkelchte/pilot in virtualenv with tensorflow GPU.
"""

import os,shutil,sys, time
import numpy as np
import subprocess, shlex

net='mobile'
for world in ['corridor', 'floor','radiator','poster','ceiling','blocked_hole','arc','doorway']:
  command="python ../pilot/main.py --log_tag {0}_{1} --dataset {0} --network {1} --max_episodes 100".format(world, net)
  subprocess.Popen(shlex.split(command))