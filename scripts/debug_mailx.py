#!/usr/bin/python

import os,shutil,sys, time
import numpy as np
import subprocess, shlex
import json

import collections

import argparse
import xml.etree.cElementTree as ET


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import tablib


print("TEST MAILX")
mailcommand="mailx -s {0} -a {1} ".format('testing', '/esat/opal/kkelchte/docker_home/tensorflow/log/offline_offpolicy/report/report.pdf')
# for f in log_folders: 
  # if os.path.isfile(f+'/log.xls'): mailcommand+=" -a {0}/log.xls".format(f)
p_msg = subprocess.Popen(shlex.split("echo {0} : {1} is finished.".format(time.strftime("%Y-%m-%d_%I:%M:%S"), '/esat/opal/kkelchte/docker_home/tensorflow/log/offline_offpolicy')), stdout=subprocess.PIPE)
p_mail = subprocess.Popen(shlex.split(mailcommand+" klaas.kelchtermans@esat.kuleuven.be"),stdin=p_msg.stdout, stdout=subprocess.PIPE)
print(p_mail.communicate())


