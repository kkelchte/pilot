#!/usr/bin/python
""" 
Create separate condor jobs for each recorder of data.
If all condor jobs are ready, call a cleanup data action.
Combine different jobs in a DAG (directed acyclic graph)
All unknown arguments are passed to tensorflow.
Author: Klaas Kelchtermans
Dependencies: github.com/kkelchte/pilot in virtualenv with tensorflow GPU: condor_online and condor_offline
"""

import sys, os, os.path
import subprocess, shlex
import shutil
import time
import signal
import argparse
import yaml
import fnmatch
import numpy as np

##########################################################################################################################
# STEP 1 Load Parameters

parser = argparse.ArgumentParser(description="""Dag_train_and_evaluate submits a Directed Acyclic Graph (DAG) of condor jobs for creating training a number of policies offline and evaluating them online, printing the results with a final script when everything is finished.""")

# ==========================
#   General Settings
# ==========================
parser.add_argument("-t", "--log_tag", default='testing', type=str, help="log_tag: tag used to name logfolder.")
parser.add_argument("--number_of_models", default=10, type=int, help="Define the number of models trained simultaneously over condor.")
parser.add_argument("--summary_dir", default='tensorflow/log/', type=str, help="Choose the directory to which tensorflow should save the summaries relative to $HOME.")
parser.add_argument("--home", default='/esat/opal/kkelchte/docker_home', type=str, help="Absolute path to source of code on Opal.")


FLAGS, others = parser.parse_known_args()

# display and save all settings
print("\nSettings:")
for f in FLAGS.__dict__: print("{0}: {1}".format( f, FLAGS.__dict__[f]))
print("Others: {0}".format(others))

##########################################################################################################################
# STEP 2 For each model launch condor_offline without submitting
for model in range(FLAGS.number_of_models):
  command = "python condor_offline.py -t {0}/{1} --dont_submit --summary_dir {2}".format(FLAGS.log_tag, model, FLAGS.summary_dir)
  for e in others: command=" {0} {1}".format(command, e)
  subprocess.call(shlex.split(command)) 

##########################################################################################################################
# STEP 3 Add for each model an online condor job without submitting for evaluation/training online
for model in range(FLAGS.number_of_models):
  command="python condor_online.py -t {0}/{1}_eva --dont_submit --home {2} --summary_dir {3} --checkpoint_path {0}/{1}".format(FLAGS.log_tag, model, FLAGS.home, FLAGS.summary_dir)
  for e in others: command=" {0} {1}".format(command, e)
  subprocess.call(shlex.split(command)) 

##########################################################################################################################
# STEP 4 Call a python script that parses the results and prints some stats
command="python condor_offline.py -t {0}/results --dont_submit -pp pilot/scripts -ps get_results.py --motherdir {0} --endswith _eva --home {3}".format(FLAGS.log_tag, FLAGS.destination, FLAGS.summary_dir, FLAGS.home)
for e in others: command=" {0} {1}".format(command, e)
subprocess.call(shlex.split(command)) 


##########################################################################################################################
# STEP 5 Create DAG file that links the different jobs
dag_dir="{0}/{1}{2}/DAG".format(FLAGS.home, FLAGS.summary_dir, FLAGS.log_tag)
try:
  os.makedirs(dag_dir)
except OSError:
  print("Found existing log folder: {0}/{1}{2}".format(FLAGS.home, FLAGS.summary_dir, FLAGS.log_tag))
with open(dag_dir+"/dag_file",'w') as df:
  df.write("# File name: dag_file \n")
  for rec in range(FLAGS.number_of_recorders):
    df.write("JOB r{0} {1}/{2}{3}/{0}/condor/online.condor \n".format(rec, FLAGS.home, FLAGS.summary_dir, FLAGS.log_tag))
  df.write("JOB clean {0}/{1}{2}/clean/condor/offline.condor \n".format(FLAGS.home, FLAGS.summary_dir, FLAGS.log_tag))
  df.write("\n")
  recs=""  
  for rec in range(FLAGS.number_of_recorders): recs="{0} r{1}".format(recs, rec)
  df.write("PARENT {0} CHILD clean\n".format(recs))
  df.write("\n")
  for rec in range(FLAGS.number_of_recorders): df.write("Retry r{0} 4 \n".format(rec))
  df.write("Retry clean 4 \n")
##########################################################################################################################
# STEP 5 submit DAG file
subprocess.call(shlex.split("condor_submit_dag {0}".format(dag_dir+"/dag_file")))
print("Submission done.")
print("Monitor with: ")
print("tail -f {0}/dag_file.nodes.log".format(dag_dir))
print("or: ")
print("tail -f {0}/dag_file.dagman.out".format(dag_dir))
time.sleep(1)


