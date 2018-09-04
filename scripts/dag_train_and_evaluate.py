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


def save_call(command):
  """Start command in subprocess.
  In case exit code is non-zero, exit with same exit code. 
  """
  ex_code=subprocess.call(shlex.split(command))
  if ex_code != 0: sys.exit(ex_code)

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
parser.add_argument("--wall_time_train", default=3*60*60, type=int, help="Maximum time job is allowed to train.")
parser.add_argument("--wall_time_eva", default=3*60*60, type=int, help="Maximum time job is allowed to evaluate.")
parser.add_argument("--dont_retry", action='store_true', help="Don't retry if job ends with exit code != 1 --> usefull for debugging as previous log-files are overwritten.")


FLAGS, others = parser.parse_known_args()

# display and save all settings
print("\nDAG_TRAIN_AND_EVALUATE settings:")
for f in FLAGS.__dict__: print("{0}: {1}".format( f, FLAGS.__dict__[f]))
print("Others: {0}".format(others))

##########################################################################################################################
# STEP 2 For each model launch condor_offline without submitting
for model in range(FLAGS.number_of_models):
  command = "python condor_offline.py -t {0}/{1} --dont_submit --summary_dir {2} --wall_time {3}".format(FLAGS.log_tag, model, FLAGS.summary_dir, FLAGS.wall_time_train)
  for e in others: command=" {0} {1}".format(command, e)
  save_call(command)

##########################################################################################################################
# STEP 3 Add for each model an online condor job without submitting for evaluation/training online
for model in range(FLAGS.number_of_models):
  command="python condor_online.py -t {0}/{1}_eva --dont_submit --home {2} --summary_dir {3} --checkpoint_path {0}/{1} --wall_time {4}".format(FLAGS.log_tag, model, FLAGS.home, FLAGS.summary_dir, FLAGS.wall_time_eva)
  for e in others: command=" {0} {1}".format(command, e)
  save_call(command)

##########################################################################################################################
# STEP 4 Call a python script that parses the results and prints some stats
command="python condor_offline.py -t {0}/results --dont_submit -pp pilot/scripts -ps get_results.py --mother_dir {0} --endswith _eva --home {1} --wall_time {2}".format(FLAGS.log_tag, FLAGS.home, 10*60)
for e in others: command=" {0} {1}".format(command, e)
save_call(command)

##########################################################################################################################
# STEP 5 Call a python script that creates a report
command="python condor_offline.py -t {0}/report --dont_submit -pp pilot/scripts -ps save_results_as_pdf.py --mother_dir {0} --home {1} --wall_time {2} --summary_dir {3}".format(FLAGS.log_tag, FLAGS.home, 10*60, FLAGS.summary_dir)
for e in others: command=" {0} {1}".format(command, e)
save_call(command)


##########################################################################################################################
# STEP 5 Create DAG file that links the different jobs
dag_dir="{0}/{1}{2}/DAG".format(FLAGS.home, FLAGS.summary_dir, FLAGS.log_tag)
try:
  os.makedirs(dag_dir)
except OSError:
  print("Found existing log folder: {0}/{1}{2}".format(FLAGS.home, FLAGS.summary_dir, FLAGS.log_tag))
with open(dag_dir+"/dag_file_"+FLAGS.log_tag,'w') as df:
  df.write("# File name: dag_file_"+FLAGS.log_tag+" \n")
  for model in range(FLAGS.number_of_models):
    df.write("JOB m{0}_train {1}/{2}{3}/{0}/condor/offline.condor \n".format(model, FLAGS.home, FLAGS.summary_dir, FLAGS.log_tag))
    df.write("JOB m{0}_eva {1}/{2}{3}/{0}_eva/condor/online.condor \n".format(model, FLAGS.home, FLAGS.summary_dir, FLAGS.log_tag))
  df.write("JOB results {1}/{2}{3}/results/condor/offline.condor \n".format('', FLAGS.home, FLAGS.summary_dir, FLAGS.log_tag))
  df.write("JOB report {1}/{2}{3}/report/condor/offline.condor \n".format('', FLAGS.home, FLAGS.summary_dir, FLAGS.log_tag))
  df.write("\n")
  for model in range(FLAGS.number_of_models):
    df.write("PARENT m{0}_train CHILD m{0}_eva\n".format(model))
  eva_jobs=""  
  for model in range(FLAGS.number_of_models): eva_jobs="{0} m{1}_eva".format(eva_jobs, model)
  df.write("PARENT {0} CHILD results\n".format(eva_jobs))
  df.write("PARENT results CHILD report\n")
  df.write("\n")
  if not FLAGS.dont_retry:
    for model in range(FLAGS.number_of_models): 
      df.write("Retry m{0}_train 2 \n".format(model))
      df.write("Retry m{0}_eva 3 \n".format(model))
    df.write("Retry results 2 \n")
    df.write("Retry report 3 \n")

##########################################################################################################################
# STEP 6 submit DAG file
save_call("condor_submit_dag {0}".format(dag_dir+"/dag_file_"+FLAGS.log_tag))

print("Submission done.")
print("Monitor with: ")
print("tail -f {0}/dag_file_{1}.nodes.log".format(dag_dir, FLAGS.log_tag))
print("or: ")
print("tail -f {0}/dag_file_{1}.dagman.out".format(dag_dir, FLAGS.log_tag))
time.sleep(1)


