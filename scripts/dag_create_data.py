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

parser = argparse.ArgumentParser(description="""Dag_create_data submits a DAG of condor jobs for creating a dataset and cleanup the data.""")

# ==========================
#   General Settings
# ==========================
parser.add_argument("-t", "--log_tag", default='testing', type=str, help="log_tag: tag used to name logfolder.")
parser.add_argument("--number_of_recorders", default=10, type=int, help="Define the number of condor jobs that are gathering data simultaneously over condor.")
parser.add_argument("--destination", default="dag_dataset", type=str, help="Define the name of the new dataset.")
parser.add_argument("--summary_dir", default='tensorflow/log/', type=str, help="Choose the directory to which tensorflow should save the summaries relative to $HOME.")
parser.add_argument("--home", default='/esat/opal/kkelchte/docker_home', type=str, help="Absolute path to source of code on Opal.")
parser.add_argument("--wall_time_rec", default=3*60*60, type=int, help="Maximum time job is allowed to take.")
parser.add_argument("--dont_retry", action='store_true', help="Don't retry if job ends with exit code != 1 --> usefull for debugging as previous log-files are overwritten.")


FLAGS, others = parser.parse_known_args()

# display and save all settings
print("\nDAG CREATE DATA settings:")
for f in FLAGS.__dict__: print("{0}: {1}".format( f, FLAGS.__dict__[f]))
print("Others: {0}".format(others))

##########################################################################################################################
# STEP 2 For each recorder launch condor_online without submitting
for rec in range(FLAGS.number_of_recorders):
  command = "python condor_online.py -t {0}/{1} --dont_submit --summary_dir {2} --data_location {0}_{1} --wall_time {3}".format(FLAGS.log_tag, rec, FLAGS.summary_dir, FLAGS.wall_time_rec)
  for e in others: command=" {0} {1}".format(command, e)
  subprocess.call(shlex.split(command)) 

##########################################################################################################################
# STEP 3 Add condor_offline cleanup data: note that cleanup data should never take more than 3hours...
command="python condor_offline.py -t {0}/clean --dont_submit -pp ensemble_v0/scripts -ps clean_dataset.py --startswith {0} --destination {1} --home {3} --wall_time {4}".format(FLAGS.log_tag, FLAGS.destination, FLAGS.summary_dir, FLAGS.home, 3*60*60)
for e in others: command=" {0} {1}".format(command, e)
subprocess.call(shlex.split(command)) 

##########################################################################################################################
# STEP 4 Create DAG file that links the different jobs
dag_dir="{0}/{1}{2}/DAG".format(FLAGS.home, FLAGS.summary_dir, FLAGS.log_tag)
try:
  os.makedirs(dag_dir)
except OSError:
  print("Found existing log folder: {0}/{1}{2}".format(FLAGS.home, FLAGS.summary_dir, FLAGS.log_tag))
with open(dag_dir+"/dag_file_"+FLAGS.log_tag.replace('/','_'),'w') as df:
  df.write("# File name: dag_file_"+FLAGS.log_tag.replace('/','_')+" \n")
  for rec in range(FLAGS.number_of_recorders):
    df.write("JOB r{0} {1}/{2}{3}/{0}/condor/online.condor \n".format(rec, FLAGS.home, FLAGS.summary_dir, FLAGS.log_tag))
  df.write("JOB clean {0}/{1}{2}/clean/condor/offline.condor \n".format(FLAGS.home, FLAGS.summary_dir, FLAGS.log_tag))
  df.write("\n")
  recs=""  
  for rec in range(FLAGS.number_of_recorders): recs="{0} r{1}".format(recs, rec)
  df.write("PARENT {0} CHILD clean\n".format(recs))
  if not FLAGS.dont_retry:
    df.write("\n")
    for rec in range(FLAGS.number_of_recorders): df.write("Retry r{0} 4 \n".format(rec))
    df.write("Retry clean 2 \n")
##########################################################################################################################
# STEP 5 submit DAG file
subprocess.call(shlex.split("condor_submit_dag {0}".format(dag_dir+"/dag_file_"+FLAGS.log_tag.replace('/','_'))))
print("Submission done.")
print("Monitor with: ")
print("tail -f {0}/dag_file_{1}.nodes.log".format(dag_dir,FLAGS.log_tag.replace('/','_')))
print("or: ")
print("tail -f {0}/dag_file_{1}.dagman.out".format(dag_dir,FLAGS.log_tag.replace('/','_')))
time.sleep(1)


