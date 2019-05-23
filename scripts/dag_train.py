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

def add_other_arguments(command, skiplist, skipnextlist, others):
  """Append arguments from 'others' to command if they are not in skiplist.
  Skip also next argument if the argument is in skipnextlist.
  return augmented command.
  """
  break_next = False
  for e in others: 
    if break_next: # don't add another --checkpoint_path in case this was set
      break_next = False 
    elif e in skipnextlist:
      break_next = True
    elif e in skiplist:
      pass
    else:
      command="{0} {1}".format(command, e)
  return command

##########################################################################################################################
# STEP 1 Load Parameters

parser = argparse.ArgumentParser(description="""Train models over different learning rates and send result as pdf.""")

# ==========================
#   General Settings
# ==========================
parser.add_argument("-t", "--log_tag", default='testing', type=str, help="log_tag: tag used to name logfolder.")
# parser.add_argument("--number_of_models", default=10, type=int, help="Define the number of models trained simultaneously over condor.")
parser.add_argument("--summary_dir", default='tensorflow/log/', type=str, help="Choose the directory to which tensorflow should save the summaries relative to $HOME.")
parser.add_argument("--home", default='/esat/opal/kkelchte/docker_home', type=str, help="Absolute path to source of code on Opal.")
parser.add_argument("--wall_time", default=3*60*60, type=int, help="Maximum time job is allowed to train.")
# parser.add_argument("--wall_time_eva", default=3*60*60, type=int, help="Maximum time job is allowed to evaluate.")
parser.add_argument("--dont_retry", action='store_true', help="Don't retry if job ends with exit code != 1 --> usefull for debugging as previous log-files are overwritten.")
parser.add_argument('--learning_rates', default=[0.1,0.01,0.001,0.0001,0.00001],nargs='+', help="Seeds to use over different models.")

FLAGS, others = parser.parse_known_args()

# display and save all settings
print("\nDAG_TRAIN settings:")
for f in FLAGS.__dict__: print("{0}: {1}".format( f, FLAGS.__dict__[f]))
print("Others: {0}".format(others))

models=[str(lr).replace('.','') for lr in FLAGS.learning_rates]
##########################################################################################################################
# STEP 2 For each model launch condor_offline without submitting
for modelindex, model in enumerate(models):
  command = "python condor_offline.py -t {0}/lr_{1} --dont_submit --summary_dir {2} --wall_time {3} --random_seed {4} --learning_rate {5}".format(FLAGS.log_tag, model, FLAGS.summary_dir, FLAGS.wall_time, 123, FLAGS.learning_rates[modelindex])
  command = add_other_arguments(command, [], ['--learning_rate'], others)
  save_call(command)

##########################################################################################################################
# STEP 3 Call a python script that creates a report
# command="python condor_offline.py -t {0}/report --dont_submit -pp pytorch_pilot/scripts -ps save_results_as_pdf.py --mother_dir {0} --home {1} --wall_time {2} --summary_dir {3}".format(FLAGS.log_tag, FLAGS.home, 10*60, FLAGS.summary_dir)
command="python condor_offline.py -t {0}/report --dont_submit --rammem 3 --gpumem 0 -pp pytorch_pilot/scripts -ps parse_results_to_pdf.py --mother_dir {0} --home {1} --wall_time {2}".format(FLAGS.log_tag, FLAGS.home, 5*60)
command=add_other_arguments(command, 
                            skiplist=[],
                            skipnextlist=['-pp','--python_project','--gpumem','--rammem', '-ps','--mother_dir','--home','--wall_time','--endswith','--copy_dataset'],
                            others)
save_call(command)


##########################################################################################################################
# STEP 5 Create DAG file that links the different jobs
dag_dir="{0}/{1}{2}/DAG".format(FLAGS.home, FLAGS.summary_dir, FLAGS.log_tag)
try:
  os.makedirs(dag_dir)
except OSError:
  print("Found existing log folder: {0}/{1}{2}".format(FLAGS.home, FLAGS.summary_dir, FLAGS.log_tag))
with open(dag_dir+"/dag_file_"+FLAGS.log_tag.replace('/','_'),'w') as df:
  df.write("# File name: dag_file_"+FLAGS.log_tag+" \n")
  for model in models:
    df.write("JOB m{0}_train {1}/{2}{3}/lr_{0}/condor/offline.condor \n".format(model, FLAGS.home, FLAGS.summary_dir, FLAGS.log_tag))
  df.write("JOB report {1}/{2}{3}/report/condor/offline.condor \n".format('', FLAGS.home, FLAGS.summary_dir, FLAGS.log_tag))
  df.write("\n")
  train_jobs=""
  for model in models: train_jobs="{0} m{1}_train".format(train_jobs, model)
  df.write("PARENT {0} CHILD report\n".format(train_jobs))
  # df.write("PARENT results CHILD report\n")
  df.write("\n")
  if not FLAGS.dont_retry:
    for model in models: 
      df.write("Retry m{0}_train 2 \n".format(model))
    df.write("Retry report 3 \n")

##########################################################################################################################
# STEP 6 submit DAG file
save_call("condor_submit_dag {0}".format(dag_dir+"/dag_file_"+FLAGS.log_tag.replace('/','_')))

print("Submission done.")
print("Monitor with: ")
print("tail -f {0}/dag_file_{1}.nodes.log".format(dag_dir, FLAGS.log_tag.replace('/','_')))
print("or: ")
print("tail -f {0}/dag_file_{1}.dagman.out".format(dag_dir, FLAGS.log_tag.replace('/','_')))
time.sleep(1)


