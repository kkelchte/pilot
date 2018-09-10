#!/usr/bin/python
""" 
Script for launching condor jobs that are trained in a virtual env defined in my home dir on asgard.
The network in tensorflow is trained on an offline dataset rather than in a singularity environment like condor_online.py.

All unknown arguments are passed to tensorflow.
Author: Klaas Kelchtermans
Dependencies: github.com/kkelchte/pilot in virtualenv with tensorflow GPU.
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

parser = argparse.ArgumentParser(description="""Condor_offline submits a condor job for training a DNN from an offline dataset.""")

# ==========================
#   General Settings
# ==========================
parser.add_argument("--summary_dir", default='tensorflow/log/', type=str, help="Choose the directory to which tensorflow should save the summaries relative to $HOME.")
parser.add_argument("--data_root", default='pilot_data/', type=str, help="Choose the directory to which tensorflow should save the summaries relative to $HOME.")
# parser.add_argument("--code_root", default='~', type=str, help="Choose the directory to which tensorflow should save the summaries.")
parser.add_argument("--home", default='/esat/opal/kkelchte/docker_home', type=str, help="Absolute path to location with tensorflow code and data root.")
parser.add_argument("-t", "--log_tag", default='testing', type=str, help="LOGTAG: tag used to name logfolder.")
parser.add_argument("--dont_submit",action='store_true', help="In case you dont want to submit the job.")
parser.add_argument("--copy_dataset",action='store_true', help="Copy the dataset to the tmp folder in case it cant be loaded in ram.")
parser.add_argument("--dont_summarize_locally",action='store_true', help="Keep tensorflow/log on Opal.")

# ==========================
#   Tensorflow Settings
# ==========================
parser.add_argument("-pp","--python_project",default='ensemble_v0/pilot', type=str, help="Define in which python project the executable should be started with ~/tenorflow/PROJECT_NAME/main.py: q-learning/pilot, ensemble_v0/pilot, ddpg, ....")
parser.add_argument("-ps","--python_script",default='main.py', type=str, help="Define which python module should be started within the project: e.g. main.py or data.py.")

#===========================
#   Condor Machine Settings
#===========================
parser.add_argument("--gpumem",default=1900, type=int,help="define the number of mbs required in your GPU.")
parser.add_argument("--rammem",default=15, type=int,help="define the number of gigs required in your RAM.")
parser.add_argument("--diskmem",default=50, type=int,help="define the number of gigs required on your HD.")
parser.add_argument("--evaluate_after",action='store_true', help="After training a new condor job can be submitted to evaluate the model after.")
parser.add_argument("--wall_time",default=60*60*3, help="After training a new condor job can be submitted to evaluate the model after.")
parser.add_argument("--not_nice",action='store_true', help="In case you want higher priority.")

#===========================
#   Evaluation Params --> hard coded bellow: -n 20 -w canyon --reuse_default --fsm nn_turtle_fsm -p eva_params.yaml
#===========================

FLAGS, others = parser.parse_known_args()

# 4 main directories have to be defined in order to make it also runnable from a read-only system-installed singularity image.
# if FLAGS.summary_dir[0] != '/':  # 1. Tensorflow log directory for saving tensorflow logs and xterm logs
#   FLAGS.summary_dir=FLAGS.home+'/'+FLAGS.summary_dir
# if FLAGS.data_root[0] != '/':  # 2. Pilot_data directory for saving data
#   FLAGS.data_root=FLAGS.home+'/'+FLAGS.data_root
# if FLAGS.code_root == '~': # 3. location for tensorflow code (and also catkin workspace though they are found with rospack)
#   FLAGS.code_root = FLAGS.home



# display and save all settings
print("\nCONDOR OFFLINE settings:")
for f in FLAGS.__dict__: print("{0}: {1}".format( f, FLAGS.__dict__[f]))
print("Others: {0}".format(others))

##########################################################################################################################
# STEP 2 Create description and logging directories

description="{0}_{1}".format(FLAGS.log_tag.replace('/','_'),time.strftime("%Y-%m-%d_%I-%M-%S"))

print description
condor_output_dir=FLAGS.home+'/'+FLAGS.summary_dir+FLAGS.log_tag+"/condor"
if os.path.isfile("{0}/offline_{1}.log".format(condor_output_dir,description)):
    os.remove("{0}/offline_{1}.log".format(condor_output_dir,description))
    os.remove("{0}/offline_{1}.out".format(condor_output_dir,description))
    os.remove("{0}/offline_{1}.err".format(condor_output_dir,description))

temp_dir=condor_output_dir+"/.tmp"
if os.path.isdir(temp_dir):	shutil.rmtree(temp_dir)

condor_file="{0}/offline.condor".format(condor_output_dir)
# condor_file="{0}/offline_{1}.condor".format(condor_output_dir,description)
shell_file="{0}/run_{1}.sh".format(temp_dir,description)

try:
	os.makedirs(condor_output_dir)
except: pass
try:
	os.makedirs(temp_dir)
except: pass
##########################################################################################################################
# STEP 3 Create condor file 

condor_submit = open(condor_file,'w')

condor_submit.write("Universe         = vanilla\n")
condor_submit.write("RequestCpus      = 4 \n")
condor_submit.write("Request_GPUs     = 1 \n")
condor_submit.write("RequestMemory    = {0}G \n".format(FLAGS.rammem))
condor_submit.write("RequestDisk      = {0}G \n".format(FLAGS.diskmem))

condor_submit.write("match_list_length = 4 \n")

blacklist=" && (machine != \"andromeda.esat.kuleuven.be\") \
		 	&& (machine != \"kochab.esat.kuleuven.be\") "
# && \
#             (machine != \"amethyst.esat.kuleuven.be\") && \
#             (machine != \"vega.esat.kuleuven.be\") && \
#             (machine != \"wasat.esat.kuleuven.be\") && \
#             (machine != \"unuk.esat.kuleuven.be\") && \
#             (machine != \"emerald.esat.kuleuven.be\") && \
#             (machine != \"wulfenite.esat.kuleuven.be\") && \
#             (machine != \"chokai.esat.kuleuven.be\") && \
#             (machine != \"pyrite.esat.kuleuven.be\") && \
#             (machine != \"ymir.esat.kuleuven.be\") "
condor_submit.write("Requirements = (CUDARuntimeVersion == 9.1) && (CUDAGlobalMemoryMb >= {0}) && (CUDACapability >= 3.5) && (machine =!= LastRemoteHost) && (target.name =!= LastMatchName1) && (target.name =!= LastMatchName2) {1} \n".format(FLAGS.gpumem, blacklist))
condor_submit.write("+RequestWalltime = {0} \n".format(FLAGS.wall_time))

if not FLAGS.not_nice: condor_submit.write("Niceuser = true \n")

condor_submit.write("Initialdir       = {0}\n".format(temp_dir))
condor_submit.write("Executable       = {0}\n".format(shell_file))
condor_submit.write("Log              = {0}/condor_{1}.log\n".format(condor_output_dir, description))
condor_submit.write("Output           = {0}/condor_{1}.out\n".format(condor_output_dir, description))
condor_submit.write("Error            = {0}/condor_{1}.err\n".format(condor_output_dir, description))
condor_submit.write("Notification = Error \n")
condor_submit.write("Queue\n")

condor_submit.close()

subprocess.call(shlex.split("chmod 711 {0}".format(condor_file)))


##########################################################################################################################
# STEP 4 Create executable shell file 

executable = open(shell_file,'w')

executable.write("#!/bin/bash \n")
executable.write("echo \"[condor_script] started executable for condor_offline: $(date +%F_%H:%M)\" \n")

if FLAGS.python_script == 'main.py':
  # in case of offline training work locally in /tmp and copy dataset
  if '--dataset' in others and not '--load_data_in_ram' in others and FLAGS.copy_dataset:
    dataset=others[others.index('--dataset')+1]
    if os.path.isfile('{0}/{1}{2}.tar.gz'.format(FLAGS.home, FLAGS.data_root, dataset)):
      executable.write("echo \"Copy compressed dataset to /tmp $(date +%H:%M:%S)\" \n")
      executable.write("cp {0}/{1}{2}.tar.gz /tmp\n".format(FLAGS.home, FLAGS.data_root, dataset))
      executable.write("echo \"extract dataset to /tmp $(date +%H:%M:%S)\" \n")
      # extraction of 10 g took around 3min on opal
      executable.write("tar -xzf /tmp/{0}.tar.gz -C /tmp \n".format(dataset))
      executable.write("sed -i 's/esat\/opal\/kkelchte\/docker_home\/pilot_data/tmp/' /tmp/{0}/*.txt \n".format(dataset))
    else:      
      executable.write("echo \"Copy dataset as separate files to /tmp. $(date +%H:%M:%S)\" \n")
      executable.write("while read line ; do \n")
      executable.write("  cp -r $line /tmp\n")
      executable.write("  echo /tmp/$(basename $line) >> /tmp/train_set.txt\n")
      executable.write("done < {0}/{1}{2}/train_set.txt \n".format(FLAGS.home, FLAGS.data_root, others[others.index('--dataset')+1]))
      executable.write("while read line ; do \n")
      executable.write("  cp -r $line /tmp\n")
      executable.write("  echo /tmp/$(basename $line) >> /tmp/val_set.txt\n")
      executable.write("done < {0}/{1}{2}/val_set.txt \n".format(FLAGS.home, FLAGS.data_root, others[others.index('--dataset')+1]))
      executable.write("while read line ; do \n")
      executable.write("  cp -r $line /tmp\n")
      executable.write("  echo /tmp/$(basename $line) >> /tmp/test_set.txt\n")
      executable.write("done < {0}/{1}{2}/test_set.txt \n".format(FLAGS.home, FLAGS.data_root, others[others.index('--dataset')+1]))
      others[others.index('--dataset')+1]='.'
    executable.write("echo \"Finished data copy $(date +%H:%M:%S)\" \n")  
    FLAGS.data_root='/tmp'

  # create temporary log directory
  executable.write("mkdir -p /tmp/{0}{1} \n".format(FLAGS.summary_dir, FLAGS.log_tag))
  # adjust data_root to absolute path (as home environment will be set to /tmp)
  if FLAGS.data_root != '/tmp': FLAGS.data_root = '{0}/{1}'.format(FLAGS.home, FLAGS.data_root)
  # check if there is a previous checkpoint saved in summary_dir
  # if os.path.isdir(FLAGS.home+'/'+FLAGS.summary_dir+FLAGS.log_tag): # log_tag is still motherdir/run_number (without date tag)
  try:
    # search for checkpoint and config file in current directory and sub directories
    date_dirs = [FLAGS.home+'/'+FLAGS.summary_dir+FLAGS.log_tag+'/'+d for d in ['.']+sorted(os.listdir(FLAGS.home+'/'+FLAGS.summary_dir+FLAGS.log_tag)) if os.path.isfile(FLAGS.home+'/'+FLAGS.summary_dir+FLAGS.log_tag+'/'+d+'/checkpoint') and os.path.isfile(FLAGS.home+'/'+FLAGS.summary_dir+FLAGS.log_tag+'/'+d+'/configuration.xml')]
    checkpoint_file=date_dirs[-1]+'/checkpoint'
    conf_file=date_dirs[-1]+'/configuration.xml'
    if date_dirs[-1].split('/')[-1].startswith('2018'): # incase model is saved in date-tagged folder
      date_tag=date_dirs[-1].split('/')[-1]
      executable.write("mkdir -p /tmp/{0}{1}/{2} \n".format(FLAGS.summary_dir, FLAGS.log_tag, date_tag))
      for f in [checkpoint_file, conf_file]: executable.write("cp {0} /tmp/{1}{2}/{3} \n".format(f, FLAGS.summary_dir, FLAGS.log_tag, date_tag))
    else:
      for f in [checkpoint_file, conf_file]: executable.write("cp {0} /tmp/{1}{2} \n".format(f, FLAGS.summary_dir, FLAGS.log_tag))
  except:
    print("Did not find previous checkpoint in {0}".format(FLAGS.summary_dir+FLAGS.log_tag))
    # in case no previous moodel is found copy checkpoint path
    if '--checkpoint_path' in others:
      checkpoint_path=others[others.index('--checkpoint_path')+1]
    else:
      checkpoint_path='mobilenet_025'
    executable.write('mkdir -p /tmp/{0}{1} \n'.format(FLAGS.summary_dir,checkpoint_path))
    executable.write('cp {0}/{1}{2}/checkpoint /tmp/{1}{2} \n'.format(FLAGS.home,FLAGS.summary_dir,checkpoint_path))

  executable.write('export HOME=/tmp \n')

else:
  executable.write("export HOME={0}\n".format(FLAGS.home))

executable.write("export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64:/users/visics/kkelchte/local/lib/cudnn-7.1/lib64 \n")
executable.write("source /users/visics/kkelchte/tensorflow_1.8/bin/activate \n")
executable.write("export PYTHONPATH=/users/visics/kkelchte/tensorflow_1.8/lib/python2.7/site-packages:{0}/tensorflow/{1}:{0}/tensorflow/tf_cnnvis \n".format(FLAGS.home, FLAGS.python_project+'/..'))
command="python {0}/tensorflow/{1}/{2}".format(FLAGS.home,FLAGS.python_project,FLAGS.python_script)
command="{0} --summary_dir {1} ".format(command, FLAGS.summary_dir)
command="{0} --data_root {1} ".format(command, FLAGS.data_root)
# command="{0} --log_tag {1} ".format(command, FLAGS.log_tag)
FLAGS.log_tag=FLAGS.log_tag+'/'+time.strftime("%Y-%m-%d_%I%M")
command="{0} --log_tag {1} ".format(command, FLAGS.log_tag)
for e in others: command=" {0} {1}".format(command, e)
print("Command: {0}".format(command))
executable.write("{0}\n".format(command))
executable.write("retVal=$? \n")
executable.write("echo \"check exit code: $retVal\" \n")

executable.write("if [ -e /tmp/{0}{1}/checkpoint ] ; then \n".format(FLAGS.summary_dir, FLAGS.log_tag))
executable.write("echo \"[condor_script] change checkpoint with sed\"\n")
executable.write("sed -i 's/\/tmp/{0}/' /tmp/{1}{2}/checkpoint \n".format(FLAGS.home.replace('/','\/'), FLAGS.summary_dir, FLAGS.log_tag))
executable.write("fi \n") 

executable.write("echo \"[condor_script] copy back $(date +%F_%H:%M)\"\n")
executable.write("cp -r /tmp/{0}* {1}/{0} \n".format(FLAGS.summary_dir, FLAGS.home))

executable.write("if [ $retVal -ne 0 ]; then \n")
executable.write("    echo Error \n")
executable.write("    exit $retVal \n")
executable.write("fi \n")
executable.write("echo \"[condor_script] done: $(date +%F_%H:%M)\"\n")

if FLAGS.evaluate_after: # create default run_script call for evaluating with reuse_default canyons on turtle with 20 runs
	command_online="-t {0} ".format(FLAGS.log_tag)
	if FLAGS.not_nice: command_online="{0} --not_nice ".format(command_online)
	command_online="{0} --gpumem {1} ".format(command_online, FLAGS.gpumem)
	command_online="{0} --rammem {1} ".format(command_online, FLAGS.rammem)
	command_online="{0} --diskmem {1} ".format(command_online, FLAGS.diskmem)
	command_online="{0} --wall_time {1} ".format(command_online, 60*40)
	command_online="{0} -e --reuse_default_world -n 20 -m {1} ".format(command_online, FLAGS.log_tag)
	command_online="{0} --paramfile eva_params.yaml -w canyon --fsm nn_turtle_fsm".format(command_online)
	print("command online: \n {0}".format(command_online))

	executable.write("if [ $( ls {0}/{1}/2018* | grep my-model | wc -l ) -gt 2 ] ; then \n".format(FLAGS.summary_dir, FLAGS.log_tag))
	executable.write("  echo \" $(date +%F_%H:%M) [condor_shell_script] Submit condor online job for evaluation \" \n")
	executable.write("  ssh opal {0}/tensorflow/{1}/../scripts/condor_online.py {2} \n".format(FLAGS.home, FLAGS.python_project, command_online))
	executable.write("else \n")
	executable.write("  echo \"Training model {0} offline has failed.\" \n".format(FLAGS.log_tag))
	executable.write("fi \n")

executable.close()

subprocess.call(shlex.split("chmod 711 {0}".format(shell_file)))


##########################################################################################################################
# STEP 5 Submit
print("for more information: \n cd {0}".format(condor_output_dir))
if not FLAGS.dont_submit:
  subprocess.call(shlex.split("condor_submit {0}".format(condor_file)))
  print("Submission done.")
  print("Monitor with: ")
  print("tail -f {0}/condor_{1}.out".format(condor_output_dir, description))
  time.sleep(1)
else:
  print("Job was not submitted.")
    
