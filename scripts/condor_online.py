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

parser = argparse.ArgumentParser(description="""Condor_offline submits a condor job for training a DNN online in a singularity image.""")

# ==========================
#   General Settings
# ==========================
parser.add_argument("--summary_dir", default='tensorflow/log/', type=str, help="Choose the directory to which tensorflow should save the summaries relative to $HOME.")
parser.add_argument("--data_root", default='pilot_data/', type=str, help="Choose the directory to which tensorflow should save the summaries relative to $HOME.")
# parser.add_argument("--code_root", default='~', type=str, help="Choose the directory to which tensorflow should save the summaries.")
parser.add_argument("--home", default='/esat/opal/kkelchte/docker_home', type=str, help="Absolute path to source of code on Opal from which necessary libs are copied to /tmp/home.")
parser.add_argument("-t", "--log_tag", default='testing', type=str, help="LOGTAG: tag used to name logfolder.")
parser.add_argument("--dont_submit",action='store_true', help="In case you dont want to submit the job.")

# ==========================
#   Tensorflow Settings
# ==========================
parser.add_argument("-pp","--python_project",default='pytorch_pilot/pilot', type=str, help="Define in which python project the executable should be started with ~/tenorflow/PROJECT_NAME/main.py: q-learning/pilot, pilot/pilot, ddpg, ....")
parser.add_argument("-pe","--python_environment",default='sing', type=str, help="Define which environment should be loaded in shell when launching tensorlfow. Possibilities: sing, docker, virtualenv.")

#===========================
#   Condor Machine Settings
#===========================
parser.add_argument("--gpumem",default=1900, type=int,help="define the number of gigs required in your GPU.")
parser.add_argument("--cpus",default=11, type=int,help="define the number of cpu cores.")
parser.add_argument("--rammem",default=15, type=int,help="define the number of gigs required in your RAM.")
parser.add_argument("--diskmem",default=50, type=int,help="define the number of gigs required on your HD.")
parser.add_argument("--wall_time",default=60*60*2, help="After training a new condor job can be submitted to evaluate the model after.")
parser.add_argument("--not_nice",action='store_true', help="In case you want higher priority.")
parser.add_argument("--use_blacklist",action='store_true', help="Avoid list of 'lesser' machines.")
parser.add_argument('--blacklist', default=[],nargs='+', help="Define the good machines.")
parser.add_argument("--use_greenlist",action='store_true', help="Enforce list of 'better' machines.")
parser.add_argument('--greenlist', default=[],nargs='+', help="Define the good machines.")

#===========================
#   Evaluation Params ~ parsed like others and added to run_script.py ==> avoid copying arguments defined at 2 locations.
#===========================

FLAGS, others = parser.parse_known_args()

# 4 main directories have to be defined in order to make it also runnable from a read-only system-installed singularity image.
# --> does not work when data_root and summarydir is made temporarily in /tmp/home
# if FLAGS.summary_dir[0] != '/':  # 1. Tensorflow log directory for saving tensorflow logs and xterm logs
#   FLAGS.summary_dir=FLAGS.home+'/'+FLAGS.summary_dir
# if FLAGS.data_root[0] != '/':  # 2. Pilot_data directory for saving data
#   FLAGS.data_root=FLAGS.home+'/'+FLAGS.data_root
# if FLAGS.code_root == '~': # 3. location for tensorflow code (and also catkin workspace though they are found with rospack)
#   FLAGS.code_root = FLAGS.home

# display and save all settings
print("\nCONDOR ONLINE settings:")
for f in FLAGS.__dict__: print("{0}: {1}".format( f, FLAGS.__dict__[f]))
print("Others: {0}".format(others))

##########################################################################################################################
# STEP 2 Create description and logging directories

description="{0}_{1}".format(FLAGS.log_tag.replace('/','_'),time.strftime("%Y-%m-%d_%I-%M-%S"))

print description
condor_output_dir=FLAGS.home+'/'+FLAGS.summary_dir+FLAGS.log_tag+"/condor"
if os.path.isfile("{0}/online_{1}.log".format(condor_output_dir,description)):
    os.remove("{0}/online_{1}.log".format(condor_output_dir,description))
    os.remove("{0}/online_{1}.out".format(condor_output_dir,description))
    os.remove("{0}/online_{1}.err".format(condor_output_dir,description))

temp_dir=condor_output_dir+"/.tmp"
if os.path.isdir(temp_dir):	shutil.rmtree(temp_dir)

condor_file="{0}/online.condor".format(condor_output_dir)
# condor_file="{0}/online_{1}.condor".format(condor_output_dir,description)
shell_file="{0}/run_{1}.sh".format(temp_dir,description)
sing_file="{0}/sing_{1}.sh".format(temp_dir,description)

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
condor_submit.write("RequestCpus      = {0} \n".format(FLAGS.cpus))
if FLAGS.gpumem != 0:
    condor_submit.write("Request_GPUs     = 1 \n")
condor_submit.write("RequestMemory    = {0}G \n".format(FLAGS.rammem))
condor_submit.write("RequestDisk      = {0}G \n".format(FLAGS.diskmem))
condor_submit.write("match_list_length = 6 \n")

# condor_submit.write("Should_transfer_files = true\n")
# condor_submit.write("transfer_input_files = {0}/tensorflow/{1}/scripts/prescript_sing.sh,{0}/tensorflow/{1}/scripts/postscript_sing.sh\n".format(FLAGS.home, FLAGS.python_project+'/..'))
# condor_submit.write("+PreCmd = \"prescript_sing.sh\"\n")
# condor_submit.write("+PostCmd = \"postscript_sing.sh\"\n")
# condor_submit.write("when_to_transfer_output = ON_EXIT_OR_EVICT\n")
condor_submit.write("periodic_release = ( HoldReasonCode == 1 && HoldReasonSubCode == 0 ) || HoldReasonCode == 26\n")

requirements="(HasSingularity)"
if FLAGS.gpumem != 0:
    requirements+=" && (CUDAGlobalMemoryMb >= {0}) && (CUDACapability >= 3.5)".format(FLAGS.gpumem)
if FLAGS.use_blacklist or len(FLAGS.blacklist) != 0:
    if len(FLAGS.blacklist) == 0:
        blacklist=" && (machine != \"virgo.esat.kuleuven.be\") \
                    && (machine != \"leo.esat.kuleuven.be\") \
                    && (machine != \"cancer.esat.kuleuven.be\") \
                    && (machine != \"libra.esat.kuleuven.be\") "
        requirements+=" {0}".format(blacklist)
    else:
        for m in blacklist:
            requirements+=" && (machine != \"{0}.esat.kuleuven.be\")".format(m)
if FLAGS.use_greenlist or len(FLAGS.greenlist) != 0:
    if len(FLAGS.greenlist)==0:
        greenlist=" && ( (machine == \"andromeda.esat.kuleuven.be\") \
                || (machine == \"asahi.esat.kuleuven.be\") \
                || (machine == \"bandai.esat.kuleuven.be\") \
                || (machine == \"ena.esat.kuleuven.be\") \
                || (machine == \"chokai.esat.kuleuven.be\") \
                || (machine == \"daisen.esat.kuleuven.be\") \
                || (machine == \"estragon.esat.kuleuven.be\") \
                || (machine == \"fuji.esat.kuleuven.be\") \
                || (machine == \"hoo.esat.kuleuven.be\") \
                || (machine == \"vauxite.esat.kuleuven.be\") \
                || (machine == \"vladimir.esat.kuleuven.be\") )"
#               || (machine == \"goryu.esat.kuleuven.be\") \

    else:
        greenlist="&& ((machine == \"{0}.esat.kuleuven.be\")".format(FLAGS.greenlist[0])
        for m in FLAGS.greenlist[1:]:
            greenlist+=" || (machine == \"{0}.esat.kuleuven.be\")".format(m)
        greenlist+=")"
    requirements+=" {0}".format(greenlist)


condor_submit.write("Requirements = {0} \n".format(requirements))
# condor_submit.write("Requirements = (HasSingularity) && (CUDAGlobalMemoryMb >= {0}) && (CUDACapability >= 3.5) && (machine =!= LastRemoteHost) && (target.name =!= LastMatchName0) && (target.name =!= LastMatchName1) && (target.name =!= LastMatchName2) && (target.name =!= LastMatchName3)  && (target.name =!= LastMatchName4) && (target.name =!= LastMatchName5) {1} {2}\n".format(FLAGS.gpumem, blacklist, greenlist))
# condor_submit.write("Requirements = (CUDARuntimeVersion == 9.1) && (CUDAGlobalMemoryMb >= {0}) && (CUDACapability >= 3.5) && (target.name =!= LastMatchName1) && (target.name =!= LastMatchName2) {1} {2}\n".format(FLAGS.gpumem, blacklist, greenlist))
condor_submit.write("+RequestWalltime = {0} \n".format(FLAGS.wall_time))

if not FLAGS.not_nice: condor_submit.write("Niceuser = true \n")

condor_submit.write("Initialdir       = {0}\n".format(temp_dir))
condor_submit.write("Executable       = {0}\n".format(sing_file))
condor_submit.write("Arguments        = {0}\n".format(shell_file))
condor_submit.write("Log              = {0}/condor_{1}.log\n".format(condor_output_dir, description))
condor_submit.write("Output           = {0}/condor_{1}.out\n".format(condor_output_dir, description))
condor_submit.write("Error            = {0}/condor_{1}.err\n".format(condor_output_dir, description))
condor_submit.write("Notification = Error \n")
# condor_submit.write("stream_error = True \n")
# condor_submit.write("stream_output = True \n")
condor_submit.write("Queue\n")

condor_submit.close()

subprocess.call(shlex.split("chmod 711 {0}".format(condor_file)))


##########################################################################################################################
# STEP 4 Create executable shell file: called one singularity is started 

executable = open(shell_file,'w')

executable.write("#!/bin/bash \n")
executable.write("echo started executable within singularity. \n")
# executable.write("cd /esat/opal/kkelchte/docker_home \n")
executable.write("cd /tmp/home \n")
executable.write("source .entrypoint_xpra \n")
# executable.write("source .entrypoint_xpra_no_build \n")
executable.write("roscd simulation_supervised/python \n")
executable.write("echo PWD: $PWD \n")

# create command and add arguments to it
command="python run_script.py -pe {0} -pp {1}".format(FLAGS.python_environment, FLAGS.python_project)
command="{0} --summary_dir {1} ".format(command, FLAGS.summary_dir)
command="{0} --data_root {1} ".format(command, FLAGS.data_root)
command="{0} --log_tag {1} ".format(command, FLAGS.log_tag)
# command="{0} --log_tag {1} ".format(command, FLAGS.log_tag+'/pilot')
for e in others: command="{0} {1}".format(command, e)

executable.write("{0} \n".format(command))

executable.write("retRunScript=$? \n")
executable.write("echo \"got exit code within sing image: $retRunScript \"\n")
executable.write("if [ $retRunScript -ne 0 ]; then \n")
executable.write("    echo Error in run_script \n")
executable.write("    exit $retRunScript \n")
executable.write("fi \n")

# executable.write("{0}  >> {1}/condor_{2}.dockout 2>&1\n".format(command, condor_output_dir, description))
# executable.write("echo \"[condor_shell_script] done: $(date +%F_%H:%M)\"\n")

executable.close()

subprocess.call(shlex.split("chmod 711 {0}".format(shell_file)))


##########################################################################################################################
# STEP 5 Create singularity: used to startup singularity with one argument: the shell file

sing = open(sing_file,'w')

# create sing file to ls gluster directory : bug of current singularity + fedora 27 version
sing.write("#!/bin/bash\n")

# Check if there is already a singularity running
sing.write("sleep 2 \n")
sing.write("echo check if Im already running on this machine \n")

sing.write("ClusterId=$(cat $_CONDOR_JOB_AD | grep ClusterId | cut -d '=' -f 2 | tail -1 | tr -d [:space:]) \n")
sing.write("ProcId=$(cat $_CONDOR_JOB_AD | grep ProcId | tail -1 | cut -d '=' -f 2 | tr -d [:space:]) \n")
sing.write("JobStatus=$(cat $_CONDOR_JOB_AD | grep JobStatus | head -1 | cut -d '=' -f 2 | tr -d [:space:]) \n")
sing.write("RemoteHost=$(cat $_CONDOR_JOB_AD | grep RemoteHost | head -1 | cut -d '=' -f 2 | cut -d '@' -f 2 | cut -d '.' -f 1) \n")
sing.write("Command=$(cat $_CONDOR_JOB_AD | grep Cmd | grep kkelchte | head -1 | cut -d '/' -f 8) \n")

sing.write("while [ $(condor_who | grep kkelchte | wc -l) != 1 ] ; do \n")
sing.write("  echo \"[$(date +%F_%H:%M:%S) $Command ] two jobs are running on $RemoteHost, I better leave...\" \n")
sing.write("  ssh opal /usr/bin/condor_hold ${ClusterId}.${ProcId} \n")
sing.write("  while [ $JobStatus = 2 ] ; do \n")
sing.write("    ssh opal /usr/bin/condor_hold ${ClusterId}.${ProcId} \n")
sing.write("    JobStatus=$(cat $_CONDOR_JOB_AD | grep JobStatus | head -1 | cut -d '=' -f 2 | tr -d [:space:]) \n")
sing.write("    echo \"[$(date +%F_%H:%M:%S) $Command ] sleeping, status: $JobStatus\" \n")
sing.write("    sleep $(( RANDOM % 30 )) \n")
sing.write("  done \n")
sing.write("  echo \"[$(date +%F_%H:%M:%S) $Command ] Put $Command on hold, status: $JobStatus\" \n")
sing.write("done \n")

sing.write("echo \"[$(date +%F_%H:%M:%S) $Command ] only $(condor_who | grep kkelchte | wc -l) job is running on $RemoteHost so continue...\" \n")
sing.write("echo \"HOST: $RemoteHost\" \n")
sing.write("\n")

###### Copy docker_home to local tmp
# copy docker_home
sing.write("echo 'make home in tmp' \n")
sing.write("mkdir -p /tmp/home \n")
sing.write("cd /tmp/home \n")
sing.write("echo 'cp entrypoint_xpra' \n")
sing.write("cp {0}/.entrypoint_xpra . \n".format(FLAGS.home))
sing.write("echo 'make data and tensorflow dir' \n")
sing.write("mkdir {0} \n".format(FLAGS.data_root))
sing.write("mkdir -p {0} \n".format(FLAGS.summary_dir))

if '--checkpoint_path' in others:
    # copy checkpoint if it's there
    sing.write("echo 'cp checkpoint' \n")
    checkpoint_path=others[others.index('--checkpoint_path')+1]
    sing.write("mkdir -p /tmp/home/{0}{1} \n".format(FLAGS.summary_dir, checkpoint_path))
    sing.write("if [ -e {1}/{2}{0}/my-model ] ; then \n".format(checkpoint_path, FLAGS.home, FLAGS.summary_dir))
    sing.write("cp {1}/{2}{0}/my-model {2}{0} \n".format(checkpoint_path, FLAGS.home, FLAGS.summary_dir))
    # sing.write("else \n")
    # sing.write("echo 'failed to copy checkpoint' \n")
    # sing.write("cp {1}/{2}{0}/*/my-model {2}{0} || echo 'failed to copy checkpoint' \n".format(checkpoint_path, FLAGS.home, FLAGS.summary_dir))
    sing.write("fi \n")
    sing.write("if [ -e {1}/{2}{0}/configuration.xml ] ; then \n".format(checkpoint_path, FLAGS.home, FLAGS.summary_dir))
    sing.write("cp {1}/{2}{0}/configuration.xml {2}{0} \n".format(checkpoint_path, FLAGS.home, FLAGS.summary_dir))
    # sing.write("else \n")
    # sing.write("cp {1}/{2}{0}/*/configuration.xml {2}{0}  || echo 'failed to copy configuration' \n".format(checkpoint_path, FLAGS.home, FLAGS.summary_dir))
    sing.write("fi \n")

# copy current log_tag folder if it's there
sing.write("if [ -d {1}/{2}{0} ] ; then \n".format(FLAGS.log_tag, FLAGS.home, FLAGS.summary_dir))
sing.write("mkdir -p {2}{0} \n".format(FLAGS.log_tag, FLAGS.home, FLAGS.summary_dir))
sing.write("cp -r {1}/{2}{0}/* {2}{0} \n".format(FLAGS.log_tag, FLAGS.home, FLAGS.summary_dir))
# sing.write("else \n")
# sing.write("cp {1}/{2}{0}/*/configuration.xml {2}{0}  || echo 'failed to copy configuration' \n".format(checkpoint_path, FLAGS.home, FLAGS.summary_dir))
sing.write("fi \n")

sing.write("echo 'cp tensorflow {0} project' \n".format(FLAGS.python_project))
sing.write("cp -r {1}/tensorflow/{0} tensorflow/ \n".format(FLAGS.python_project.split('/')[0], FLAGS.home))
# sing.write("cp -r {0}/tensorflow/tf_cnnvis tensorflow/ \n".format(FLAGS.home))
sing.write("echo 'cp simulation_supervised' \n")
sing.write("cp -r {0}/simsup_ws . \n".format(FLAGS.home))
######

sing.write("sing_image=\"ros_gazebo_tensorflow_writable.img\"\n")
# sing.write("echo check if gluster is accessible: \n")
# sing.write("if [ -f /gluster/visics/singularity/$sing_image ] ; then \n")
# sing.write("  sing_loc=\"/gluster/visics/singularity\" \n")
# sing.write("else \n")
sing.write("sing_loc=\"/esat/opal/kkelchte/singularity_images\" \n")
# sing.write("fi\n")
sing.write("echo \"exec $1 in singularity image $sing_loc/$sing_image\"\n")

# sing.write("echo \"exec $1 in singularity image /gluster/visics/singularity/ros_gazebo_tensorflow_drone_ws.img\" \n")
# sing.write("cd /esat/opal/kkelchte/singularity_images\n")
# sing.write("cd /gluster/visics/singularity\n")
# sing.write(" ls /esat/opal/kkelchte/singularity_images\n")
# sing.write("ls /gluster/visics/singularity\n")

# sing.write("cd $sing_loc\n")
# sing.write("pwd\n")
# sing.write("ls\n")
# sing.write("sleep 1\n")
# sing.write("/usr/bin/singularity exec --nv $sing_loc/$sing_image $1 \n")

# copy singularity image
sing.write("echo cp $sing_loc/$sing_image /tmp\n")
sing.write("cp $sing_loc/$sing_image /tmp\n")

sing.write("/usr/bin/singularity exec --nv /tmp/$sing_image $1 \n")
sing.write("retVal=$? \n")
sing.write("echo \"got exit code $retVal\" \n")

###### Copy data and log back to opal
sing.write("echo 'copy pilot data back' \n")
sing.write("cp -r /tmp/home/{1}* {0}/{1} \n".format(FLAGS.home, FLAGS.data_root))
sing.write("echo 'copy tensorflow log back' \n")
sing.write("cp -r /tmp/home/{1}* {0}/{1} \n".format(FLAGS.home, FLAGS.summary_dir))
#####
sing.write("if [ $retVal -ne 0 ]; then \n")
sing.write("    echo Error \n")
sing.write("    exit $retVal \n")
sing.write("fi \n")

sing.write("echo \"[$(date +%F_%H:%M:%S)] $Command : leaving $RemoteHost.\" \n")
# singularity tend to not always shut down properly so strong kill the condor node
# sing.write("kill -9 $(ps -ef | grep kkelchte | grep condor_pid_ns_init | grep $Command | cut -d ' ' -f 3) \n")

sing.close()

subprocess.call(shlex.split("chmod 711 {0}".format(sing_file)))

##########################################################################################################################
# STEP 6 Submit
if not FLAGS.dont_submit:
  subprocess.call(shlex.split("condor_submit {0}".format(condor_file)))
  print("Submission done.")
  print("Monitor with: ")
  print("tail -f {0}/condor_{1}.dockout".format(condor_output_dir, description))
  time.sleep(1)
else:
  print("Job was not submitted.")
  
