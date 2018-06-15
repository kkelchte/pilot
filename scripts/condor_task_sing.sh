#!/usr/bin/bash
# This scripts sets some parameters for running a tasks,

usage() { echo "Usage: $0 [-g GPU4: true in case you want to use GPU of 4G or bigger.]
		[-t TAG: tag this test for log folder ]
		[-s SCRIPT: define which scripts from simulation_supervised should run, default evaluate.]
		[-m MODELDIR: checkpoint to initialize weights with in logfolder, leave open for training from scratch]
    [-n NUMBER_OF_FLIGHTS: number of episodes flights training online]
    [-w \" WORLDS \" : space-separated list of gazebo worlds ex \" canyon forest sandbox \" ]
    [-p \" PARAMS \" : space-separated list of tensorflow flags ex \" --auxiliary_depth True --max_episodes 20 \" ]" 1>&2; exit 1; }
GPU4=false
MODELDIR=''
SCRIPT='evaluate_model_turtle.sh'
WALLTIME=$((2*60*60))
while getopts ":g:t:s:m:n:w:p:q:" o; do
    case "${o}" in
    		g)
						GPU4=${OPTARG} ;;
        t)
            TAG=${OPTARG} ;;
        s)
						SCRIPT=${OPTARG} ;;
				m)
						MODELDIR=${OPTARG} ;;
				n)
						NUMBER_OF_FLIGHTS=${OPTARG} ;;
				w)
						WORLDS+=(${OPTARG}) ;;
				p)
						PARAMS+=(${OPTARG}) ;;
        q)
            WALLTIME=${OPTARG} ;;
        *)
            usage ;;
    esac
done
shift $((OPTIND-1))


echo
echo "$(tput setaf 3)--------: CONDOR ONLINE :--------"
echo "TAG: $TAG"
echo "GPU4: $GPU4"
echo "MODELDIR: $MODELDIR"
echo "NUMBER_OF_FLIGHTS: $NUMBER_OF_FLIGHTS"
echo "WORLDS: ${WORLDS[@]}"
echo "SCRIPT: $SCRIPT"
echo "PARAMS: ${PARAMS[@]}"
echo "WALLTIME: ${WALLTIME}"
echo
tput sgr 0 

# Add -g false to avoid to loose computation time for displaying control or depth predictions
COMMAND=(./scripts/${SCRIPT} -s start_python_sing_ql.sh -g false)
if [ ! -z "$TAG" ] ; then
  COMMAND+=(-t $TAG)
fi
if [ ! -z "$MODELDIR" ] ; then
	COMMAND+=(-m $MODELDIR)
#   if [ $MODELDIR = 'mobilenet_025' ] ; then
#     PARAMS+=(--continue_training False)
#   fi
# else 
#   PARAMS+=(--scratch True)
fi
if [ ! -z "$NUMBER_OF_FLIGHTS" ] ; then
	COMMAND+=(-n $NUMBER_OF_FLIGHTS)
fi
if [[ ! -z "$WORLDS" ]] ; then
    for w in "${WORLDS[@]}" ; do
      COMMAND+=(-w $w)
    done
fi
if [[ ! -z "$PARAMS" ]] ; then
    for p in "${PARAMS[@]}" ; do
      COMMAND+=(-p $p)
    done
fi

echo "COMMAND: ${COMMAND[@]}"

# change up to two / by _
description="$(echo "$TAG" | sed  -e "s/\//_/" | sed  -e "s/\//_/")_${NAME}_$(date +%F_%H%M)"
# description="${TAG}_${NAME}_$(date +%F_%H%M)"
condor_output_dir="/esat/opal/kkelchte/docker_home/tensorflow/log/${TAG}/condor"
temp_dir="/esat/opal/kkelchte/docker_home/tensorflow/log/${TAG}/condor/.tmp"
condor_file="${temp_dir}/online_${description}.condor"
shell_file="${temp_dir}/run_${description}.sh"
sing_file="${temp_dir}/sing_${description}.sh"

mkdir -p $condor_output_dir
mkdir -p $temp_dir

#--------------------------------------------------------------------------------------------
# Delete previous log files if they are there
if [ -d $condor_output_dir ];then
	rm -f "$condor_output_dir/online_${description}.log"
	rm -f "$condor_output_dir/online_${description}.out"
	rm -f "$condor_output_dir/online_${description}.err"
else
	mkdir $condor_output_dir
fi
#--------------------------------------------------------------------------------------------
echo "Universe         = vanilla" > $condor_file
echo "RequestCpus      = 6"      >> $condor_file
echo "Request_GPUs     = 1"      >> $condor_file
echo "RequestMemory    = 3G"     >> $condor_file
echo "RequestDisk      = 19G"   >> $condor_file

#### Add code to avoid 2 jobs on the same machine
# Add prescript to mount /esat folder

echo "Should_transfer_files = true" >> $condor_file
echo "transfer_input_files = /esat/opal/kkelchte/docker_home/tensorflow/q-learning/scripts/prescript_sing.sh,/esat/opal/kkelchte/docker_home/tensorflow/q-learning/scripts/postscript_sing.sh" >> $condor_file
echo "+PreCmd = \"prescript_sing.sh\"" >> $condor_file
echo "+PostCmd = \"postscript_sing.sh\"" >> $condor_file
echo "when_to_transfer_output = ON_EXIT_OR_EVICT" >> $condor_file


# echo "job_machine_attrs = Machine" >> $condor_file
# echo "job_machine_attrs_history_length = 4" >> $condor_file
# previous_machines="target.machine =!= MachineAttrMachine0 && \
#    target.machine =!= MachineAttrMachine1 && \
#    target.machine =!= MachineAttrMachine2 && \
#    target.machine =!= MachineAttrMachine3"

echo "periodic_release = HoldReasonCode == 1 && HoldReasonSubCode == 0" >> $condor_file

# blacklist="( machineowner == \"Visics\" && machine != \"andromeda.esat.kuleuven.be\")"
blacklist=" && (machineowner == \"Visics\") && \
          (machine != \"andromeda.esat.kuleuven.be\") && \
          (machine != \"vega.esat.kuleuven.be\") && \
          (machine != \"wasat.esat.kuleuven.be\") && \
          (machine != \"nickeline.esat.kuleuven.be\") && \
          (machine != \"unuk.esat.kuleuven.be\") && \
          (machine != \"ymir.esat.kuleuven.be\") && \
          (machine != \"emerald.esat.kuleuven.be\") && \
          (machine != \"pollux.esat.kuleuven.be\") && \
          (machine != \"umbriel.esat.kuleuven.be\") && \
          (machine != \"triton.esat.kuleuven.be\") && \
          (machine != \"amethyst.esat.kuleuven.be\") && \
          (machine != \"ulexite.esat.kuleuven.be\") && \
          (machine != \"garnet.esat.kuleuven.be\") && \
      	  (machine != \"estragon.esat.kuleuven.be\") && \
          (machine != \"spinel.esat.kuleuven.be\") && \
          (machine != \"diamond.esat.kuleuven.be\") && \
	        (machine != \"ricotta.esat.kuleuven.be\")"

# greenlist=""
# greenlist=" && ((machine == \"citrine.esat.kuleuven.be\") || \
#             (machine == \"pyrite.esat.kuleuven.be\") || \
#             (machine == \"opal.esat.kuleuven.be\") || \
#             (machine == \"kunzite.esat.kuleuven.be\") || \
#             (machine == \"iolite.esat.kuleuven.be\") || \
#             (machine == \"hematite.esat.kuleuven.be\") || \
#             (machine == \"amethyst.esat.kuleuven.be\") || \

# from umbriel this is due to driver version mismatch

if [ $GPU4 = true ] ; then
  GPU_MEM=3900
else 
  GPU_MEM=1900
fi

echo "Requirements = (CUDAGlobalMemoryMb >= $GPU_MEM) && (CUDACapability >= 3.5) && HasSingularity $blacklist $greenlist">> $condor_file

# if [[ -z $blacklist ]] ; then
#   echo "Requirements = (CUDAGlobalMemoryMb >= $GPU_MEM) && (CUDACapability >= 3.5) && HasSingularity">> $condor_file
#   # echo "Requirements = (CUDAGlobalMemoryMb >= $GPU_MEM) && (CUDACapability >= 3.5) && HasSingularity && $previous_machines">> $condor_file
# else
#   echo "Requirements = (CUDAGlobalMemoryMb >= $GPU_MEM) && (CUDACapability >= 3.5) && HasSingularity && $blacklist">> $condor_file
#   # echo "Requirements = (CUDAGlobalMemoryMb >= $GPU_MEM) && (CUDACapability >= 3.5) && HasSingularity && $blacklist && $previous_machines">> $condor_file
# fi

# wall time ==> generally assumed a job should take 6hours longest,
# job will be killed after this walltime
# default is 6h ~ 21600
# 15 hours
# echo "+RequestWalltime = 54000"  >> $condor_file
# 10 hours
# echo "+RequestWalltime = 36000"  >> $condor_file
# 3 hours:
# echo "+RequestWalltime = 10800"  >> $condor_file
# 1 hour:
# echo "+RequestWalltime = 3600"  >> $condor_file
# 30 min:
# echo "+RequestWalltime = 1800"  >> $condor_file
# 10 min:
# echo "+RequestWalltime = 59"  >> $condor_file
echo "+RequestWalltime = $WALLTIME"  >> $condor_file

# echo "Niceuser = true"           >> $condor_file

echo "Initialdir       = $temp_dir"   >> $condor_file
# echo "Initialdir       = /esat/opal/kkelchte/docker_home/tensorflow/q-learning/scripts"   >> $condor_file
# echo "Executable       = /usr/bin/singularity" >> $condor_file
echo "Executable       = $sing_file" >> $condor_file
# echo "Executable       = /esat/opal/kkelchte/docker_home/tensorflow/q-learning/scripts/start_singularity.sh" >> $condor_file

echo "Arguments        = $shell_file" >> $condor_file
echo "Log 	           = $condor_output_dir/condor_${description}.log" >> $condor_file
echo "Output           = $condor_output_dir/condor_${description}.out" >> $condor_file
echo "Error            = $condor_output_dir/condor_${description}.err" >> $condor_file
echo "Notification = Error"      >> $condor_file

echo "Queue"                     >> $condor_file
#--------------------------------------------------------------------------------------------
echo "#!/bin/bash"           > $shell_file
echo "echo started singularity."           >> $shell_file
echo "source /esat/opal/kkelchte/docker_home/.entrypoint_xpra" >> $shell_file
echo "roscd simulation_supervised" >> $shell_file
echo "echo PWD: \$PWD" >> $shell_file
echo "${COMMAND[@]}  >> $condor_output_dir/condor_${description}.dockout" >> $shell_file
echo "echo \"[condor_shell_script] done: \$(date +%F_%H:%M)\"" >> $shell_file
#--------------------------------------------------------------------------------------------
# create sing file to ls gluster directory : bug of current singularity + fedora 27 version
echo "#!/bin/bash"                                                                                     > $sing_file
# echo "echo \"exec \$1 in singularity image /esat/opal/kkelchte/singularity_images/ros_gazebo_tensorflow.img\" " >> $sing_file
echo "echo \"exec \$1 in singularity image /gluster/visics/singularity/ros_gazebo_tensorflow.imgs\" " >> $sing_file
# echo "cd /esat/opal/kkelchte/singularity_images"                                                                 >> $sing_file
echo "cd /gluster/visics/singularity"                                                                 >> $sing_file
echo "pwd"                                                                                            >> $sing_file
# echo "ls /esat/opal/kkelchte/singularity_images"                                                                 >> $sing_file
echo "ls /gluster/visics/singularity"                                                                 >> $sing_file
echo "sleep 1"                                                                                        >> $sing_file
# echo "/usr/bin/singularity exec --nv /esat/opal/kkelchte/singularity_images/ros_gazebo_tensorflow.img \$1"      >> $sing_file
echo "/usr/bin/singularity exec --nv /gluster/visics/singularity/ros_gazebo_tensorflow.imgs \$1"      >> $sing_file
#--------------------------------------------------------------------------------------------

chmod 600 $condor_file
chmod 711 $shell_file
chmod 711 $sing_file

condor_submit $condor_file
echo $condor_file
