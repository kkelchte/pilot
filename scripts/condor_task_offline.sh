#!/usr/bin/bash
# This scripts sets some parameters for running a tasks,


usage() { echo "Usage: $0 [-g GPU4: true in case you want to use GPU of 4G or bigger.]
        [-t TAG: tag this test for log folder ]
        [-s SCRIPT: define which python scripts from pilot should run]
        [-m MODELDIR: checkpoint to initialize weights with in logfolder, leave open for training from scratch]
        [-e EVALUATE: add -e option if model should be evaluated ]
        [-n NUMBER_OF_FLIGHTS: for evaluation]
        [-w \" WORLDS \" : space-separated list of gazebo worlds for evaluating online ex \" canyon forest sandbox \" ]
        [-p \" PARAMS \" : space-separated list of tensorflow flags ex \" --auxiliary_depth True --max_episodes 20 \" ]" 1>&2; exit 1; }
GPU4=false
SCRIPT="main.py"
MODELDIR=''
EVALUATE=false
NUMBER_OF_FLIGHTS=50
WALLTIME=$((60*60*2))
while getopts ":g:t:s:m:e:n:w:p:q:" o; do
    case "${o}" in
                g)
                        GPU4=${OPTARG} ;;
                t)
                        TAG=${OPTARG} ;;
                s)
                        SCRIPT=${OPTARG} ;;
                m)
                        MODELDIR=${OPTARG} ;;
                e)
                        EVALUATE=${OPTARG} ;;
                n)
                        NUMBER_OF_FLIGHTS=${OPTARG} ;;
                w)
                        WORLDS=(${OPTARG}) ;;
                p)
                        PARAMS=(${OPTARG}) ;;
                q)
                        WALLTIME=${OPTARG} ;;
        *)
            usage ;;
    esac
done
shift $((OPTIND-1))

tput setaf 3
echo
echo "--------: CONDOR OFFLINE :--------"
echo "TAG: $TAG"
echo "GPU4: $GPU4"
echo "MODELDIR: $MODELDIR"
echo "SCRIPT: $SCRIPT"
echo "PARAMS: ${PARAMS[@]}"
echo "WALLTIME: ${WALLTIME}"
if [ $EVALUATE ] ; then
    echo "EVALUATE: $EVALUATE"
    echo "WORLDS: ${WORLDS[@]}"
    echo "NUMBER_OF_FLIGHTS: $NUMBER_OF_FLIGHTS"
fi
echo
tput sgr 0 
# exit

COMMAND_OFFLINE=(${SCRIPT})
if [ ! -z "$TAG" ] ; then
  COMMAND_OFFLINE+=(--log_tag ${TAG}/$(date +%F_%H%M))
fi
if [[ -n "$PARAMS" ]] ; then
    echo ${PARAMS[@]}
    for p in "${PARAMS[@]}" ; do 
        echo $p
        COMMAND_OFFLINE+=($p)
    done
fi
echo "COMMAND_OFFLINE: ${COMMAND_OFFLINE[@]}"
if [ $EVALUATE = true ] ; then
    COMMAND_ONLINE=()
    if [ $GPU4 = true ] ; then
      COMMAND_ONLINE+=(-g true)
    fi
    if [ ! -z "$TAG" ] ; then
      COMMAND_ONLINE+=(-t ${TAG}_eva)
    fi
    COMMAND_ONLINE+=(-m $TAG)
    if [ ! -z "$NUMBER_OF_FLIGHTS" ] ; then
        COMMAND_ONLINE+=(-n $NUMBER_OF_FLIGHTS)
        COMMAND_ONLINE+=(-q $((NUMBER_OF_FLIGHTS * 60 * 20)) )
    fi
    if [[ ! -z "$WORLDS" ]] ; then
      for w in "${WORLDS[@]}" ; do
        COMMAND_ONLINE+=(-w $w)
      done
    fi
    COMMAND_ONLINE+=(-e true)
    COMMAND_ONLINE+=(-p eva_params.yaml)
    # COMMAND_ONLINE+=(-p '--load_config' -p 'True')
    echo "COMMAND_ONLINE: ${COMMAND_ONLINE[@]}"
fi

# change up to two / by _
description="$(echo "$TAG" | sed  -e "s/\//_/" | sed  -e "s/\//_/")_${NAME}_$(date +%F_%H%M)"
# condor_output_dir='/users/visics/kkelchte/condor/log'
condor_output_dir="/esat/opal/kkelchte/docker_home/tensorflow/log/${TAG}/condor"
temp_dir="/esat/opal/kkelchte/docker_home/tensorflow/log/${TAG}/condor/.tmp"
# temp_dir="/users/visics/kkelchte/condor/.tmp"
# temp_dir="/esat/opal/kkelchte/docker_home/tensorflow/q-learning/scripts/.tmp"
condor_file="${temp_dir}/offline_${description}.condor"
shell_file="${temp_dir}/run_${description}.sh"
mkdir -p $condor_output_dir
mkdir -p $temp_dir

#--------------------------------------------------------------------------------------------
# Delete previous log files if they are there
if [ -d $condor_output_dir ];then
    rm -f "$condor_output_dir/offline_${description}.log"
    rm -f "$condor_output_dir/offline_${description}.out"
    rm -f "$condor_output_dir/offline_${description}.err"
else
    mkdir $condor_output_dir
fi
#--------------------------------------------------------------------------------------------
echo "Universe         = vanilla" > $condor_file
echo "RequestCpus      = 4"      >> $condor_file
echo "Request_GPUs     = 1"      >> $condor_file
echo "RequestMemory    = 15G"     >> $condor_file
# echo "RequestMemory    = 30G"     >> $condor_file
echo "RequestDisk      = 50G"   >> $condor_file
#blacklist="distributionversion == \"26\""
# blacklist="(machine != \"andromeda.esat.kuleuven.be\")"
blacklist="(machine != \"andromeda.esat.kuleuven.be\") && \
                    (machine != \"amethyst.esat.kuleuven.be\") && \
                    (machine != \"vega.esat.kuleuven.be\") && \
                    (machine != \"wasat.esat.kuleuven.be\") && \
                    (machine != \"unuk.esat.kuleuven.be\") && \
                    (machine != \"emerald.esat.kuleuven.be\") && \
                    (machine != \"wulfenite.esat.kuleuven.be\") && \
                    (machine != \"chokai.esat.kuleuven.be\") && \
                    (machine != \"pyrite.esat.kuleuven.be\") && \
                    (machine != \"ymir.esat.kuleuven.be\") "
                    # (machine != \"pyrite.esat.kuleuven.be\") && \
                    # (machine != \"lesath.esat.kuleuven.be\") && \
                    # (machine != \"nickeline.esat.kuleuven.be\") && \
                    # (machine != \"pollux.esat.kuleuven.be\") && \
                    # (machine != \"realgar.esat.kuleuven.be\") "

# From umbriel --> due to no GPU

if [ $GPU4 = true ] ; then
    GPUMEM=3900
else 
    GPUMEM=1900
fi
if [ -z "$blacklist" ] ; then
    echo "Requirements = (CUDARuntimeVersion == 9.1) && (CUDAGlobalMemoryMb >= $GPUMEM) && (CUDACapability >= 3.5)">> $condor_file  
else
    echo "Requirements = (CUDARuntimeVersion == 9.1) && (CUDAGlobalMemoryMb >= $GPUMEM) && (CUDACapability >= 3.5) && $blacklist">> $condor_file
fi
# wall time ==> generally assumed a job should take 6hours longest,
# job will be killed after this walltime
# default is 6h ~ 21600
# 15 hours
# echo "+RequestWalltime = 54000"  >> $condor_file
# 10 hours
# echo "+RequestWalltime = 36000"  >> $condor_file
# 3 hours:
echo "+RequestWalltime = $WALLTIME"  >> $condor_file
# 1 hour:
# echo "+RequestWalltime = 3600"  >> $condor_file
# 30 min:
# echo "+RequestWalltime = 1800"  >> $condor_file
# 10 min:
# echo "+RequestWalltime = 59"  >> $condor_file
echo "Niceuser = true"           >> $condor_file

echo "Initialdir       = $temp_dir"   >> $condor_file
echo "Executable       = $shell_file" >> $condor_file
echo "Log              = $condor_output_dir/condor_${description}.log" >> $condor_file
echo "Output           = $condor_output_dir/condor_${description}.out" >> $condor_file
echo "Error            = $condor_output_dir/condor_${description}.err" >> $condor_file
echo "Notification = Error"      >> $condor_file


echo "Queue"                     >> $condor_file
#--------------------------------------------------------------------------------------------
echo "#!/bin/bash"           > $shell_file
echo "echo started executable"           >> $shell_file

# >>>>>SMALL HACK TO COPY CUDA 8 AVOIDING JOB DISCONNECTIONS
# echo "cp -r /users/visics/kkelchte/local/cuda-8.0 /tmp"           >> $shell_file
# echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tmp/cuda-8.0/lib64:/users/visics/kkelchte/local/lib/cudnn/lib64:">>$shell_file
#<<<<<<

echo "export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64:/users/visics/kkelchte/local/lib/cudnn-7.0/lib64">>$shell_file
echo "source /users/visics/kkelchte/tensorflow/bin/activate">>$shell_file
# echo "source /users/visics/kkelchte/tensorflow/bin/activate">>$shell_file
echo "export PYTHONPATH=/users/visics/kkelchte/tensorflow/lib/python2.7/site-packages:/esat/opal/kkelchte/docker_home/tensorflow/q-learning">>$shell_file
echo "export HOME=/esat/opal/kkelchte/docker_home">>$shell_file
echo "cd /esat/opal/kkelchte/docker_home/tensorflow/q-learning/pilot">>$shell_file
echo "echo ${COMMAND_OFFLINE[@]}">>$shell_file
echo "python ${COMMAND_OFFLINE[@]}">>$shell_file
echo "echo \"[condor_shell_script] done: \$(date +%F_%H:%M)\"" >> $shell_file

# >>>>CLEANUP CUDA 8 AVOIDING JOB DISCONNECTIONS
# echo "rm -r /tmp/cuda-8.0"           >> $shell_file
# <<<<<

if [ $EVALUATE = true ] ; then
    echo "if [ \$( ls /esat/opal/kkelchte/docker_home/tensorflow/log/${TAG}/2018* | grep my-model | wc -l ) -gt 2 ] ; then " >> $shell_file
    echo "  echo \" \$(date +%F_%H:%M) [condor_shell_script] Submit condor online job for evaluation \" " >> $shell_file
    echo "  ssh opal /esat/opal/kkelchte/docker_home/tensorflow/q-learning/scripts/condor_task_sing.sh ${COMMAND_ONLINE[@]}" >> $shell_file
    echo "else">> $shell_file
    echo "  echo \"Training model ${TAG} offline has failed.\" ">> $shell_file
    echo "fi">> $shell_file
    # echo "/usr/bin/bash /users/visics/kkelchte/condor/condor_task_sing.sh ${COMMAND_ONLINE[@]} " >> $shell_file
fi
#--------------------------------------------------------------------------------------------

chmod 711 $condor_file
chmod 711 $shell_file

condor_submit $condor_file
echo $condor_file
