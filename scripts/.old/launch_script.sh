#!/bin/bash
######################################################
# Modular launch script to launch everything together within container
# You don't have to run this script in a running container.
# You should make sure that the environment variables $HOME_HOST and $HOME_CONT are correct
HOME_HOST=/home/$USER/docker_home
HOME_CONT=/home/$USER
######################################################
# Settings:
# -t TAG
# -m MODELDIR
# -o OFFLINE
# -n NUMBER_OF_FLIGHTS
# -s SCRIPT
# -w WORLDS
# -p PARAMS
######################################################

usage() { echo "Usage: $0 [-t TAG: tag this test for log folder ]
    [-m MODELDIR: checkpoint to initialize weights with in logfolder, use None for training from scratch]
    [-o OFFLINE: true or false]
    [-n NUMBER_OF_FLIGHTS: number of episodes training offline or flights training online]
    [-s SCRIPT]
    [-w \" WORLDS \" : space-separated list of environments ex \" canyon forest sandbox \" for training online]
    [-p \" PARAMS \" : space-separated list of tensorflow flags ex \" --auxiliary_depth True --max_episodes 20 \" ]" 1>&2; exit 1; }

SCRIPT='evaluate_model.sh'
OFFLINE=false
while getopts ":t:m:o:n:s:w:p:" o; do
    case "${o}" in
        t)
            TAG=${OPTARG}
            ;;
        m)
            MODELDIR=${OPTARG}
            ;;
        o)
            OFFLINE=${OPTARG}
            ;;
        n)
            NUMBER_OF_FLIGHTS=${OPTARG}
            ;;
        s)
            SCRIPT=${OPTARG}
            ;;
        w)
            WORLDS=(${OPTARG})
            ;;
        p)
            PARAMS=(${OPTARG})
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

echo
echo "$(tput setaf 3)--------: LAUNCH :--------"
echo "TAG: $TAG"
echo "OFFLINE: $OFFLINE"
echo "MODELDIR: $MODELDIR"
echo "NUMBER_OF_FLIGHTS: $NUMBER_OF_FLIGHTS"
echo "WORLDS: ${WORLDS[@]}"
echo "SCRIPT: $SCRIPT"
echo "PARAMS: ${PARAMS[@]}"
echo
tput sgr 0 
############################### OFFLINE
if [ $OFFLINE = true ] ; then
    test_container(){
        sudo nvidia-docker run -it --rm \
            -v $HOME_HOST:$HOME_CONT \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            --name doshico_container \
            doshico_image $@
    }
    if [ -n $TAG ] ; then
        PARAMS+=(--log_tag $TAG)
    fi
    if [ -n $MODELDIR ] ; then
        PARAMS+=(--checkpoint $MODELDIR)
        if [ $MODELDIR = 'mobilenet_025' ] ; then
            PARAMS+=(--continue_training False)
        fi
    fi
    if [ -n $NUMBER_OF_FLIGHTS ] ; then
        PARAMS+=(--max_episodes $NUMBER_OF_FLIGHTS)
    fi
    test_command="python $HOME_CONT/tensorflow/pilot/pilot/main.py --offline True ${PARAMS[@]}"
    echo $test_command
    test_container $test_command
else
############################### ONLINE
    test_container(){
        sudo nvidia-docker run -it --rm \
            -v $HOME_HOST:$HOME_CONT \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            --name doshico_container \
            doshico_image $HOME_CONT/.entrypoint $@
        # The entrypoint sources the simsup_ws and drone_ws environment
    }
    CMD=($HOME/simsup_ws/src/simulation_supervised/simulation_supervised/scripts/${SCRIPT})
    if [ -n $TAG ] ; then
        CMD+=(-t $TAG)
    fi
    if [ -n $MODELDIR ] ; then
        CMD+=(-m $MODELDIR)
        # In case you finetune from image-net, randomly initialize prediction layers. 
        if [ $MODELDIR = 'mobilenet_025' ] ; then
            PARAMS+=(--continue_training False)
        fi
    fi
    if [ -n $NUMBER_OF_FLIGHTS ] ; then
        CMD+=(-n $NUMBER_OF_FLIGHTS)
    fi
    if [[ -n $WORLDS ]] ; then
        for w in "${WORLDS[@]}" ; do
            CMD+=(-w $w)
        done
    fi
    if [[ -n $PARAMS ]] ; then
        for p in "${PARAMS[@]}" ; do
            CMD+=(-p $p)
        done
    fi
    test_container ${CMD[@]}
fi