#!/bin/bash
######################################################
# Modular launch script to launch everything together within container
# You don't have to run this script in a running container.
# You should make sure that the environment variables $HOME_HOST and $HOME_CONT are correct
HOME_HOST=/home/$USER/docker_home
HOME_CONT=/home/$USER/
######################################################
# Settings:
# -t TAG
# -o OFFLINE
# -n NUMBER_OF_FLIGHTS
# -s SCRIPT
# -m MODELDIR
# -w WORLDS
# -p PARAMS
######################################################

usage() { echo "Usage: $0 [-t TAG: tag this test for log folder ]
    [-m MODELDIR: checkpoint to initialize weights with in logfolder]
    [-o OFFLINE: true or false]
    [-n NUMBER_OF_FLIGHTS]
    [-s SCRIPT]
    [-w \" WORLDS \" : space-separated list of environments ex \" canyon forest sandbox \"]
    [-p \" PARAMS \" : space-separated list of tensorflow flags ex \" --auxiliary_depth True --max_episodes 20 \" ]" 1>&2; exit 1; }

while getopts ":t:m:n:p:w:s:" o; do
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
        w)
            WORLDS=(${OPTARG})
            ;;
        s)
            SCRIPT=${OPTARG}
            ;;
        p)
            PARAMS="${OPTARG}"
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

echo
echo "$(tput setaf 3)--------: TEST :--------"
echo "TAG: $TAG"
echo "OFFLINE: $OFFLINE"
echo "MODELDIR: $MODELDIR"
echo "NUMBER_OF_FLIGHTS: $NUMBER_OF_FLIGHTS"
echo "WORLDS: $WORLDS"
echo "SCRIPT: $SCRIPT"
echo "PARAMS: $PARAMS"
echo
tput sgr 0 

# test_in_docker_container(){
#     sudo nvidia-docker run -it \
#         --rm \
#         -v $HOME_HOST:$HOME_CONT \
#         -v /tmp/.X11-unix:/tmp/.X11-unix \
#         --name doshico_container \
#         doshico_image $@
# }
# source $HOME/drone_ws/devel/setup.bash
# source $HOME/simsup_ws/devel/setup.bash
# echo $GAZEBO_MODEL_PATH
# $HOME/simsup_ws/src/simulation_supervised/simulation_supervised/scripts/evaluate_model.sh -t test1 -m auxd -n 5 -w "esat_v1 esat_v2" -p "--load_config True --continue_training True"
# echo


