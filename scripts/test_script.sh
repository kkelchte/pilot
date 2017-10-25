#!/bin/bash
######################################################
# Settings:
# -t TEST
# -m MODELDIR
# -w WORLDS
# -p PARAMS
######################################################

usage() { echo "Usage: $0 [-t LOGTAG: tag used to name logfolder]
    [-m MODELDIR: checkpoint to initialize weights with in logfolder]
    [-n NUMBER_OF_FLIGHTS]
    [-w \" WORLDS \" : space-separated list of environments ex \" canyon forest sandbox \"]
    [-s \" python_script \" : choose the python script to launch tensorflow: start_python or start_python_docker]
    [-p \" PARAMS \" : space-separated list of tensorflow flags ex \" --auxiliary_depth True --max_episodes 20 \" ]" 1>&2; exit 1; }

python_script="start_python_docker.sh"
NUMBER_OF_FLIGHTS=2

while getopts ":t:m:n:p:w:s:" o; do
    case "${o}" in
        t)
            TAG=${OPTARG}
            ;;
        m)
            MODELDIR=${OPTARG}
            ;;
        n)
            NUMBER_OF_FLIGHTS=${OPTARG}
            ;;
        w)
            WORLDS=(${OPTARG})
            ;;
        s)
            python_script=${OPTARG}
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