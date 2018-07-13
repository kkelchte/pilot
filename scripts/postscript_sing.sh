#!/bin/bash
# RemoteHost=$(cat $_CONDOR_JOB_AD | grep RemoteHost | head -1 | cut -d '=' -f 2 | cut -d '@' -f 2 | cut -d '.' -f 1)
Command=$(cat $_CONDOR_JOB_AD | grep Cmd | grep kkelchte | head -1 | cut -d '/' -f 8)

# rm /tmp/singlebel

# singularity fails to kill all consequently. 
echo "[$(date +%F_%H:%M)] $Command : Killall -u on $RemoteHost." >> /users/visics/kkelchte/condor/pre_post_script.out

killall -u kkelchte