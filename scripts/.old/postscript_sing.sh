#!/bin/bash
RemoteHost=$(cat $_CONDOR_JOB_AD | grep RemoteHost | head -1 | cut -d '=' -f 2 | cut -d '@' -f 2 | cut -d '.' -f 1)

echo "[$(date +%F_%H:%M)] Clean up singlebel on $RemoteHost." >> /users/visics/kkelchte/condor/out/pre_post_script.out
rm /tmp/singlebel

# singularity fails to kill all consequently. 
killall -u kkelchte