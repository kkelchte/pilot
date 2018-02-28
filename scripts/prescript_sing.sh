#!/bin/bash

# printenv >> /users/visics/kkelchte/condor/condor_env
# cat $_CONDOR_JOB_AD >> /users/visics/kkelchte/condor/condor_adv

ClusterId=$(cat $_CONDOR_JOB_AD | grep ClusterId | cut -d '=' -f 2 | tail -1 | tr -d [:space:])
ProcId=$(cat $_CONDOR_JOB_AD | grep ProcId | tail -1 | cut -d '=' -f 2 | tr -d [:space:])
JobStatus=$(cat $_CONDOR_JOB_AD | grep JobStatus | head -1 | cut -d '=' -f 2 | tr -d [:space:])
RemoteHost=$(cat $_CONDOR_JOB_AD | grep RemoteHost | head -1 | cut -d '=' -f 2 | cut -d '@' -f 2 | cut -d '.' -f 1)

# if [[ true ]]; then
if [[ -e /tmp/singlebel ]]; then
	echo "[$(date +%F_%H:%M)] Singlebel exists on $RemoteHost" >> /users/visics/kkelchte/condor/out/pre_post_script.out
	echo "[$(date +%F_%H:%M)] Hold: ${ClusterId}.${ProcId}" >> /users/visics/kkelchte/condor/out/pre_post_script.out
	# put job on idle or hold for reason X
	while [ $JobStatus = 2 ] ; do
		ssh qayd /usr/bin/condor_hold ${ClusterId}.${ProcId}
		# ssh qayd /usr/bin/condor_hold -reason 'singlebel is taken.' -subcode 0 ${ClusterId}.${ProcId}
		JobStatus=$(cat $_CONDOR_JOB_AD | grep JobStatus | head -1 | cut -d '=' -f 2 | tr -d [:space:])
		echo "[$(date +%F_%H:%M)] sleeping, status: $JobStatus" >> /users/visics/kkelchte/condor/out/pre_post_script.out
		sleep 10
	done
	echo "[$(date +%F_%H:%M)] done" >> /users/visics/kkelchte/condor/out/pre_post_script.out
else
	echo "[$(date +%F_%H:%M)] Create singlebel on $RemoteHost." >> /users/visics/kkelchte/condor/out/pre_post_script.out
	touch /tmp/singlebel
fi
