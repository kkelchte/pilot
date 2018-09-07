#!/bin/bash
# Script should be run without entrypoint
#


echo "##########################   test: NVIDIA   ###########################################################"
if [ -e /dev/nvidia0 ] ; then echo '[TEST_SINGULARITY]: nvidia drivers are loaded.';
else
	tput setaf 1
	echo '[ERROR] nvidia drivers are not loaded:'
	echo 'ls /dev'; ls /dev
	tput sgr 0
	exit
fi

# if [ 0 -gt $(nvidia-smi | wc -l) ] ; then echo '[TEST_SINGULARITY]: nvidia library is found.';
# else
# 	tput setaf 1
# 	echo '[ERROR] nvidia library not found:'
# 	echo 'ls /bin | grep nvidia'; ls /bin | grep nvidia
# 	tput sgr 0
# 	exit
# fi

echo "##########################   test: NFS  ###########################################################"
if [ -e /esat/qayd/kkelchte ] ; then echo '[TEST_SINGULARITY]: ESAT NFS loaded correctly.';
else
	tput setaf 1
	echo '[ERROR] /esat not mounted:'
	echo 'ls /'; ls /
	echo 'ls /esat'; ls /esat
	tput sgr 0
	exit
fi
echo "##########################   test: XPRA  ###########################################################"
export HOME=/esat/qayd/kkelchte/docker_home
export XAUTHORITY=$HOME/.Xauthority
export DISPLAY=:100
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=''

xpra --xvfb="Xorg -noreset -nolisten tcp \
    -config /etc/xpra/xorg.conf\
    -logfile ${HOME}/.xpra/Xorg-${DISPLAY}.log" \
    start $DISPLAY
sleep 3
# test
if [ $(xdpyinfo | grep GLX | wc -w) -ge 2 ] ; then
	echo "[TEST_SINGULARITY]: started xpra with GL successfully"
else
	tput setaf 1
	echo "ERROR: failed to start xpra with GLX."
	echo "------xdpyinfo"
	xdpyinfo 
	echo "------ps -ef | xpra"
	ps -ef | grep xpra
	echo "------printenv"
	printenv
	tput sgr 0
	exit
fi
echo "##########################   test: ROS  ###########################################################"
export HOME=/esat/qayd/kkelchte/docker_home
source /opt/ros/$ROS_DISTRO/setup.bash
roscore &
sleep 10
if [ -z "$(ps -ef | grep ROS)" ] ; then
	tput setaf 1
	echo '[ERROR] roscore not started.'
	echo '$(ps -ef | grep ROS)'; $(ps -ef | grep ROS)
	echo 'printenv | grep ROS'; printenv | grep ROS
	tput sgr 0
	exit
fi
echo "##########################   test: drone_ws  ###########################################################"
export HOME=/esat/qayd/kkelchte/docker_home
source $HOME/drone_ws/devel/setup.bash --extend
ROS_PACKAGE_PATH=$HOME/drone_ws/src:$ROS_PACKAGE_PATH
roscd hector_quadrotor
echo $PWD
if [ "$(basename $PWD)" != "hector_quadrotor" ] ; then
	tput setaf 1
	echo '[ERROR] drone_ws not sourced.'
	echo '$(basename $PWD)'; $(basename $PWD)
	tput sgr 0
	exit
fi
echo "##########################   test: simsup_ws  ###########################################################"
export HOME=/esat/qayd/kkelchte/docker_home
source $HOME/simsup_ws/devel/setup.bash --extend
export GAZEBO_MODEL_PATH=$HOME/simsup_ws/src/simulation_supervised/simulation_supervised_demo/models
ROS_PACKAGE_PATH=$HOME/simsup_ws/src:$ROS_PACKAGE_PATH
roscd simulation_supervised
echo $PWD
if [ "$(basename $PWD)" != "simulation_supervised" ] ; then
	tput setaf 1
	echo '[ERROR] simsup_ws not sourced.'
	echo '$(basename $PWD)'; $(basename $PWD)
	tput sgr 0
	exit
fi

echo "##########################   test: tensorflow  ###########################################################"

export PYTHONPATH=$PYTHONPATH:$HOME/tensorflow/ensemble_v0

# add cuda libraries for tensorflow, could be included in docker build...
# export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-8.0/lib64:/usr/local/cudnn/lib64:$LD_LIBRARY_PATH
# export PATH=$PATH:/usr/local/nvidia/bin
python  -c "import tensorflow as tf; tf.Session()"

echo "#####################################################################################"
