#!/bin/bash

# PLEASE NOTE
# We assume that you fulfilled step_one.sh before running this script
# Please run the script bellow within a doshico container.
# This can be done with the following command after setting the variables:
# HOME_HOST=/home/klaas/docker_home
# HOME_CONT=/home/$USER
# sudo nvidia-docker run -it \
# 	--rm \
# 	-v $HOME_HOST:$HOME_CONT \
# 	-v /tmp/.X11-unix:/tmp/.X11-unix \
# 	--name doshico_container \
# 	doshico_image $HOME_CONT/step_two.sh -i true -t false
# Place this file in $HOME_HOST
clear

INTERACTIVE=true
usage() { 
		echo 
		echo "Testing and installation script. Options: 
		-i true or false for interaction." 1>&2; exit 1; }
while getopts ":i:h:t:" o; do
	case "${o}" in
		i)
			INTERACTIVE=${OPTARG} ;;
		h)
			usage ;;
		*)
			usage ;;
	esac
done
shift $((OPTIND-1))

continue_question(){
	if [ $INTERACTIVE = true ] ; then
		read -p "$(tput setaf 2) Do you want to continue? ([y],n) " answer 
		tput sgr 0
		answer="_$answer"
		if [ $answer == "_n" ] ; then
			exit
		fi
	fi
}
# Make check if you are within a docker-container
if [ ! -f /.dockerenv ] ; then
	echo "If you have ROS-gazebo-tensorflow installed within a docker environment,"
	echo "pleas launch this script from within this docker enviroment."
	continue_question
fi

tput setaf 3 # print this text yellow
echo --------------------------------------------------------------------------------------------------------
echo
echo
echo "Step 2: Installing and testing of the online performance by flying in ESAT simulated environment."
echo
echo
echo --------------------------------------------------------------------------------------------------------
echo
tput sgr 0 # reset text color to default

clone_git(){
	if [ ! -d $HOME/$1/src/$2 ] ; then
		mkdir -p $HOME/$1/src
		cd $HOME/$1 && catkin_make
		cd $HOME/$1/src
		git clone https://github.com/${2}
		if [ ! -d $HOME/$1/src/$(basename $2) ] ; then
				echo "Something went wrong... with $2 in $1."
				exit
			fi
	else
		echo "Found $2 @ $HOME/$1/src/$2"
	fi
	cd $HOME/$1 && catkin_make
	source $HOME/$1/devel/setup.bash
	echo "source $HOME/$1/devel/setup.bash">>$HOME/.bashrc
	continue_question
}
echo
echo "$(tput setaf 3)--------: Create a catkin workspace for drone simulator :--------$(tput sgr 0)"
echo
clone_git drone_ws kkelchte/hector_quadrotor
echo

echo
echo "$(tput setaf 3)--------: Add bebop drivers :--------$(tput sgr 0)"
echo
clone_git drone_ws kkelchte/bebop_autonomy
# Note: fails at first build due to header file not found. 
# header file: bebop_msgs/Ardrone3MediaRecordStateVideoStateChanged.h
# after rebuild, the problem is gone.

echo
echo "$(tput setaf 3)--------: Create a catkin workspace for simulation supervised :--------$(tput sgr 0)"
echo
clone_git simsup_ws kkelchte/simulation_supervised
echo "export GAZEBO_MODEL_PATH=$HOME/simsup_ws/src/simulation_supervised/simulation_supervised_demo/models" >> $HOME/.bashrc
# Note: if catkin_make failed due to bebop_msg not found:
#		- or you delete the bebop_msg req in CMakeLists.txt in Simulation_supervised_tools
#		- or you source $HOME/drone_ws/devel/setup.bash after cloning bebop_autonomy

echo
echo "$(tput setaf 3)--------: Create an entrypoint besides the .bashrc for when scripts are executed directly :--------$(tput sgr 0)"
echo
if [ ! -e $HOME/.entrypoint ] ; then
	touch $HOME/.entrypoint
	chmod 700 $HOME/.entrypoint
	echo '#!/bin/bash' >> $HOME/.entrypoint
	echo 'set -e' >> $HOME/.entrypoint 
	cat $HOME/.bashrc >> $HOME/.entrypoint
	echo "exec \"\$@\"" >> $HOME/.entrypoint
fi
echo
echo "$(tput setaf 2)Finished installation step_two."
echo "Please test your installation with the test script. (outside a docker container)"
echo "$(tput setaf 3) $HOME_HOST/tensorflow/pilot_online/scripts/test_script.sh$(tput setaf 2)"
echo "Goodluck!"

tput sgr 0

