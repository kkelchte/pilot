#!/bin/bash


# Because F27 can't find any singularity image over the network: 
# not on opal nor on gluster, you have to make the job first visit
# the directory in which the image is saved and start singularity from there.

# echo "exec $1 in singularity image /esat/opal/kkelchte/singularity_images/ros_gazebo_tensorflow_xpra.img"
# ls /esat/opal/kkelchte/singularity_images/
# sleep 0.1
# /usr/bin/singularity exec --nv /esat/opal/kkelchte/singularity_images/ros_gazebo_tensorflow_xpra.img $1


echo "exec $1 in singularity image /gluster/visics/singularity/ros_gazebo_tensorflow.imgs"
cd /gluster/visics/singularity
pwd
ls /gluster/visics/singularity
sleep 1
/usr/bin/singularity exec --nv /gluster/visics/singularity/ros_gazebo_tensorflow.imgs $1
