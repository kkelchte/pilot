#!/bin/bash
# Test script combo offline training and online evaluation:
echo
echo "--------------------------------------------------"
echo "Started test_train_evaluate.sh: $(date +%H:%M:%S )"
echo "--------------------------------------------------"
echo

# 1. Train model in tensorflow on small dataset
export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64:/users/visics/kkelchte/local/lib/cudnn-7.0/lib64
source /users/visics/kkelchte/tensorflow/bin/activate 
export PYTHONPATH=/users/visics/kkelchte/tensorflow/lib/python2.7/site-packages
export HOME=/esat/opal/kkelchte/docker_home

python ~/tensorflow/pilot/pilot/main.py --max_episodes 5 --load_data_in_ram --network mobile --discrete
ret=$?
if [ $ret -ne 0 ]; then
     #Handle failure
     exit
fi
# 2. Evaluate trained model in singularity 
singularity exec --nv /esat/opal/kkelchte/singularity_images/ros_gazebo_tensorflow_drone_ws.img /esat/opal/kkelchte/docker_home/tensorflow/pilot/scripts/evaluate_in_singularity.sh

echo
echo "--------------------------------------------------"
echo "Ended test_train_evaluate.sh: $(date +%H:%M:%S )"
echo "--------------------------------------------------"
echo