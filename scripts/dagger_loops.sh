#!/bin/bash
# This scripts evaluate the model in log/testing and store data in pilot_data/...
# launch file from singularity image with source tensorflow/pytorch_pilot_beta/scripts/dagger_loops.sh
# cd /esat/opal/kkelchte/docker_home
# source .entrypoint_graph
# source .entrypoint_graph_debug
# source .entrypoint_xpra
# source .entrypoint_xpra_no_build
# roscd simulation_supervised/python
roscd simulation_supervised/python

# pwd

###################
# EVALUATION      #
###################

run_simulation(){
  name="$1"
  model="$2"
  pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --pause_simulator --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --scaled_input"
  script_args="-ds --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 1 --evaluation --final_evaluation_runs 0 --python_project pytorch_pilot_beta/pilot"
  python run_script.py -t $name $script_args $pytorch_args
}
###############
# iteration 0
###############

# for i in 0 1 2 ; do
#   run_simulation chapter_policy_learning/DAGGER/reference/${i}_eva chapter_policy_learning/recover/reference/final/${i}
# done


###############
# iteration 1
###############

# run_simulation chapter_policy_learning/DAGGER/model_0/record1 chapter_policy_learning/DAGGER/model_0/iteration1/lr_000001
# run_simulation chapter_policy_learning/DAGGER/model_1/record1 chapter_policy_learning/DAGGER/model_1/iteration1/lr_0001
# run_simulation chapter_policy_learning/DAGGER/model_2/record1 chapter_policy_learning/DAGGER/model_2/iteration1/lr_00001

###############
# iteration 2
###############

run_simulation chapter_policy_learning/DAGGER/model_0/record2 chapter_policy_learning/DAGGER/model_0/iteration2/lr_000001
run_simulation chapter_policy_learning/DAGGER/model_1/record2 chapter_policy_learning/DAGGER/model_1/iteration2/lr_00001
run_simulation chapter_policy_learning/DAGGER/model_2/record2 chapter_policy_learning/DAGGER/model_2/iteration2/lr_0001


cd