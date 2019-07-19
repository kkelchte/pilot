#!/bin/bash
# This scripts evaluate the model in log/testing 2 times in canyon and saves result in log/testing_online
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

# for i in 1 2 ; do
#   run_simulation chapter_policy_learning/dagger/reference/${i}_eva chapter_policy_learning/recover/reference/final/${i}
# done


###############
# iteration 1
###############

# for i in 0 1 2 ; do
#   run_simulation chapter_policy_learning/dagger/model_${i}/record chapter_policy_learning/dagger/model_${i}/iteration1/0
# done

###############
# iteration 2
###############

# for i in 0 1 2 ; do
#   run_simulation chapter_policy_learning/dagger/model_${i}/record_2 chapter_policy_learning/dagger/model_${i}/iteration2/0
# done
# Add train/val/test set.txt to data folder:
# for i in 0 1 2 ; do echo $i; cp -r model_$i/record/*.txt model_$i/record_2/; echo $PWD/model_$i/record_2/00000_esatv3 >> model_$i/record_2/train_set.txt; done


cd