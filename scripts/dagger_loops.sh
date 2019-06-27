#!/bin/bash
# This scripts evaluate the model in log/testing 2 times in canyon and saves result in log/testing_online
# cd /esat/opal/kkelchte/docker_home
source .entrypoint_graph
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
  script_args="-ds --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 2 --evaluation --final_evaluation_runs 0 --python_project pytorch_pilot_beta/pilot"
  python run_script.py -t $name $script_args $pytorch_args
}
###############
# iteration 0
###############

# for i in 0 1 2 ; do
#   run_simulation chapter_policy_learning/how_to_recover_normalized/dagger_opal/reference/${i}_eva chapter_policy_learning/how_to_recover_normalized/res18_reference_pretrained/final/${i}
# done


###############
# iteration 1
###############

# run_simulation chapter_policy_learning/how_to_recover_normalized/dagger_opal/model_0/record chapter_policy_learning/how_to_recover_normalized/dagger_opal/model_0/learning_rates/4
# run_simulation chapter_policy_learning/how_to_recover_normalized/dagger_opal/model_1/record chapter_policy_learning/how_to_recover_normalized/dagger_opal/model_1/learning_rates/0
# run_simulation chapter_policy_learning/how_to_recover_normalized/dagger_opal/model_2/record chapter_policy_learning/how_to_recover_normalized/dagger_opal/model_2/learning_rates/4


###############
# iteration 2
###############
