#!/bin/bash
chapter=chapter_policy_learning
section=how_to_recover_normalized/dagger_opal
pytorch_args="--network res18_net --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --scaled_input\
 --batch_size 32 --loss MSE --optimizer SGD --clip 1.0 --weight_decay 0 --normalized_output"

echo "####### chapter: $chapter #######"
echo "####### section: $section #######"

train(){
  cd ..
  script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 "
  condor_args="--wall_time_train $((24*60*60)) --wall_time_eva $((3*3600)) --rammem 7 --gpumem_train 1800 --gpumem_eva 1800"
  dag_args="--model_names $(seq 0 4) --random_numbers 123 --learning_rates 0.0001 0.00001 0.000001"
  python dag_train_and_evaluate.py $pytorch_args $condor_args $dag_args $script_args -t $*
  cd launch
}

##########################################
# Evaluate 3 seeds of reference
##########################################
# start_sing
# ./tensorflow/pytorch_pilot_beta/scripts/dagger_loops.sh

####################################################
# Train and evaluate with different learning rates #
####################################################

# train testing --load_config --checkpoint_path chapter_policy_learning/how_to_recover_normalized/res18_reference_pretrained/final/0 --dataset esatv3_expert/dagger_reference --load_data_in_ram --rammem 7 --max_episodes 15000
  
# First iteration
# for i in 0 1 2 ; do
for i in 1 2 ; do
  train $chapter/$section/baseline_${i}/learning_rates --load_config --checkpoint_path chapter_policy_learning/how_to_recover_normalized/res18_reference_pretrained/final/${i} --dataset esatv3_expert/dagger_reference --load_data_in_ram --rammem 7 --max_episodes 15000
#   train $chapter/$section/model_$i/learning_rates --load_config --checkpoint_path chapter_policy_learning/how_to_recover_normalized/res18_reference_pretrained/final/${i} --dataset chapter_policy_learning/how_to_recover_normalized/dagger_opal/reference/${i}_eva --load_data_in_ram --rammem 7 --max_episodes 15000
done

# Second iteration
# train $chapter/$section/model_0/learning_rates2 --load_config --checkpoint_path $chapter/$section/model_0/learning_rates/4 --dataset chapter_policy_learning/how_to_recover_normalized/dagger_opal/model_0/record --load_data_in_ram --rammem 7 --max_episodes 20000
# train $chapter/$section/model_1/learning_rates2 --load_config --checkpoint_path $chapter/$section/model_1/learning_rates/0 --dataset chapter_policy_learning/how_to_recover_normalized/dagger_opal/model_1/record --load_data_in_ram --rammem 7 --max_episodes 20000
# train $chapter/$section/model_2/learning_rates2 --load_config --checkpoint_path $chapter/$section/model_2/learning_rates/4 --dataset chapter_policy_learning/how_to_recover_normalized/dagger_opal/model_2/record --load_data_in_ram --rammem 7 --max_episodes 20000


sleep 3
condor_q
echo "cd /esat/opal/kkelchte/docker_home/tensorflow/log/$chapter/$section"
