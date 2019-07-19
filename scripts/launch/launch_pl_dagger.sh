#!/bin/bash
chapter=chapter_policy_learning
section=DAGGER
pytorch_args="--network tinyv3_nfc_net --n_frames 2 --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --scaled_input\
 --batch_size 32 --loss MSE --optimizer SGD --clip 1.0 --weight_decay 0 --normalized_output"

echo "####### chapter: $chapter #######"
echo "####### section: $section #######"

train(){
  cd ..
  script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 "
  condor_args="--wall_time_train $((24*60*60)) --wall_time_eva $((3*3600)) --rammem 7 --gpumem_train 1800 --gpumem_eva 1800"
  dag_args="--model_names lr_01 lr_001 lr_0001 lr_00001 lr_000001 --random_numbers 123 --learning_rates 0.1 0.01 0.001 0.0001 0.00001"
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



# First iteration
for i in 0 1 2 ; do
  train $chapter/$section/baseline_${i}/iteration1 --load_config --checkpoint_path chapter_policy_learning/recover/reference/final/${i} --dataset esatv3_expert/dagger_12K --load_data_in_ram --rammem 7 --max_episodes 15000
  train $chapter/$section/model_$i/iteration1 --load_config --checkpoint_path chapter_policy_learning/recover/reference/final/${i} --control_file supervised_info.txt --dataset chapter_policy_learning/dagger/reference/${i}_eva --load_data_in_ram --rammem 7 --max_episodes 15000
done

# # Second iteration
# for i in 0 1 2 ; do
#   train $chapter/$section/model_${i}/iteration2 --load_config --checkpoint_path $chapter/$section/model_${i}/iteration1/0 --control_file supervised_info.txt  --dataset chapter_policy_learning/dagger/model_${i}/record --load_data_in_ram --rammem 7 --max_episodes 20000
#   train $chapter/$section/baseline_${i}/iteration2 --load_config --checkpoint_path $chapter/$section/baseline_${i}/iteration1/0 --dataset esatv3_expert/dagger_15K --load_data_in_ram --rammem 7 --max_episodes 20000
# done



# Third iteration
# for i in 0 1 2 ; do
#   train $chapter/$section/model_${i}/iteration3 --load_config --checkpoint_path $chapter/$section/model_${i}/iteration2/0 --control_file supervised_info.txt  --dataset chapter_policy_learning/dagger/model_${i}/record_2 --load_data_in_ram --rammem 7 --max_episodes 25000
#   train $chapter/$section/baseline_${i}/iteration3 --load_config --checkpoint_path $chapter/$section/baseline_${i}/iteration2/0 --dataset esatv3_expert/dagger_17K --load_data_in_ram --rammem 7 --max_episodes 25000
# done


sleep 3
condor_q
echo "cd /esat/opal/kkelchte/docker_home/tensorflow/log/$chapter/$section"
