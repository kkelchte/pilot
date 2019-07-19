#!/bin/bash
chapter=chapter_policy_learning
section=recover
pytorch_args="--network tinyv3_nfc_net --n_frames 2 --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --scaled_input\
 --batch_size 32 --loss MSE --optimizer SGD --clip 1.0 --weight_decay 0 --normalized_output"




echo "####### chapter: $chapter #######"
echo "####### section: $section #######"

pretrain(){
  cd ..
  condor_args_pretraining="--wall_time $((24*60*60)) --gpumem 1900"
  python dag_train.py $pytorch_args $condor_args_pretraining -t $*
  cd launch
}

train(){
  cd ..
  script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 "
  condor_args="--wall_time_train $((24*60*60)) --wall_time_eva $((3*3600)) --rammem 7 --gpumem_train 1800 --gpumem_eva 1800  --python_project pytorch_pilot_beta/pilot"
  dag_args="--model_names $(seq 0 2) --random_numbers $(seq 56431 56441)"
  python dag_train_and_evaluate.py $pytorch_args $condor_args $dag_args $script_args -t $*
  cd launch
}

##########################################
# Pretrain for different learning rates
##########################################


for i in 2 3 4 5 10 20; do
pretrain $chapter/$section/reference_shifted_$i/learning_rates --dataset esatv3_expert/recovery_reference --load_data_in_ram --rammem 7 --max_episodes 10000 --shift_control_indices $i --python_project pytorch_pilot_beta/pilot
done
# pretrain $chapter/$section/recovery/learning_rates --dataset esatv3_recovery --load_data_in_ram --rammem 15 --max_episodes 10000
# for noise in gau ou uni ; do
#   pretrain $chapter/$section/noise_shifted_2/$noise/learning_rates --dataset esatv3_expert_stochastic/$noise --load_data_in_ram --rammem 7 --max_episodes 10000 --shift_control_indices 2 --python_project pytorch_pilot_beta/pilot
# done

##############################
# Set winning learning rate
##############################
for i in 2 3 4 5 10 20; do
train $chapter/$section/reference_shifted_$i/final --dataset esatv3_expert/recovery_reference --load_data_in_ram --rammem 7 --max_episodes 10000 --shift_control_indices $i --python_project pytorch_pilot_beta/pilot --learning_rate 0.001
done
# train $chapter/$section/reference_shifted_1/final --dataset esatv3_expert/recovery_reference --load_data_in_ram --rammem 7 --max_episodes 10000 --learning_rate 0.001 --shift_control_indices 1
# train $chapter/$section/recovery_shifted/final --dataset esatv3_recovery --load_data_in_ram --rammem 15 --max_episodes 10000 --learning_rate 0.01
#train $chapter/$section/noise_shifted_2/uni/final --dataset esatv3_expert_stochastic/uni --load_data_in_ram --rammem 7 --max_episodes 10000 --learning_rate 0.001 --shift_control_indices 2
#train $chapter/$section/noise_shifted_2/gau/final --dataset esatv3_expert_stochastic/gau --load_data_in_ram --rammem 7 --max_episodes 10000 --learning_rate 0.01 --shift_control_indices 2
#train $chapter/$section/noise_shifted_2/ou/final --dataset esatv3_expert_stochastic/ou --load_data_in_ram --rammem 7 --max_episodes 10000 --learning_rate 0.001 --shift_control_indices 2

##############################
# DAGGER
##############################
# cd ..
# for seed in 0 1 2 ; do
#   name="$chapter/$section/DAGGER/$seed"
#   local_pytorch_args="--load_config --checkpoint_path $chapter/$section/res18_reference/final/0"
#   script_args="--evaluation --pause_simulator -ds --z_pos 1 -w esatv3 --random_seed $((seed*32+512)) --number_of_runs 1 --final_evaluation_runs 0 --python_project pytorch_pilot/pilot"
#   condor_args="--wall_time $((24*60)) --use_greenlist --cpus 16"
#   python condor_online.py -t $name $pytorch_args $local_pytorch_args $script_args $condor_args
# done
# cd launch
##############################
# ON-POLICY
##############################
# cd ..
# # Should be able to make this more 'offline' --> implement at online.py
# for seed in 0 1 2 ; do
#   name="$chapter/$section/on-policy/$seed"
#   local_pytorch_args="--on-policy --min_buffer_size 1000 --buffer_size 10000"
#   script_args="--pause_simulator --z_pos 1 -w esatv3 --random_seed $((seed*32+512)) --number_of_runs 100 --final_evaluation_runs 5 --python_project pytorch_pilot/pilot"
#   condor_args="--wall_time $((2*24*60)) --use_greenlist --cpus 16 --gpumem 5000"
#   python condor_online.py -t $name $pytorch_args $local_pytorch_args $script_args $condor_args
# done
# cd launch
##############################
# Create datasets
##############################

# recovery
# name="data_collection/recovery_expert"
# pytorch_args="--alpha 1 --pause_simulator --speed 0.8 --turn_speed 0.8 --action_bound 0.9"
# script_args=" --recovery -ds --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 3 --evaluation --final_evaluation_runs 0 --python_project pytorch_pilot/pilot"
# condor_args="--wall_time $((2*10*60+30*60)) --use_greenlist --cpus 16"
# python condor_online.py -t $name $pytorch_args $script_args $condor_args


# stochastic
# cd ..
# for sigma in 9 5 1 ; do
#   name="data_collection/stochastic_expert/gau/$sigma"
#   pytorch_args="--noise gau --alpha 1 --sigma_yaw 0.$sigma --pause_simulator --speed 0.8 --turn_speed 0.8 --action_bound 0.9"
#   script_args=" -ds --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 2 --evaluation --final_evaluation_runs 0 --python_project pytorch_pilot_beta/pilot"
#   condor_args="--wall_time $((2*10*60+30*60)) --use_greenlist --cpus 16"
#   python condor_online.py -t $name $pytorch_args $script_args $condor_args
# done
# for sigma in 9 5 1 ; do
#   name="data_collection/stochastic_expert/uni/$sigma"
#   pytorch_args="--noise uni --alpha 1 --sigma_yaw 0.$sigma --pause_simulator --speed 0.8 --turn_speed 0.8 --action_bound 0.9"
#   script_args=" -ds --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 2 --evaluation --final_evaluation_runs 0 --python_project pytorch_pilot_beta/pilot"
#   condor_args="--wall_time $((2*10*60+30*60)) --use_greenlist --cpus 16"
#   python condor_online.py -t $name $pytorch_args $script_args $condor_args
# done
# for sigma in 9 5 1 ; do
#   name="data_collection/stochastic_expert/ou/$sigma"
#   pytorch_args="--noise ou --alpha 1 --sigma_yaw 0.$sigma --pause_simulator --speed 0.8 --turn_speed 0.8 --action_bound 0.9"
#   script_args=" -ds --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 2 --evaluation --final_evaluation_runs 0 --python_project pytorch_pilot_beta/pilot"
#   condor_args="--wall_time $((2*10*60+30*60)) --use_greenlist --cpus 16"
#   python condor_online.py -t $name $pytorch_args $script_args $condor_args
# done
# cd launch




sleep 3
condor_q
echo "cd /esat/opal/kkelchte/docker_home/tensorflow/log/$chapter/$section"
