#!/bin/bash
chapter=chapter_policy_learning
section=how_to_recover
pytorch_args="--network res18_net --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --scaled_input\
 --max_episodes 10000 --batch_size 32 --loss MSE --optimizer SGD --clip 1.0 --weight_decay 0"

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
  condor_args="--wall_time_train $((24*60*60)) --wall_time_eva $((3*3600)) --rammem 7 --gpumem_train 1800 --gpumem_eva 1800"
  dag_args="--model_names $(seq 0 2) --random_numbers $(seq 56431 56441)"
  python dag_train_and_evaluate.py $pytorch_args $condor_args $dag_args $script_args -t $*
  cd launch
}

##########################################
# Pretrain for different learning rates
##########################################


pretrain $chapter/$section/res18_reference/learning_rates --dataset esatv3_expert/recovery_reference --load_data_in_ram --rammem 5
pretrain $chapter/$section/res18_reference_pretrained/learning_rates --dataset esatv3_expert/recovery_reference --load_data_in_ram --rammem 5 --pretrained

pretrain $chapter/$section/res18_recovery/learning_rates --dataset esatv3_recovery --load_data_in_ram --rammem 7
pretrain $chapter/$section/res18_recovery_pretrained/learning_rates --dataset esatv3_recovery --load_data_in_ram --rammem 7 --pretrained

for noise in gau ou uni ; do
  pretrain $chapter/$section/res18_noise_pretrained/$noise/learning_rates --dataset esatv3_expert_stochastic/$noise --load_data_in_ram --rammem 7 --pretrained
  pretrain $chapter/$section/res18_noise/$noise/learning_rates --dataset esatv3_expert_stochastic/$noise --load_data_in_ram --rammem 7
done

##############################
# Set winning learning rate
##############################

# train $chapter/$section/res18_reference/final --dataset esatv3_expert/2500 --load_data_in_ram --rammem 5 --learning_rate 0.001 
# train $chapter/$section/res18_reference_pretrained/final --dataset esatv3_expert/2500 --load_data_in_ram --rammem 5 --pretrained --learning_rate 0.01

# train $chapter/$section/res18_recovery/redo --dataset esatv3_recovery --load_data_in_ram --rammem 5 --learning_rate 0.1
# train $chapter/$section/res18_recovery/final --dataset esatv3_recovery --load_data_in_ram --rammem 7 --learning_rate 0.001

# train $chapter/$section/res18_noise/gau/final --dataset esatv3_expert_stochastic/gau --load_data_in_ram --rammem 7 --learning_rate 0.1 --python_project pytorch_pilot_beta/pilot
# train $chapter/$section/res18_noise/ou/final --dataset esatv3_expert_stochastic/ou --load_data_in_ram --rammem 7 --learning_rate 0.1
# train $chapter/$section/res18_noise/uni/final --dataset esatv3_expert_stochastic/uni --load_data_in_ram --rammem 7 --learning_rate 0.1

##############################
# Create datasets
##############################

# recovery
# name="data_collection/recovery_expert"
# pytorch_args="--alpha 1 --pause_simulator --speed 0.8 --turn_speed 0.8 --action_bound 0.9"
# script_args=" --recovery -ds --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 3 --evaluation --final_evaluation_runs 0 --python_project pytorch_pilot_beta/pilot"
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
