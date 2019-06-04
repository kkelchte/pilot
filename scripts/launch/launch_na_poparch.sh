#!/bin/bash
chapter=chapter_neural_architectures
section=popular_architectures
pytorch_args="--dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --normalized_output\
 --pretrained  --max_episodes 10000 --batch_size 32 --loss CrossEntropy --clip 1 --weight_decay 0"

echo "####### chapter: $chapter #######"
echo "####### section: $section #######"

pretrain(){
  cd ..
  condor_args_pretraining="--wall_time $((24*60*60)) --rammem 7 --copy_dataset"
  python dag_train.py $pytorch_args $condor_args_pretraining -t $*
  cd launch
}
train(){
  cd ..
  script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --evaluation"
  condor_args="--wall_time_train $((24*60*60)) --wall_time_eva $((4*3600)) --rammem 7 --copy_dataset"
  dag_args="--model_names 0 1 2 --random_numbers 123 456 789"
  python dag_train_and_evaluate.py $pytorch_args $condor_args $dag_args $script_args -t $*
  cd launch
}

#######################################
# Pretrain for different learning rates
#######################################

# for AR in inception_net dense_net vgg16_net squeeze_net ; do
#   pretrain $chapter/$section/${AR}_finetuned/learning_rates --network $AR --gpumem 6000 --feature_extract --optimizer SGD --scaled_input 
#   pretrain $chapter/$section/${AR}_end-to-end/learning_rates --network $AR --gpumem 6000                  --optimizer SGD --scaled_input 
# done
# for AR in res18_net alex_net ; do
#   pretrain $chapter/$section/${AR}_finetuned/learning_rates --network $AR --gpumem 1900 --feature_extract --optimizer SGD --scaled_input 
#   pretrain $chapter/$section/${AR}_end-to-end/learning_rates --network $AR --gpumem 1900                  --optimizer SGD --scaled_input 
# done

#######################################
# Set winning learning rate
#######################################

# for AR in inception_net  vgg16_net squeeze_net ; do
for AR in inception_net ; do
  train $chapter/$section/${AR}_finetuned/final --network $AR --gpumem_train 6000 --gpumem_eva 6000 --feature_extract --optimizer SGD --scaled_input --learning_rate 0.001
  train $chapter/$section/${AR}_end-to-end/final --network $AR --gpumem_train 6000 --gpumem_eva 6000                  --optimizer SGD --scaled_input --learning_rate 0.01
done
for AR in dense_net ; do
  train $chapter/$section/${AR}_finetuned/final --network $AR --gpumem_train 6000 --gpumem_eva 6000 --feature_extract --optimizer SGD --scaled_input --learning_rate 0.01
  train $chapter/$section/${AR}_end-to-end/final --network $AR --gpumem_train 6000 --gpumem_eva 6000                  --optimizer SGD --scaled_input --learning_rate 0.01
done
for AR in vgg16_net squeeze_net ; do
  train $chapter/$section/${AR}_finetuned/final --network $AR --gpumem_train 6000 --gpumem_eva 6000 --feature_extract --optimizer SGD --scaled_input --learning_rate 0.1
  train $chapter/$section/${AR}_end-to-end/final --network $AR --gpumem_train 6000 --gpumem_eva 6000                  --optimizer SGD --scaled_input --learning_rate 0.01
done
for AR in res18_net ; do
  train $chapter/$section/${AR}_finetuned/final --network $AR --gpumem_train 1900 --gpumem_eva 1900 --feature_extract --optimizer SGD --scaled_input --learning_rate 0.01
  train $chapter/$section/${AR}_end-to-end/final --network $AR --gpumem_train 1900 --gpumem_eva 1900                  --optimizer SGD --scaled_input --learning_rate 0.01
done
for AR in alex_net ; do
  train $chapter/$section/${AR}_finetuned/final --network $AR --gpumem_train 1900 --gpumem_eva 1900 --feature_extract --optimizer SGD --scaled_input --learning_rate 0.01
  train $chapter/$section/${AR}_end-to-end/final --network $AR --gpumem_train 1900 --gpumem_eva 1900                  --optimizer SGD --scaled_input --learning_rate 0.1
done


sleep 3
condor_q
echo "cd /esat/opal/kkelchte/docker_home/tensorflow/log/$chapter/$section"