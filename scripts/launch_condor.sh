#!/bin/bash
# Script used to invoke the condor online (singularity) and offline tasks.

# -------OFFLINE-------
#for i in $(seq 5) ; do 
# 	echo $i 
#	./condor_task_offline_3_q.sh -q $((60*60*5)) -t three_q/model_$i -m three_q/model_$i -p "--continue_training True --discrete True --max_episodes 160 --num_outputs 3 --dataset canyon --random_seed $((3000*$i+1539)) --network three_q_net --depth_q_learning True --subsample 1"
	
# 	# ./condor_task_offline_naive_q.sh -q $((60*60*5)) -t naive_q/model_$i -m mobilenet_025 -p "--continue_training False --dataset canyon --random_seed $((3000*$i+1539)) --network naive_q_net --naive_q_learning True" 
#done

# -------CREATE DATASET------------
ME=105
WT=$((3*60*60))
for i in $(seq 10) ; do
	echo $i
	./condor_task_sing.sh -t rec_$i -q $WT -s create_data_turtle.sh -n $ME -w "canyon" -p "--random_seed $((i*1234)) --epsilon 1 --epsilon_decay 0"
	sleep 1
done

# -------Continue Training online with prefill replay buffer------------
# ME=1000
# WT=$((2*24*60*60))
# LR=0.01
# for i in $(seq 3) ; do	
# 	./condor_task_sing.sh -t onl_can_coll_cont_$i  -q $WT -s train_model.sh -n $ME -w "canyon" -m off_can_coll_cont -p "--prefill True --continue_training True --normalized_replay True --random_seed $((i*1234)) --load_config True"
# 	# ./condor_task_sing.sh -t onl_can_depth_cont_$i  -q $WT -s train_model.sh -n $ME -w "canyon" -m off_can_depth_cont -p "--prefill True --continue_training True --normalized_replay True --random_seed $((i*1234)) --load_config True"
# done

# -------GRIDSEARCH-------
# echo "| i | LR | EPSILON | BFS | NUM | TYPE |" > /esat/opal/kkelchte/docker_home/tensorflow/log/gridsearchtags
# echo "|-|-|-|-|-| " >> /esat/opal/kkelchte/docker_home/tensorflow/log/gridsearchtags
# i=0
# ME=20000
# WT=$((3*24*60*60))
# BFS=1000
# EPSILON_DECAY=0.001
# EPSILON=0.5
# LR=0.001
# i=0
# for i in $(seq 3) ; do
# 	./condor_task_sing.sh -t coll_q_$i   -q $WT -s train_model.sh -n $ME -w "canyon" -p "--normalized_replay True --buffer_size $BFS --learning_rate $LR --random_seed $((i*1234)) --epsilon $EPSILON --epsilon_decay $EPSILON_DECAY"
# 	./condor_task_sing.sh -t depth_q_$i   -q $WT -s train_model.sh -n $ME -w "canyon" -p "--network depth_q_net --normalized_replay True --buffer_size $BFS --learning_rate $LR --random_seed $((i*1234)) --epsilon $EPSILON --epsilon_decay $EPSILON_DECAY"
# done

# ------------DOSHICO-------------
# for i in $(seq 0 55) ; do
# # # for i in 0 ; do
# 	WT=$((3*60*60*4))
# # 	# ME=$((3*5*100))
#  	./condor_task_offline.sh -q $WT -t doshico_auxd_gm0001/doshico_$i -m mobilenet_025 -e true -n 20 -w "esat_v1 esat_v2" -p "--grad_mul_weight 0.001 --batch_size 64 --max_episodes 20 --learning_rate 0.1 --dataset overview --random_seed $((3000*$i+1539)) --n_fc True --auxiliary_depth True"
# 	./condor_task_offline.sh -q $WT -t doshico_naux_gm0001/doshico_$i -m mobilenet_025 -e true -n 20 -w "esat_v1 esat_v2" -p "--grad_mul_weight 0.001 --batch_size 64 --max_episodes 20 --learning_rate 0.1 --dataset overview --random_seed $((3000*$i+1539)) --n_fc True --auxiliary_depth False"
#         ./condor_task_offline.sh -q $WT -t doshico_fc_gm0001/doshico_$i -m mobilenet_025 -e true -n 20 -w "esat_v1 esat_v2" -p "--grad_mul_weight 0.001 --batch_size 64 --max_episodes 20 --learning_rate 0.1 --dataset overview --random_seed $((3000*$i+1539)) --n_fc False --auxiliary_depth False"
# #       ./condor_task_offline.sh -q $WT -t doshico_auxd_gm01/doshico_$i -m mobilenet_025 -e true -n 20 -w "esat_v1 esat_v2" -p "--grad_mul_weight 0.1 --batch_size 64 --max_episodes 20 --learning_rate 0.1 --dataset overview --random_seed $((3000*$i+1539)) --n_fc True --auxiliary_depth True"
# #	./condor_task_offline.sh -q $WT -t doshico_auxd_gm05/doshico_$i -m mobilenet_025 -e true -n 20 -w "esat_v1 esat_v2" -p "--grad_mul_weight 0.5 --batch_size 64 --max_episodes 20 --learning_rate 0.1 --dataset overview --random_seed $((3000*$i+1539)) --n_fc True --auxiliary_depth True"
# # 	./condor_task_offline.sh -q $WT -t doshico_naux/doshico_$i -m mobilenet_025 -e true -n 20 -w "esat_v1 esat_v2" -p "--batch_size 64 --max_episodes 34 --learning_rate 0.1 --dataset overview --random_seed $((3000*$i+1539)) --n_fc True" 
# # 	./condor_task_offline.sh -q $WT -t doshico_fc/doshico_$i -m mobilenet_025 -e true -n 20 -w "esat_v1 esat_v2" -p "--batch_size 64 --max_episodes 50 --learning_rate 0.1 --dataset overview --random_seed $((3000*$i+1539))" 
# done

watch condor_q
