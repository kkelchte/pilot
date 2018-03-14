#!/bin/bash
# Script used to invoke the condor online (singularity) and offline tasks.

# -------OFFLINE-------

# for i in $(seq 3); do
# # for d in canyon_rl_turtle canyon_rl_turtle_150 canyon_rl_turtle_30  canyon_rl_turtle_300 canyon_rl_turtle_600 canyon_rl_turtle_75 ; do
./condor_task_offline.sh -q $((60*60*24)) -e true -n 20 -w "canyon" -t off_depth_turtle/model_canyon_rl_turtle_30_1 -p "--network depth_q_net --dataset canyon_rl_turtle_30"
./condor_task_offline.sh -q $((60*60*24)) -e true -n 20 -w "canyon" -t off_depth_turtle/model_canyon_rl_turtle_30_2 -p "--network depth_q_net --dataset canyon_rl_turtle_30"
./condor_task_offline.sh -q $((60*60*24)) -e true -n 20 -w "canyon" -t off_depth_turtle/model_canyon_rl_turtle_30_b -p "--network depth_q_net --dataset canyon_rl_turtle_30b"
./condor_task_offline.sh -q $((60*60*24)) -e true -n 20 -w "canyon" -t off_depth_turtle/model_canyon_rl_turtle_30_c -p "--network depth_q_net --dataset canyon_rl_turtle_30c"




for i in 1 2 ; do
	for c in 5 6 7 ; do
		./condor_task_offline.sh -q $((60*60*24)) -e true -n 20 -w "canyon" -t off_coll_turtle/model_cl${c}_$i -p "--collision_file collision_info_${c}.txt --normalize_data --random_seed $((i*1354))"
	done
	sleep 1
done	

for i in 1 2 3 ; do
	./condor_task_offline.sh -q $((60*60*24)) -e true -n 20 -w "canyon" -t off_coll_turtle/model_eps05_$i -p "--epsilon_offline --epsilon 0.5 --normalize_data --random_seed $((i*1354))"
	./condor_task_offline.sh -q $((60*60*24)) -e true -n 20 -w "canyon" -t off_coll_turtle/model_ce_$i -p "--loss ce --normalize_data --random_seed $((i*1354))"
	./condor_task_offline.sh -q $((60*60*24)) -e true -n 20 -w "canyon" -t off_coll_turtle/model_ref_$i -p "--normalize_data --random_seed $((i*1354))"
	sleep 1
done

#for i in $(seq 5) ; do
#	./condor_task_offline.sh -q $((60*60*24)) -e true -n 20 -w "canyon" -t off_depth_turtle/model_$i -p "--network depth_q_net --loss absolute --random_seed $((i*1329)) --dataset canyon_rl_turtle_600 --max_episodes 900"
#	sleep 1
#done
# ./condor_task_offline.sh -q $((60*60*24)) -e true -n 20 -w "canyon" -t off_coll_turtle/model_01 -p "--normalize_data True --learning_rate 0.1"
# ./condor_task_offline.sh -q $((60*60*24)) -e true -n 20 -w "canyon" -t off_coll_turtle/model_tf15_001 -p "--max_episodes 1000 --normalize_data True --learning_rate 0.01"
# ./condor_task_offline.sh -q $((60*60*24)) -e true -n 20 -w "canyon" -t off_coll_turtle/model_tf15_0001 -p "--max_episodes 1000 --normalize_data True --learning_rate 0.001"
# ./condor_task_offline.sh -q $((60*60*24)) -e true -n 20 -w "canyon" -t off_coll_turtle/model_00001 -p "--normalize_data True --learning_rate 0.0001"

# ./condor_task_offline.sh -q $((60*60*24)) -e true -n 20 -w "canyon" -t off_depth_turtle/model_tf15 -p "--max_episodes 1000 --network depth_q_net --learning_rate 0.01 --loss absolute"

# ./condor_task_offline.sh -q $((60*60*24*2)) -t off_depth_drone/model_001 -p "--max_episodes 10000 --network depth_q_net --dataset canyon_rl --learning_rate 0.01 --loss absolute --min_depth 0.001 --max_depth 2.0"
# ./condor_task_offline.sh -q $((60*60*24*2)) -t off_depth_drone/model_0001 -p "--max_episodes 10000 --network depth_q_net --dataset canyon_rl --learning_rate 0.001 --loss absolute --min_depth 0.001 --max_depth 2.0"


# # -------CREATE DATASET------------
# ME=105
# WT=$((3*60*60))
# for i in $(seq 10) ; do
# 	echo $i
# 	./condor_task_sing.sh -t rec_$i -q $WT -s create_data_turtle.sh -n $ME -w "canyon" -p "--random_seed $((i*1234)) --epsilon 1 --epsilon_decay 0"
# 	sleep 1
# done


# -------ONLINE---------
# for i in $(seq 3); do
# 	./condor_task_sing.sh -q $((60*60*24*3)) -t on_coll_turtle/model_straight_default_$i -s train_model_turtle.sh -n 10000 -p "--epsilon 0.5"
# 	# ./condor_task_sing.sh -q $((60*60*24*3)) -t on_depth_turtle/model_001_$i -s train_model_turtle.sh -n 10000 -p "--epsilon 0.5 --network depth_q_net --loss absolute"
# done

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
