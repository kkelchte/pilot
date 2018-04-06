#!/bin/bash
# Script used to invoke the condor online (singularity) and offline tasks.

# -------OFFLINE-------

# -------DIFFERENT DATASETS
# for i in $(seq 3); do
# 	d="canyon_rl_turtle_collision_free_epsilon03"
# 	# for d in canyon_rl_turtle canyon_rl_turtle_150 canyon_rl_turtle_30  canyon_rl_turtle_300 canyon_rl_turtle_600 canyon_rl_turtle_75 ; do
# 		./condor_task_offline.sh -q $((60*60*24)) -t off_depth_turtle/model_coll_free_eps03_${i}  -e true -n 20 -w "canyon" -p "--dataset $d --random_seed $((i*1354)) --loss absolute --network depth_q_net --max_episodes 800"	
# # # 		# ./condor_task_offline.sh -q $((60*60*24)) -e true -n 20 -w "canyon" -t off_coll_turtle/model_${d}_$i -p "--dataset $d --random_seed $((i*1354)) --loss ce --collision_file collision_info_7.txt --normalize_data"
# 		sleep 1
#  	# done
# done

# -------COLLISION_FREE DATASET
#for i in $(seq 3); do
#	./condor_task_offline.sh -q $((60*60*24)) -e true -n 20 -w "canyon" -t off_depth_turtle/model_colfree_${i} -p "--dataset canyon_rl_turtle_collision_free --random_seed $((i*1354)) --loss absolute --network depth_q_net --max_episodes 1000"	
#done

# -------CLIP COLLISIONS FROM RANDOM DATASET
# for i in $(seq 1) ; do
# 	./condor_task_offline.sh -q $((60*60*24)) -e true -n 20 -w "canyon" -t off_depth_turtle/model_clipcoll_200n_$i -p "--network depth_q_net --loss absolute --random_seed $((i*1329)) --dataset canyon_rl_turtle_clip_collision --learning_rate 0.1 --max_episodes 200"
# 	./condor_task_offline.sh -q $((60*60*24)) -e true -n 20 -w "canyon" -t off_depth_turtle/model_clipcoll_400n_$i -p "--network depth_q_net --loss absolute --random_seed $((i*1329)) --dataset canyon_rl_turtle_clip_collision --learning_rate 0.1 --max_episodes 400"
# 	./condor_task_offline.sh -q $((60*60*24)) -e true -n 20 -w "canyon" -t off_depth_turtle/model_clipcoll_800n_$i -p "--network depth_q_net --loss absolute --random_seed $((i*1329)) --dataset canyon_rl_turtle_clip_collision --learning_rate 0.1 --max_episodes 800"
# 	sleep 1
# done

# -------NO DATA NORMALIZATION
# for i in $(seq 3); do
# 	./condor_task_offline.sh -q $((60*60*24)) -e true -n 20 -w "canyon" -t off_coll_turtle/model_nonorm_$i -p "--dataset canyon_rl_turtle --random_seed $((i*1354)) --loss ce --collision_file collision_info_7.txt"
# 	sleep 1
# done


# # -------CREATE DATASET------------
# ME=12
# WT=$((2*60*60))
# for i in $(seq 12) ; do
# 	echo $i
# 	./condor_task_sing.sh -t rec_$i -q $WT -s create_data_turtle.sh -n $ME -m off_depth_turtle/turtle_depth_abs_clip0001-2_001 -w "canyon" -p "--action_quantity 9 --action_smoothing --epsilon 0.3"
# 	sleep 1
# done


# -------ONLINE---------
for i in $(seq 1); do
	# for i in 2; do
	# ./condor_task_sing.sh -q $((60*60*30)) -t on_depth_turtle/model_eps03_$i -s train_model_turtle.sh -n 800 -p "--epsilon 0.3 --random_seed $((i*1354)) --network depth_q_net --loss absolute --learning_rate 0.1 --buffer_size 5000"
	# ./condor_task_sing.sh -q $((60*60*30)) -t on_depth_turtle/model_eps05_$i -s train_model_turtle.sh -n 800 -p "--epsilon 0.5 --random_seed $((i*1354)) --network depth_q_net --loss absolute --learning_rate 0.1 --buffer_size 5000"
	./condor_task_sing.sh -q $((60*60*50)) -t on_depth_turtle/model_ref5000_$i -s train_model_turtle.sh -n 800 -p "--random_seed $((i*1354)) --network depth_q_net --loss absolute --learning_rate 0.1 --buffer_size 5000"
	./condor_task_sing.sh -q $((60*60*50)) -t on_depth_turtle/model_sttvar_$i -s train_model_turtle.sh -n 800 -p "--replay_priority state_variance --random_seed $((i*1354)) --network depth_q_net --loss absolute --learning_rate 0.1 --buffer_size 5000"
	./condor_task_sing.sh -q $((60*60*50)) -t on_depth_turtle/model_sttvar_priokeep_$i -s train_model_turtle.sh -n 800 -p "--prioritized_keeping --replay_priority state_variance --random_seed $((i*1354)) --network depth_q_net --loss absolute --learning_rate 0.1 --buffer_size 5000"
	./condor_task_sing.sh -q $((60*60*50)) -t on_depth_turtle/model_actvar_$i -s train_model_turtle.sh -n 800 -p "--replay_priority action_variance --random_seed $((i*1354)) --network depth_q_net --loss absolute --learning_rate 0.1 --buffer_size 5000"
	./condor_task_sing.sh -q $((60*60*50)) -t on_depth_turtle/model_trgtvar_$i -s train_model_turtle.sh -n 800 -p "--replay_priority trgt_variance --random_seed $((i*1354)) --network depth_q_net --loss absolute --learning_rate 0.1 --buffer_size 5000"
	./condor_task_sing.sh -q $((60*60*50)) -t on_depth_turtle/model_rndact_$i -s train_model_turtle.sh -n 800 -p "--replay_priority random_action --random_seed $((i*1354)) --network depth_q_net --loss absolute --learning_rate 0.1 --buffer_size 5000"
	./condor_task_sing.sh -q $((60*60*50)) -t on_depth_turtle/model_rndact_priokeep_$i -s train_model_turtle.sh -n 800 -p "--prioritized_keeping --replay_priority random_action --random_seed $((i*1354)) --network depth_q_net --loss absolute --learning_rate 0.1 --buffer_size 5000"
done


# for i in canyon_rl_turtle_30_2 ; do 
# ./condor_task_sing.sh -q $((60*60*24*3)) -m off_coll_turtle/model_$i  -t off_coll_turtle/model_$i -s evaluate_model_turtle.sh -n 20
# done
# for i in canyon_rl_turtle_30_1 canyon_rl_turtle_300_1 canyon_rl_turtle_600_1 canyon_rl_turtle_30_2 canyon_rl_turtle_1 ; do
# 	./condor_task_sing.sh -q $((60*60*24*3)) -m off_depth_turtle/model_$i -t off_depth_turtle/model_$i -s evaluate_model_turtle.sh -n 20
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
