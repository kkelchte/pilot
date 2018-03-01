#!/bin/bash
# Script used to invoke the condor online (singularity) and offline tasks.

# -------OFFLINE-------
#for i in $(seq 5) ; do 
# 	echo $i 
#	./condor_task_offline_3_q.sh -q $((60*60*5)) -t three_q/model_$i -m three_q/model_$i -p "--continue_training True --discrete True --max_episodes 160 --num_outputs 3 --dataset canyon --random_seed $((3000*$i+1539)) --network three_q_net --depth_q_learning True --subsample 1"
	
# 	# ./condor_task_offline_naive_q.sh -q $((60*60*5)) -t naive_q/model_$i -m mobilenet_025 -p "--continue_training False --dataset canyon --random_seed $((3000*$i+1539)) --network naive_q_net --naive_q_learning True" 
#done

# -------CREATE DATASET------------

# ME=105
# WT=$((3*60*60))
# for  i in 10 11 12 15 16 17 18 2 22 23 25 26 27 28 3 30 31 32 33 34 36 37 38 39 4 40 41 42 43 46 48 49 5 50 51 52 56 58 59 6 65 66 67 68 7 73 76 79 82 85 87 88 9 91 93 98 ; do
# 	echo $i
# 	./condor_task_sing.sh -t rec_$i  -q $WT -s create_data.sh -m mobilenet_025 -n $ME -w "canyon" -p "--continue_training False --random_seed $((i*1234)) --epsilon 1"
# 	sleep 1
# done

# -------Continue Training online with prefill replay buffer------------

ME=1000
WT=$((2*24*60*60))
LR=0.01
for i in $(seq 3) ; do	
	./condor_task_sing.sh -t onl_can_coll_cont_$i  -q $WT -s train_model.sh -n $ME -w "canyon" -m off_can_coll_cont -p "--prefill True --continue_training True --normalized_replay True --random_seed $((i*1234)) --load_config True"
	# ./condor_task_sing.sh -t onl_can_depth_cont_$i  -q $WT -s train_model.sh -n $ME -w "canyon" -m off_can_depth_cont -p "--prefill True --continue_training True --normalized_replay True --random_seed $((i*1234)) --load_config True"
done


# -------GRIDSEARCH-------
# -------Reinforcement Learning --------


# ME=1000
# WT=$((24*60*60))
# LR=0.05
# BFS=10000
# for i in $(seq 1) ; do	
# 	# ./condor_task_sing.sh -t coll_norm_$i  -q $WT -s train_model.sh -n $ME -w "canyon" -p "--learning_rate $LR --random_seed 1234 --epsilon 0.1 --buffer_size $BFS --action_amplitude 1 --normalized_replay True"
# 	./condor_task_sing.sh -t coll_ep0_$i  -q $WT -s train_model.sh -n $ME -w "canyon" -p "--normalized_replay True --batch_size 64 --learning_rate $LR --random_seed $((i*1234)) --epsilon 0. --buffer_size $BFS --action_amplitude 1"
# 	# ./condor_task_sing.sh -t depth_norm_$i  -q $WT -s train_model.sh -n $ME -w "canyon" -p " --network depth_q_net --learning_rate $LR --random_seed 1234 --epsilon 0.1 --buffer_size $BFS --action_amplitude 1 --normalized_replay True"
# 	./condor_task_sing.sh -t depth_ep0_$i  -q $WT -s train_model.sh -n $ME -w "canyon" -p "--normalized_replay True --batch_size 64 --network depth_q_net --learning_rate $LR --random_seed $((i*1234)) --epsilon 0. --buffer_size $BFS --action_amplitude 1"
# done

# echo "| i | EPSILON | AMPL | NUM |" > /esat/opal/kkelchte/docker_home/tensorflow/log/gridsearchtags
# echo "|-|-|-|-|-| " >> /esat/opal/kkelchte/docker_home/tensorflow/log/gridsearchtags
# i=0
# ME=1000
# WT=$((3*24*60*60))
# LR=0.05
# BFS=10000
# ./condor_task_sing.sh -t test_coll_q  -q $WT -s train_model.sh -n $ME -w "canyon" -p "--learning_rate $LR --random_seed $((i*1234)) --epsilon 0.5"
# for k in 0 1 2 ; do
# 	for EPSILON in 0.5 0.1 ; do
#                 for AMPL in 1 5 ; do
#                                 echo "| ${i} | $EPSILON | $AMPL | $k |" >> /esat/opal/kkelchte/docker_home/tensorflow/log/gridsearchtags
#                                 ./condor_task_sing.sh -t gridsearch_coll_q_$i -q $WT -s train_model.sh -n $ME -w "canyon" -p "--buffer_size $BFS --learning_rate $LR --random_seed $((i*1234)) --epsilon $EPSILON --action_amplitude $AMPL"
#                                 ./condor_task_sing.sh -t gridsearch_depth_q_$i -q $WT -s train_model.sh -n $ME -w "canyon" -p "--buffer_size $BFS --network depth_q_net --learning_rate $LR --random_seed $((i*1234)) --epsilon $EPSILON --action_amplitude $AMPL"
#                                 i=$((i+1))
#                                 sleep 1
#                 done
#         done
# done



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

# k=0
# # ./condor_task_sing.sh -t test_coll_q	-q $WT -s train_model.sh -n $ME -w "canyon" -p "--learning_rate $LR --random_seed $((i*1234)) --epsilon 0.5"
# # for k in 0 1 ; do
# 	for BFS in 1000 10000 ; do
# 		for EPSILON in 0.5 0.1 0.01 ; do
# 			for LR in 0.1 0.01 0.001 ; do
# 				echo "| ${i} | $LR | $EPSILON | $BFS | $k | coll |" >> /esat/opal/kkelchte/docker_home/tensorflow/log/gridsearchtags
# 				./condor_task_sing.sh -t gridsearch_coll_q_$i	-q $WT -s train_model.sh -n $ME -w "canyon" -p "--buffer_size $BFS --learning_rate $LR --random_seed $((i*1234)) --epsilon $EPSILON --epsilon_decay $EPSILON_DECAY"
# 				i=$((i+1))
# 				sleep 3
# 			done
# 		done
# 	done
# # done


# # for k in 0 1 ; do
# 	for BFS in 1000 10000 ; do
# 		for EPSILON in 0.5 0.1 0.01 ; do
# 			for LR in 0.1 0.01 0.001 ; do
# 				echo "| ${i} | $LR | $EPSILON | $BFS | $k | depth |" >> /esat/opal/kkelchte/docker_home/tensorflow/log/gridsearchtags
# 				./condor_task_sing.sh -t gridsearch_depth_q_$i	-q $WT -s train_model.sh -n $ME -w "canyon" -p "--buffer_size $BFS --network depth_q_net --learning_rate $LR --random_seed $((i*1234)) --epsilon $EPSILON --epsilon_decay $EPSILON_DECAY"
# 				i=$((i+1))
# 				sleep 3
# 			done
# 		done
# 	done
# # done




# i=0
# DO=0.5
# WD=4
# echo "| i | LR | BS | GM | " > /esat/opal/kkelchte/docker_home/tensorflow/log/gridsearchtags
# echo "|-|-|-|-| " > /esat/opal/kkelchte/docker_home/tensorflow/log/gridsearchtags
# for GM in 0. 0.01 0.1 ; do
# 	for BS in 32 ; do
# 		for DO in 0.25 0.5 0.75 ; do
# 			for LR in 0.5 0.1 0.05 ; do
# 					echo "${i};$LR;$BS;$DO;$GM" >> /esat/opal/kkelchte/docker_home/tensorflow/log/gridsearchtags
# 					WT="${WALLTIME[$((i%3))]}"
# 					ME="${MAXEPISODES[$((i%3))]}"
# 					./condor_task_offline.sh -t gridsearch_for_$i -q $WT -m mobilenet_025 -e true -n 20 -w "forest" -p "--n_fc True --grad_mul_weight $GM --batch_size $BS --learning_rate $LR --max_episodes $ME --dataset forest --weight_decay ${WD}e-05  --random_seed 512 --dropout_keep_prob $DO"
# 					./condor_task_offline.sh -t gridsearch_san_$i -q $WT -m mobilenet_025 -e true -n 20 -w "sandbox" -p "--n_fc True --grad_mul_weight $GM --batch_size $BS --learning_rate $LR --max_episodes $ME --dataset sandbox --weight_decay ${WD}e-05  --random_seed 512 --dropout_keep_prob $DO"
# 					./condor_task_offline.sh -t gridsearch_can_$i -q $WT -m mobilenet_025 -e true -n 20 -w "canyon" -p "--n_fc True --grad_mul_weight $GM --batch_size $BS --learning_rate $LR --max_episodes $ME --dataset canyon --weight_decay ${WD}e-05  --random_seed 512 --dropout_keep_prob $DO"
# 					i=$((i+1))
# 			done
# 		done
# 	done
# done

# WT="$((60*60*30))"
# ME="$((150*5))"
# GM=0.0	
# DO=0.75
# BS=32
# WD=4
# LR=0.1
# for i in $(seq 0 4) ; do
# 	# ./condor_task_offline.sh -t variance_for_$i -q $WT -m mobilenet_025 -e true -n 20 -w "forest" -p "--n_fc True --grad_mul_weight $GM --batch_size $BS --learning_rate $LR --max_episodes $ME --dataset forest --weight_decay ${WD}e-05  --random_seed 1234 --dropout_keep_prob $DO"
# 	./condor_task_sing.sh -t variance_for_$i -m variance_for_$i -n 20 -w "forest" -p "--load_config True --random_seed 1234"
# 	# ./condor_task_offline.sh -t variance_for_difsd_$i -q $WT -m mobilenet_025 -e true -n 20 -w "forest" -p "--n_fc True --grad_mul_weight $GM --batch_size $BS --learning_rate $LR --max_episodes $ME --dataset forest --weight_decay ${WD}e-05  --random_seed $((3000*$i+1539)) --dropout_keep_prob $DO"
# 	./condor_task_sing.sh -t variance_for_difsd_$i -m variance_for_difsd_$i -n 20 -w "forest" -p "--load_config True --random_seed $((3000*$i+1539))"
# 	sleep 0.3	
# done
#WT="$((60*60*3*5))"
#ME="$((150*5))"
#GM=0.0
#DO=0.75
#BS=32
#WD=4
#LR=0.1
# for i in $(seq 0 4) ; do
# 	# ./condor_task_offline.sh -t variance_san_$i -q $WT -m mobilenet_025 -e true -n 20 -w "sandbox" -p "--n_fc True --grad_mul_weight $GM --batch_size $BS --learning_rate $LR --max_episodes $ME --dataset sandbox --weight_decay ${WD}e-05  --random_seed 1234 --dropout_keep_prob $DO"
# 	./condor_task_sing.sh -t variance_san_$i -m variance_san_$i -n 20 -w "sandbox" -p "--load_config True --random_seed 1234"
# 	# ./condor_task_offline.sh -t variance_san_difsd_$i -q $WT -m mobilenet_025 -e true -n 20 -w "sandbox" -p "--n_fc True --grad_mul_weight $GM --batch_size $BS --learning_rate $LR --max_episodes $ME --dataset sandbox --weight_decay ${WD}e-05  --random_seed $((3000*$i+1539)) --dropout_keep_prob $DO"
# 	./condor_task_sing.sh -t variance_san_difsd_$i -m variance_san_difsd_$i -n 20 -w "sandbox" -p "--load_config True --random_seed $((3000*$i+1539))"
# 	sleep 0.3
# done	
#	./condor_task_offline.sh -t variance_can_$i -q $WT -m mobilenet_025 -e true -n 20 -w "canyon" -p "--n_fc True --grad_mul_weight $GM --batch_size $BS --learning_rate $LR --max_episodes $ME --dataset canyon --weight_decay ${WD}e-05  --random_seed $((3000*$i+1539)) --dropout_keep_prob $DO"
#	sleep 0.3
#done

#for i in $(seq 0 14) ; do
 	#WT="${WALLTIME["6"]}"
 	#ME="${MAXEPISODES["6"]}"
	# ./condor_task_offline.sh -q $WT -t variance_seed_$i -e true -n 20 -w "canyon" -p "--batch_size 32 --max_episodes $ME --learning_rate 0.1 --dataset canyon --random_seed $((3000*$i+1539)) --weight_decay 4e-05 --dropout_keep_prob 0.5" 
# 	./condor_task_offline.sh -q $((15*60*60)) -t variance_imgnet_$i  -e true -n 20 -m mobilenet_025 -w "canyon" -p "--batch_size 32 --max_episodes 750 --learning_rate 0.1 --dataset canyon --random_seed 512" 
# # 	./condor_task_offline.sh -q $WT -t variance_seed_$i  -e true -n 20 -w "canyon" -p "--batch_size 32 --max_episodes $ME --learning_rate 0.1 --dataset canyon --random_seed $((3000*$i+15643))"
 #	./condor_task_offline.sh -q $((WT*3)) -t variance_forcanssan_$i -m mobilenet_025 -e true -n 60 -w "canyon forest sandbox" -p "--continue_training False --batch_size 32 --max_episodes 300 --learning_rate 0.1 --dataset canyon_forest_sandbox --random_seed 512"
#  	./condor_task_offline.sh -q $WT -t variance_auxd_$i  -e true -n 20 -w "canyon" -p "--max_episodes $ME --batch_size 32 --learning_rate 0.1 --dataset canyon --auxiliary_depth True --random_seed 512" 
# # # 	# ./condor_task_offline.sh -q $((4*60*60)) -t variance_imgnet_$i -m mobilenet_025  -e true -n 20 -w "canyon" -p "--batch_size 32 --max_episodes 200 --learning_rate 0.1 --dataset canyon --random_seed 512" 
# 	./condor_task_offline.sh -q $WT -t variance_nfc_$i  -e true -n 20 -w "canyon" -p "--max_episodes $ME --batch_size 32 --dataset canyon --n_fc True --random_seed 512"
# 	./condor_task_offline.sh -q $WT -t variance_auxdn_$i  -e true -n 20 -w "canyon" -p "--max_episodes $ME --batch_size 32 --dataset canyon --n_fc True --auxiliary_depth True --random_seed 512"
	# ./condor_task_offline.sh -q $WT -t variance_dscr_$i -m mobilenet_025 -e true -n 10 -w "forest" -p "--dropout_keep_prob 0.25 --weight_decay 20e-05 --max_episodes $ME --learning_rate 0.05 --batch_size 64 --dataset forest --random_seed 512 --discrete True --num_outputs 3"
#done


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


# -------SINGULARITY-------
# for i in $(seq 0 9) ; do
# 	./condor_task_sing.sh -t redo_in_diff_canyons_${i} -m variance_ref_$i -p "--load_config True" -n 5 -w "canyon"
# done
# -------DOCKER-------
# ./condor_task_docker.sh -t doshico_${i} -s evaluate_model.sh -m doshico_$i -p "--load_config True" -n 30 -w "esat_v1 esat_v2 canyon"

# WT=$((24*60*60))
# ./condor_task_offline.sh -q $WT -t off_can_coll_cont -m off_can_coll -e true -n 20 -w "canyon" -p "--load_config True --continue_training True"
# ./condor_task_offline.sh -q $WT -t off_can_depth_cont -m off_can_rl_depth -e true -n 20 -w "canyon" -p "--load_config True --continue_training True"

watch condor_q
