#!/bin/bash
# Script used to invoke the condor online (singularity) and offline tasks.




#for i in 3 ; do
#	./condor_task_offline.sh -q $((4*3*60*60)) -t depth_q_net_no_coll/dummy_${i} -p "--max_episodes 1 --learning_rate 0.1 --dataset canyon_turtle_collision_free" -e true -w 'canyon' -n 50 -q 14400
#done

# for  i in 0 1 2 3 4 5 6 7 8 9 ; do
# ./condor_task_sing.sh -d true -q $((4*7*60*60)) -t rec_$i -p params.yaml -e true -w canyon -n 1200
# 	sleep 0.5
# done

# for i in 0 1 2 3 4 ; do
# 	./condor_task_offline.sh -q $((4*7*60*60)) -t depth_q_net_no_coll/ds490_${i}  -e true -w 'canyon' -n 50 -p "--learning_rate 0.1 --dataset canyon_turtle_collision_free --random_seed $((1534+i*13249))"
# 	./condor_task_offline.sh -q $((4*7*60*60)) -t depth_q_net/ds490_${i}  -e true -w 'canyon' -n 50 -p "--learning_rate 0.1 --dataset canyon_turtle_scan_pruned --random_seed $((1534+i*13249))"
# 	#./condor_task_offline.sh -q $((4*7*60*60)) -t coll_q_net/ds490_${i} -e true -w 'canyon' -n 50 -p "--normalized(????) --network coll_q_net --loss mse --learning_rate 0.01 --dataset canyon_turtle_scan_pruned --random_seed $((14123+i*13249))"
# 	sleep 0.5
# done


# ------------TRAIN OFFLINE----------
# for i in 0 1 2 ; do
# 	for n in 500 700 ; do
# 		./condor_task_offline.sh -q $((4*7*60*60)) -t depth_q_net_no_coll/ds${n}_${i} -e true -w 'canyon' -n 20 -p "--loss absolute --learning_rate 0.1 --dataset canyon_ds_coll_free${n} --random_seed $((1534+i*13249))"	
# 		sleep 2
# 		./condor_task_offline.sh -q $((4*7*60*60)) -t coll_q_net/ds${n}_${i} -e true -w 'canyon' -n 20 -p "--network coll_q_net --loss ce --learning_rate 0.01 --dataset canyon_ds${n} --random_seed $((14123+i*13249))"
# 		sleep 2
# 		./condor_task_offline.sh -q $((4*7*60*60)) -t coll_q_net/ds${n}_norm_${i} -e true -w 'canyon' -n 20 -p "--normalize_data --network coll_q_net --loss ce --learning_rate 0.01 --dataset canyon_ds${n} --random_seed $((14123+i*13249))"
# 		sleep 2
# 	done
# done

# for i in 0 1 2 ; do
# 	for n in 50 100 200 ; do
# 		./condor_task_offline.sh -q $((4*7*60*60)) -t coll_q_net/ds${n}_${i} -e true -w 'canyon' -n 20 -p "--network coll_q_net --loss ce --learning_rate 0.01 --dataset canyon_ds${n} --random_seed $((14123+i*13249))"
# 		sleep 2
# 		./condor_task_offline.sh -q $((4*7*60*60)) -t coll_q_net/ds${n}_norm_${i} -e true -w 'canyon' -n 20 -p "--normalize_data --network coll_q_net --loss ce --learning_rate 0.01 --dataset canyon_ds${n} --random_seed $((14123+i*13249))"
# 		sleep 2
# 	done
# done
# ------------EVALUATE ONLINE----------
# for i in 0 ; do
# 	for n in 20 30 40 50 60 70 80 90 ; do
# 	# for n in 20 30 40 ; do
# 		# ./condor_task_sing.sh -t coll_q_net/ds1400_${i}_${n}k_eva -m coll_q_net/ds1400_${i}_${n}k -n 50 -w 'canyon' -p "eva_params.yaml" -e -q 12000
# 		./condor_task_sing.sh -t depth_q_net_no_coll/ds1500_${i}_${n}k_eva -m depth_q_net_no_coll/ds1500_${i}_${n}k -n 5 -w 'canyon' -p "eva_params.yaml" -e -q 12000
# 	done
# 	# ./condor_task_sing.sh -t coll_q_net/ds1500_4096_${i}_eva -m coll_q_net/ds1500_4096_${i} -n 10 -w 'canyon' -p "eva_params.yaml" -e -q 12000
# done

# ------------REAL DATA-----------

# for lr in 0.9 0.1 0.01 0.001 0.0001 ; do
# 	for max_depth in 2 1.5 1 0.5 ; do
# 		# extract strings without points for naming the files
# 		LR="$(echo $lr | sed -e 's/\.//g')"
# 		MD="$(echo $max_depth | sed -e 's/\.//g')"
# 		./condor_task_offline.sh -q $((7*60*60)) -t depth_q_net_real/scratch_${LR}_${MD} -p "--learning_rate $lr --max_depth $max_depth --max_loss 0.5 --clip_loss_to_max --dataset real_maze --random_seed $((13249))"
# 		sleep 1
# 		./condor_task_offline.sh -q $((7*60*60)) -t depth_q_net_real/transfer_${LR}_${MD} -p "--learning_rate $lr --max_depth $max_depth --max_loss 0.5 --clip_loss_to_max --dataset real_maze --random_seed $((13249)) --continue_training --checkpoint_path depth_q_net_no_coll/ds900_0"
# 		sleep 1
# 	done
# done	
# ------------Transfer learning----------
# for i in 0 1 2 ; do
# 	./condor_task_offline.sh -q $((7*60*60)) -t depth_q_net_no_coll_real/scratch_${i}_lr09_e2e -p "--max_loss 0.5 --clip_loss_to_max --learning_rate 0.9 --grad_mul_weight 1 --dataset maze_real_turtle_collision_free --random_seed $((13249+65456*i))"
# 	./condor_task_offline.sh -q $((7*60*60)) -t depth_q_net_real/scratch_${i}_lr09_e2e -p "--max_loss 0.5 --clip_loss_to_max --learning_rate 0.9 --grad_mul_weight 1 --dataset maze_real_turtle --random_seed $((13249+65456*i))"
# 	./condor_task_offline.sh -q $((7*60*60)) -t depth_q_net_no_coll_real/transfer_${i} -p "--max_loss 0.5 --clip_loss_to_max --learning_rate 0.9 --continue_training --checkpoint_path depth_q_net_no_coll/ref_5 --dataset maze_real_turtle_collision_free --random_seed $((13249+65456*i))"
# 	./condor_task_offline.sh -q $((7*60*60)) -t depth_q_net_real/transfer_${i}  -p "--max_loss 0.5 --clip_loss_to_max --learning_rate 0.9 --continue_training --checkpoint_path depth_q_net/base_4 --dataset maze_real_turtle --random_seed $((13249+65456*i))"
# 	# ./condor_task_offline.sh -q $((7*60*60)) -t coll_q_net_real/ref  -p "--network coll_q_net --learning_rate 0.01 --loss mse --dataset maze_real_turtle --random_seed $((13249))"
# done



# ------------ONLINE-------------

# for i in 0 1 2 ; do
# 	./condor_task_sing.sh -q $((24*60*60)) -t coll_q_net/online_lr001_${i} -n 10000 -w 'canyon' -p "train_coll_q_net_params.yaml"
# 	./condor_task_sing.sh -q $((24*60*60)) -t coll_q_net/online_4096_${i} -n 10000 -w 'canyon' -p "train_coll_q_net_params_4096.yaml"
# 	./condor_task_sing.sh -q $((24*60*60)) -t coll_q_net/online_e2e_${i} -n 10000 -w 'canyon' -p "train_coll_q_net_params_e2e.yaml"
# done

# ------------DOSHICO-------------

#TEST
#lr=0.1
#bs=64
#gmw=0.1
#i=1
#WT=$((60*60*2))
#TAG="$( echo test_doshico/lr${lr}_bs${bs}_gmw${gmw} | sed -En 's/\.//pg')" #change all dots in -
#./condor_task_offline.sh -q $WT -t $TAG -e true -n 10 -w "esat_v1 esat_v2" -p "--grad_mul_weight $gmw --batch_size $bs --max_episodes 30 --learning_rate $lr --random_seed $((3000*$i+1539)) --n_fc --auxiliary_depth"

# ------------GRIDSEARCH----------
# i=1
# WT=$((60*60*3))
# for lr in 0.5 0.05 0.005 0.0005 ; do
# 	for bs in 16 32 64 ; do
# 		for gmw in 0.0 0.001 0.01 0.1 ; do
#  			./condor_task_offline.sh -q $WT -t doshico_auxd/lr${lr}_bs${bs}_gmw${gmw} -e true -n 20 -w "esat_v1 esat_v2" -p "--grad_mul_weight $gmw --batch_size $bs --max_episodes 30 --learning_rate $lr --random_seed $((3000*$i+1539)) --n_fc --auxiliary_depth"
# 		done
# 	done
# done
# for i in $(seq 0 55) ; do
# # for i in 0 ; do
# WT=$((3*60*60*4))
# 	# ME=$((3*5*100))
# 	./condor_task_offline.sh -q $WT -t doshico_auxd_gm0001/doshico_$i -m mobilenet_025 -e true -n 20 -w "esat_v1 esat_v2" -p "--grad_mul_weight 0.001 --batch_size 64 --max_episodes 20 --learning_rate 0.1 --dataset overview --random_seed $((3000*$i+1539)) --n_fc --auxiliary_depth"
# ./condor_task_offline.sh -q $WT -t doshico_naux_gm0001/doshico_$i -m mobilenet_025 -e true -n 20 -w "esat_v1 esat_v2" -p "--grad_mul_weight 0.001 --batch_size 64 --max_episodes 20 --learning_rate 0.1 --dataset overview --random_seed $((3000*$i+1539)) --n_fc --auxiliary_depth"
#        ./condor_task_offline.sh -q $WT -t doshico_fc_gm0001/doshico_$i -m mobilenet_025 -e true -n 20 -w "esat_v1 esat_v2" -p "--grad_mul_weight 0.001 --batch_size 64 --max_episodes 20 --learning_rate 0.1 --dataset overview --random_seed $((3000*$i+1539)) --n_fc --auxiliary_depth"
#       ./condor_task_offline.sh -q $WT -t doshico_auxd_gm01/doshico_$i -m mobilenet_025 -e true -n 20 -w "esat_v1 esat_v2" -p "--grad_mul_weight 0.1 --batch_size 64 --max_episodes 20 --learning_rate 0.1 --dataset overview --random_seed $((3000*$i+1539)) --n_fc --auxiliary_depth"
#	./condor_task_offline.sh -q $WT -t doshico_auxd_gm05/doshico_$i -m mobilenet_025 -e true -n 20 -w "esat_v1 esat_v2" -p "--grad_mul_weight 0.5 --batch_size 64 --max_episodes 20 --learning_rate 0.1 --dataset overview --random_seed $((3000*$i+1539)) --n_fc --auxiliary_depth"
# 	./condor_task_offline.sh -q $WT -t doshico_naux/doshico_$i -m mobilenet_025 -e true -n 20 -w "esat_v1 esat_v2" -p "--batch_size 64 --max_episodes 34 --learning_rate 0.1 --dataset overview --random_seed $((3000*$i+1539)) --n_fc" 
# 	./condor_task_offline.sh -q $WT -t doshico_fc/doshico_$i -m mobilenet_025 -e true -n 20 -w "esat_v1 esat_v2" -p "--batch_size 64 --max_episodes 50 --learning_rate 0.1 --dataset overview --random_seed $((3000*$i+1539))" 
# done

while [ true ] ; do clear; condor_q; sleep 2; done
