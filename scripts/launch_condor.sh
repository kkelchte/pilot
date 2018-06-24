#!/bin/bash
# Script used to invoke the condor online (singularity) and offline tasks.


#for i in 3 ; do
#	./condor_task_offline.sh -q $((4*3*60*60)) -t depth_q_net_no_coll/dummy_${i} -p "--max_episodes 1 --learning_rate 0.1 --dataset canyon_turtle_collision_free" -e true -w 'canyon' -n 50 -q 14400
#done

#./condor_task_sing.sh -d true -q $((4*7*60*60)) -t extra_canyon_turtle_scan -p params.yaml -e true -w canyon -n 1200

for i in 0 1 2 3 4 ; do
	./condor_task_offline.sh -q $((4*7*60*60)) -t depth_q_net/base_${i}  -e true -w 'canyon' -n 50 -p "--learning_rate 0.1 --dataset canyon_turtle_scan_pruned --random_seed $((i*13249))"
	./condor_task_offline.sh -q $((4*7*60*60)) -t coll_q_net/base_${i} -e true -w 'canyon' -n 50 -p "--loss mse --learning_rate 0.01 --dataset canyon_turtle_scan_pruned --random_seed $((i*13249))"
done

# ------------DOSHICO-------------

#TEST
#lr=0.1
#bs=64
#gmw=0.1
#i=1
#WT=$((60*60*2))
#TAG="$( echo test_doshico/lr${lr}_bs${bs}_gmw${gmw} | sed -En 's/\.//pg')" #change all dots in -
#./condor_task_offline.sh -q $WT -t $TAG -e true -n 20 -w "esat_v1 esat_v2" -p "--grad_mul_weight $gmw --batch_size $bs --max_episodes 30 --learning_rate $lr --random_seed $((3000*$i+1539)) --n_fc --auxiliary_depth"

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

watch condor_q
