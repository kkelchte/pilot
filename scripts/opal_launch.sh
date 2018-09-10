#!/bin/bash
#
# This script is used to prepare next tasks to be launched on opal
#
#

############################# TRAIN SEPARATE
world=all_factors

# loss=mse
# non_expert_weight=1
# tag=${world}/ensemble_${loss}_1

# loss=mse
# non_expert_weight=0.1
# tag=${world}/ensemble_${loss}_01

# loss=smce
# non_expert_weight=1
# tag=${world}/ensemble_${loss}_1

# loss=ce
# non_expert_weight=0.1
# tag=${world}/ensemble_${loss}_01

loss=mse
non_expert_weight=1
tag=${world}/ensemble_${loss}_1_squeeze


echo "$(date +%H:%M:%S) -------- $tag "

mkdir -p /esat/opal/kkelchte/docker_home/tensorflow/log/$tag

python /esat/opal/kkelchte/docker_home/tensorflow/ensemble_v0/pilot/main.py --dataset $world  \
                                                                            --network squeeze_v1 \
                                                                            --load_data_in_ram \
                                                                            --scratch  \
                                                                            --discrete \
                                                                            --normalize_over_actions  \
                                                                            --learning_rate 0.1  \
                                                                            --max_episodes 4000 \
                                                                            --visualize_deep_dream_of_output  \
                                                                            --visualize_saliency_of_output  \
                                                                            --loss ${loss}  \
                                                                            --random_seed 654  \
                                                                            --non_expert_weight ${non_expert_weight}  \
                                                                            --log_tag $tag >> /esat/opal/kkelchte/docker_home/tensorflow/log/${tag}_output 2>&1
echo "$(date +%H:%M:%S) ---------- done "


############################# TRAIN COMBINED
# world=all_factors

#world=combined_corridor

#name=mobile
#name=squeeze_v1
#name=alex_v3
#name=alex_v4

# tag=${world}/${name}


# echo "$(date +%H:%M:%S) -------- $tag "


# mkdir -p /esat/opal/kkelchte/docker_home/tensorflow/log/
# python /esat/opal/kkelchte/docker_home/tensorflow/ensemble_v0/pilot/main.py --log_tag $tag \
#                                                                       --dataset ${world} \
#                                                                       --max_episodes 1000 \
#                                                                       --network $name \
#                                                 								      --discrete \
#                                                                       --normalize_over_actions  \
#                                                                       --visualize_deep_dream_of_output \
#                                                                       --visualize_saliency_of_output \
#                                                                       --histogram_of_weights \
#                                                                       --histogram_of_activations >> /esat/opal/kkelchte/docker_home/tensorflow/log/${tag}_output 2>&1

# echo "$(date +%H:%M:%S) ---------- done "

# TODO
# python /esat/opal/kkelchte/docker_home/tensorflow/ensemble_v0/scripts/save_results_as_pdf.py --mother_dir ${name}


# TODO
# singularity exec --nv /esat/opal/kkelchte/singularity_images/ros_gazebo_tensorflow_writable.img /esat/opal/kkelchte/docker_home/tensorflow/ensemble_v0/scripts/evaluate_in_singularity.sh >> /esat/opal/kkelchte/docker_home/tensorflow/log/${world}/${name}_output 2>&1
