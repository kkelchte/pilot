#!/bin/bash
#
# This script is used to prepare next tasks to be launched on opal
#
#
clear
############################# TRAIN SEPARATE
# world=radiator
# world=corridor
# world=floor
# world=poster
# world=ceiling
# world=blocked_hole
# world=doorway
# world=arc

# name='mobile'
# tag=${name}_factored_scratch/${world}

# echo "$(date +%H:%M:%S) ---------- $tag "


# mkdir -p /esat/opal/kkelchte/docker_home/tensorflow/log/$tag
# python /esat/opal/kkelchte/docker_home/tensorflow/pilot/pilot/main.py --log_tag $tag \
#                                                                       --dataset ${world} \
#                                                                       --max_episodes 100 \
#                                                                       --network $name \
#                                                                       --discrete \
#                                                                       --scratch \
#                                                                       --load_data_in_ram \
#                                                                       --normalize_over_actions  \
#                                                                       --visualize_deep_dream_of_output \
#                                                                       --visualize_saliency_of_output \
#                                                                       --histogram_of_weights \
#                                                                       --histogram_of_activations >> /esat/opal/kkelchte/docker_home/tensorflow/log/${tag}_output 2>&1

# echo "$(date +%H:%M:%S) ---------- done "

############################# TRAIN COMBINED
# world=all_factors

world=combined_corridor

name=mobile
#name=squeeze_v1
#name=alex_v3
#name=alex_v4

tag=${world}/${name}_scratch


echo "$(date +%H:%M:%S) ---------- $tag "


mkdir -p /esat/opal/kkelchte/docker_home/tensorflow/log/$tag
python /esat/opal/kkelchte/docker_home/tensorflow/pilot/pilot/main.py --log_tag $tag \
                                                                      --dataset ${world} \
                                                                      --max_episodes 1000 \
                                                                      --network $name \
                                                                      --discrete \
                                                                      --scratch \
                                                                      --normalize_over_actions  \
                                                                      --visualize_deep_dream_of_output \
                                                                      --visualize_saliency_of_output \
                                                                      --histogram_of_weights \
                                                                      --histogram_of_activations >> /esat/opal/kkelchte/docker_home/tensorflow/log/${tag}_output 2>&1

echo "$(date +%H:%M:%S) ---------- done "

# TODO
# python /esat/opal/kkelchte/docker_home/tensorflow/pilot/scripts/save_results_as_pdf.py --mother_dir ${name}


# TODO
# singularity exec --nv /esat/opal/kkelchte/singularity_images/ros_gazebo_tensorflow_writable.img /esat/opal/kkelchte/docker_home/tensorflow/pilot/scripts/evaluate_in_singularity.sh >> /esat/opal/kkelchte/docker_home/tensorflow/log/${world}/${name}_output 2>&1
