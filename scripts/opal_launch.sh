#!/bin/bash
#
# This script is used to prepare next tasks to be launched on opal
#
#

############################# TRAIN SEPARATE
world=all_factors

input=image
tag=${world}/committee_scratch_${input}


echo "$(date +%H:%M:%S) -------- $tag "

mkdir -p /esat/opal/kkelchte/docker_home/tensorflow/log/$tag

python /esat/opal/kkelchte/docker_home/tensorflow/ensemble_v1/pilot/main.py --dataset $world  \
                                                                            --discriminator_input ${input} \
                                                                            --network mobile \
                                                                            --load_data_in_ram \
                                                                            --histogram_of_weights \
                                                                            --discrete \
                                                                            --scratch \
                                                                            --learning_rate 0.1  \
                                                                            --max_episodes 1000 \
                                                                            --random_seed 654  \
                                                                            --non_expert_weight 0  \
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
# python /esat/opal/kkelchte/docker_home/tensorflow/ensemble_v1/pilot/main.py --log_tag $tag \
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
# python /esat/opal/kkelchte/docker_home/tensorflow/ensemble_v1/scripts/save_results_as_pdf.py --mother_dir ${name}


# TODO
# singularity exec --nv /esat/opal/kkelchte/singularity_images/ros_gazebo_tensorflow_writable.img /esat/opal/kkelchte/docker_home/tensorflow/ensemble_v1/scripts/evaluate_in_singularity.sh >> /esat/opal/kkelchte/docker_home/tensorflow/log/${world}/${name}_output 2>&1
