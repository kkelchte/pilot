#!/bin/bash
#
# This script is used to prepare next tasks to be launched on opal
#
#

# world=radiator
# world=corridor
# world=floor
# world=poster
# world=ceiling
# world=blocked_hole
# world=doorway

world=arc

name=mobile_factored

echo "$(date +%H:%M:%S) ---------- $name / $world "

mkdir -p /esat/opal/kkelchte/docker_home/tensorflow/log/${name}/${world}
python /esat/opal/kkelchte/docker_home/tensorflow/pilot/pilot/main.py --log_tag ${name}/${world} \
                                                                      --dataset ${world} \
                                                                      --load_data_in_ram \
                                                                      --max_episodes 100 \
                                                                      --discrete \
                                                                      --normalize_over_actions  \
                                                                      --visualize_deep_dream_of_output \
                                                                      --visualize_saliency_of_output \
                                                                      --histogram_of_weights \
                                                                      --histogram_of_activations >> /esat/opal/kkelchte/docker_home/tensorflow/log/${name}/${world}_output 2>&1

echo "$(date +%H:%M:%S) ---------- done "

# TODO
# python /esat/opal/kkelchte/docker_home/tensorflow/pilot/scripts/save_results_as_pdf.py --mother_dir ${name}