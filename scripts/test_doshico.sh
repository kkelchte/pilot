#!/bin/bash
######################################################
# Pick which test. If no test is selected, go through them all.
NUM=$1
if [ $NUM -gt 5 ] ; then
    echo "$(tput setaf 1) There are tests up to 5 $(tput sgr 0)."
fi
echo
echo
if [[ -z $NUM || $NUM = 0 ]] ; then
    echo "$(tput setaf 4) Training model from doshico pretrained auxd model for 10 episodes on the big overview dataset."
    echo "Duration: 15min $(tput sgr 0)"
    ./launch_script.sh -t test_offline_doshico -m auxd -o true -n 10 -p "--dataset overview --plot_depth True"
fi
if [[ -z $NUM || $NUM = 1 ]] ; then
    echo "$(tput setaf 4) Evaluate model auxd online for 9 runs in canyon, forest and sandbox."
    echo "Duration: 20min $(tput sgr 0)"
    ./launch_script.sh -t test_online_doshico_eva_auxd -m auxd -o false -w "canyon forest sandbox" -n 9 -s evaluate_model.sh -p "--load_config True"
fi
if [[ -z $NUM || $NUM = 2 ]] ; then
    echo "$(tput setaf 4) Train model naux online for 9 runs in canyon, forest and sandbox."
    echo "Duration: 20min $(tput sgr 0)"
    ./launch_script.sh -t test_online_doshico_train_naux -m naux -o false -w "canyon forest sandbox" -n 9 -s train_model.sh -p "--load_config True"
fi
if [[ -z $NUM || $NUM = 3 ]] ; then
    echo "$(tput setaf 4) Train with auxiliary depth and evaluate model pretrained imagenet online for 3 episodes in canyon, forest and sandbox, evaluate in esat_v1 and esat_v2."
    echo "Duration: 15min $(tput sgr 0)"
    ./launch_script.sh -t test_online_doshico_tra_eva_mob -m mobilenet_025 -o false -n 0 -s train_and_evaluate_model.sh -p "--auxiliary_depth True"
fi
if [[ -z $NUM || $NUM = 4 ]] ; then
    echo "$(tput setaf 4) Vizualize results with tensorflow and port forwarding.$(tput sgr 0)"
    HOME_HOST=/home/$USER/docker_home
    HOME_CONT=/home/$USER
    cmd="./usr/local/bin/tensorboard --logdir $HOME/tensorflow/log --port 6005"
    sudo nvidia-docker run -it -d --rm -p 6005:6005 -v $HOME_HOST:$HOME_CONT -v /tmp/.X11-unix:/tmp/.X11-unix --name doshico_container doshico_image $HOME_CONT/.entrypoint $cmd
    sleep 5
    firefox http://0.0.0.0:6005
    echo "Go to your firefox and check out the results in tensorboard."
    echo "Alert! Your container is running in the background, you can stop it with:"
    echo "$ sudo docker stop doshico_container"
fi
if [[ -z $NUM || $NUM = 5 ]] ; then
    echo "$(tput setaf 4) Training model offline and evaluating online while visualizing results in tensorboard"
    echo "Duration: 60min $(tput sgr 0)"
    echo "$(tput setaf 4) >>>> Starting tensorboard $(tput sgr 0)"
    HOME_HOST=/home/$USER/docker_home
    HOME_CONT=/home/$USER
    cmd="./usr/local/bin/tensorboard --logdir $HOME/tensorflow/log --port 6005"
    sudo nvidia-docker run -it -d --rm -p 6005:6005 -v $HOME_HOST:$HOME_CONT -v /tmp/.X11-unix:/tmp/.X11-unix --name doshico_container_tensorboard doshico_image $HOME_CONT/.entrypoint $cmd
    sleep 5
    firefox http://0.0.0.0:6005
    echo "$(tput setaf 4) >>>> Starting offline training of log/train_offline_new_model  $(tput sgr 0)"
    ./launch_script.sh -t train_offline_new_model -m mobilenet_025 -o true -n 1 -p "--dataset overview --plot_depth True"
    echo "$(tput setaf 4) >>>> Evaluating online $(tput sgr 0)"
    ./launch_script.sh -t test_online_freshly_trained_model -m train_offline_new_model -n 10 -s evaluate_model.sh -p "--load_config True"
fi


echo 'done'
exit 