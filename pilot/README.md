## Examples


### Train model from doshico paper

* Initializing weights with imagenet feature (checkpoint mobilenet_025)
* Initialize control weights randomly (continue_training False)
* Extract features over 3 consecutive frames and concatenate them for control prediction (n_fc True n_frames 3)
* Dataset is the offline doshico set: train in canyon, forest and sandbox; validate on esat; test on almost-collision dataset
* Dataformat is NHWC due to mobilenet pretrained weights that need to fit.

python main.py --checkpoint mobilenet_025 --data_format NHWC --dataset doshico --continue_training False --log_tag test_doshico --offline True --random_seed 100
