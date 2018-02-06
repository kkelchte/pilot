## Examples

### Branch for DDPG - Actor-Critic with collision probability estimator:
Done: made model.py useless by integrating all in actor.py
TODO:

	* implement critic.py
	* adjust rosinterface.py in finished() method:
		* labelling last x frames before bump with 1 
		* training critic and actor separately
		* updating every x runs the target critic and actor to the real

### Train model from doshico paper

* Initializing weights with imagenet feature (checkpoint mobilenet_025)
* Initialize control weights randomly (continue_training False)
* Extract features over 3 consecutive frames and concatenate them for control prediction (n_fc True n_frames 3)
* Dataset is the offline doshico set: train in canyon, forest and sandbox; validate on esat; test on almost-collision dataset
* Dataformat is NHWC due to mobilenet pretrained weights that need to fit.

```
python main.py --checkpoint mobilenet_025 --data_format NHWC --dataset doshico --continue_training False --log_tag test_doshico --offline True --random_seed 100
```

### Train Depth Q Net

```
python main.py --checkpoint mobilenet_025 --dataset doshico \
--continue_training False --log_tag depth_q_net --offline True \
--network depth_q_net --depth_q_learning True 
```

