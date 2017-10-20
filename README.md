# pilot
This repo provides the code for training a DNN policy based on mobilenet on an offline dataset or online in a gazebo environment.
Tensorflow 1.1 code for training DNN policy from an offline dataset or online with ROS and Gazebo. It is used in the [DoShiCo challenge](https://kkelchte.github.io/doshico).

## Dependencies
* [Tensorflow (>1.1)](https://www.tensorflow.org/install/) or [docker image](https://hub.docker.com/r/kkelchte/ros_gazebo_tensorflow/) up and running.
* [data]("https://homes.esat.kuleuven.be/~kkelchte/pilot_data/data.zip"): this zip file contains the offline datasets:
  * Training data: collected in the simulated environments: [canyon, forest and sandbox](https://homes.esat.kuleuven.be/~kkelchte/data/pilot_data/canyon_forest_sandbox.zip)
  * Validation data: collected in the simulated environment: [ESAT](https://homes.esat.kuleuven.be/~kkelchte/data/pilot_data/esat.zip)
  * Test data: collected in the real world: [Almost-Collision Dataset](https://homes.esat.kuleuven.be/~kkelchte/data/pilot_data/almost_collision_set.zip)
* pretrained models: 
  * [imagenet-pretrained mobilenet 0.25](https://homes.esat.kuleuven.be/~kkelchte/checkpoints/mobilenet_025.zip)
  * [doshico-pretrained NAUX](https://homes.esat.kuleuven.be/~kkelchte/checkpoints/naux.zip)
  * [doshico-pretrained AUXD](https://homes.esat.kuleuven.be/~kkelchte/checkpoints/auxd.zip)


## Installation
You can use this code from within the [docker image](https://hub.docker.com/r/kkelchte/ros_gazebo_tensorflow/) I supply for the [Doshico challenge](http://kkelchte.github.io/doshico).
```bash
$ git clone https://www.github.com/kkelchte/pilot
# within a running docker container or tensorflow-virtual environment
$$  python main.py
```
It is best to download the log folder and save it on the correct relative path as well as the data folder.

Please use the [installation instructions](https://github.com/kkelchte/doshico/tree/master/assets/instructions) provided.


