## Examples


### Train Depth Q Net

```
python main.py --checkpoint mobilenet_025 --dataset canyon \
--continue_training False --log_tag test_depth_q_net --offline True \
--network depth_q_net --depth_q_learning True 
```

### Train collision prediction network online

```
python main.py --checkpoint mobilenet_025 --dataset canyon \
--continue_training False --log_tag test_coll_pred --offline False \
--network coll_q_net
```
