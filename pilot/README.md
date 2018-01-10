## Branch for naive q-learning

### Train Depth Q Net

```
python main.py --checkpoint mobilenet_025 --dataset doshico \
--continue_training False --log_tag test_depth_q_net --offline True \
--network depth_q_net --depth_q_learning True 
```

