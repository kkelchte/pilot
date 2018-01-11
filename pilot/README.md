## Branch for naive q-learning

### Train Naive Depth Q net

```
python main.py --checkpoint mobilenet_025 --dataset canyon \
--continue_training False --log_tag test_depth_q_net --offline True \
--network naive_q_net --naive_q_learning True 
```

### Test Naive Depth Q net

```
python main.py --checkpoint naive_q/model_1 --log_tag test_depth_q_net \
--network naive_q_net --naive_q_learning True 
```
