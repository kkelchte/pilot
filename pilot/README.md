## Branch  q-learning


### Train 3-Q-Net

```
python main.py --checkpoint_path mobilenet_025 --continue_training False \
--depth_q_learning True --network three_q_net --dataset canyon --log_tag test_3_q_net \
--discrete True --subsample 4 --num_outputs 3 
```

