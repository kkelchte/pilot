#!/bin/bash
""" 
Script for launching condor jobs invoking both condor_offline.py and condor_online.py scripts.
Author: Klaas Kelchtermans
Dependencies: condor_offline.py condor_online.py
"""

# ------------TRAIN OFFLINE----------

# ------------EVALUATE ONLINE-------------
python condor_online.py -t test_online -m depth_q_net_no_coll/ds900_0 --not_nice --wall_time $((60*60)) -e --reuse_default_world -n 20

while [ true ] ; do clear; condor_q; sleep 2; done
