#!/usr/bin/python

import os,shutil,sys
import numpy as np
import subprocess, shlex
import json


import argparse
"""
-------------------------------------------------------------------------------
 Parse results from group of log folders with min, max and variance over 5 runs
 
 Exit codes:
 1: no log folders found
-------------------------------------------------------------------------------
"""
# STEP 1: parse arguments and get log folders
parser = argparse.ArgumentParser(description='Get results')
parser.add_argument('--home', default='/esat/opal/kkelchte/docker_home', type=str, help='Define the root directory: default is /esat/opal/kkelchte/docker_home/tensorflow/log')
parser.add_argument('--summary_dir', default='tensorflow/log/', type=str, help='Define the root directory: default is /esat/opal/kkelchte/docker_home/tensorflow/log')
parser.add_argument('--mother_dir', default='', type=str, help='if all runs are grouped in one mother directory in log: e.g. depth_q_net')
parser.add_argument('--startswith', default='', type=str, help='if all runs start with a tag: e.g. base')
parser.add_argument('--endswith', default='', type=str, help='if all runs ends with a tag: e.g. eva')
parser.add_argument('--real', action='store_true', help='in case of real world experiments different results output are required.')
parser.add_argument('--overleaf', action='store_true', help='plot with && like in overleaf')

FLAGS, others = parser.parse_known_args()

print("\nSettings:")
for f in FLAGS.__dict__: print("{0}: {1}".format( f, FLAGS.__dict__[f]))
print("Others: {0}".format(others))

log_root = FLAGS.home+'/'+FLAGS.summary_dir
log_folders = sorted([ log_root+FLAGS.mother_dir+'/'+d if len(FLAGS.mother_dir) != 0 else log_root+d for d in os.listdir(log_root+FLAGS.mother_dir) if d.startswith(FLAGS.startswith) and d.endswith(FLAGS.endswith) and os.path.isdir(log_root+FLAGS.mother_dir+'/'+d)])

if len(log_folders)==0:
	print "Woops, could not find anything "+log_root+FLAGS.mother_dir+" that startswith "+FLAGS.startswith+" and endswith "+FLAGS.endswith
	sys.exit(1)

# STEP 2: define information to parse: tf_log, log_named and runs
tf_params=['Distance_current', 'Distance_furthest', 'control_rate', 'image_rate', 'control_delay_std', 'image_delay_std', 'driving_duration', 'imitation_loss']

# STEP 3: parse each run into results dictionary
# results[worldname]{tf_params[0]:[], ... , run_index:[], run_img: ['/path/to/img',..],successes: [True,False,None]}
# results['total']{tf_params[0]:[], ... , run_index:[], run_img: ['/path/to/img',..],successes: [True,False,None]}
results = {}
for folder in log_folders:
	print folder
	# get tensorflow log folders within evaluation log folder
	tf_folders = [folder+'/'+f for f in os.listdir(folder) if f.startswith('201') and os.path.isdir(folder+'/'+f)]
	if len(tf_folders) == 0:
		print("Empty logfolder: {}".format(folder))
		continue
	for tf_folder in tf_folders:
		try:
			tf_log = open(tf_folder+'/tf_log', 'r').readlines()[1:]
			log_named = open(tf_folder+'/log_named', 'r').readlines()
		except IOError:
			print("Failed to read tf_folder: {}".format(tf_folder))
			continue
		for l_i, l in enumerate(tf_log):
			# get world name:
			worldname=l.split(',')[1].split('_')[-1].split(':')[0]
			# create new dict if worldname not yet in results
			if worldname not in results.keys(): results[worldname]={}
			
			def add(res, wn, k, v):
				"""Append a value to the worldname (wn) dictionary in resuts (res)
				if it gives a key error, create a new list.
				"""
				try:
					res[wn][k].append(v)
				except KeyError:
					try:
						res[wn][k]=[v]
					except KeyError:
						res[wn]={k:[v]}		
			# add each parameter within the line to the results
			for p in tf_params:
				value = [float(v.split(':')[1]) for v in l.split(',') if p in v ]
				if len(value) == 1: 
					add(results, worldname, p, value)
					add(results, os.path.basename(folder), p, value)			
			# get run index from file and subtract 1
			run=int(l.split(',')[0].split(' ')[2])-1
			add(results, worldname, 'run_index', run)
			add(results, os.path.basename(folder), 'run_index', run)
			# parse success from log_named
			add(results, os.path.basename(folder), 'success', log_named[l_i].split(' ')[0]=='success')
			if log_named[l_i].split(' ')[1]==worldname:
				add(results, worldname, 'success', log_named[l_i].split(' ')[0]=='success')
			# with index and wordname point to run image if it exists
			run_image='{0}/runs/gt_{1:05d}_{2}.png'.format(tf_folder, run, worldname) if os.path.isfile('{0}/runs/gt_{1:05d}_{2}.png'.format(tf_folder, run, worldname)) else ''
			add(results, worldname, 'run_img', run_image)
			add(results, os.path.basename(folder), 'run_img', run_image)
			# parse current condor host from events.file.name
			host=[f.split('.')[4] for f in os.listdir(tf_folder) if f.startswith('events')][0]
			add(results, os.path.basename(folder), 'host', host)
			add(results, os.path.basename(folder), 'worldnames', worldname)

			
# STEP 5: write results in json file
# write results file
if os.path.isfile(log_root+FLAGS.mother_dir+'/results.json'):
	os.rename(log_root+FLAGS.mother_dir+'/results.json', log_root+FLAGS.mother_dir+'/_old_results.json')

with open(log_root+FLAGS.mother_dir+'/results.json','w') as out:
  json.dump(results,out,indent=2, sort_keys=True)
