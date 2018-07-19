#!/usr/bin/python

import os,shutil,sys
import numpy as np
import subprocess, shlex


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
log_folders = sorted([ log_root+FLAGS.mother_dir+'/'+d for d in os.listdir(log_root+FLAGS.mother_dir) if d.startswith(FLAGS.startswith) and d.endswith(FLAGS.endswith)])

if len(log_folders)==0:
	print "Woops, could not find anything "+log_root+FLAGS.mother_dir+" that startswith "+FLAGS.startswith+" and endswith "+FLAGS.endswith
	sys.exit(1)

# STEP 2: define information to parse: xterm_python, tf_log 
for folder in log_folders:
	print folder
	
	try:
		distances = [float(e.split(':')[1]) for lf in os.listdir(folder+'/xterm_python') for l in open(folder+'/xterm_python/'+lf,'r').readlines() for e in l.split(',') if 'Distance_furthest' in e ]
		delays = [float(e.split(':')[1]) for lf in os.listdir(folder+'/xterm_python') for l in open(folder+'/xterm_python/'+lf,'r').readlines() for e in l.split(',') if 'avg_delay' in e ]
		imitations = [float(e.split(':')[1]) for lf in os.listdir(folder+'/xterm_python') for l in open(folder+'/xterm_python/'+lf,'r').readlines() for e in l.split(',') if 'imitation_loss' in e ]
	except:
		print 'Failed to parse: '+folder
	else:
		success_rate=len([d for d in distances if d > 15])
		# extract host
		hosts=[]
		try:
			ipadresses_lines=[l for l in open([folder+'/condor/'+f for f in os.listdir(folder+'/condor') if f.endswith('.log')][0],'r').readlines() if 'executing on host' in l]
			hosts = [subprocess.check_output(shlex.split("host "+l.split(' ')[8][1:].split(':')[0])).split(' ')[4].split('.')[0] for l in ipadresses_lines]
			# print 'hosts: '+str(hosts)
		except:
			pass
		if len(distances) != 0 and len(delays) != 0:
			if FLAGS.real:
				print "{0}: avg coll free dis: {1:0.4f} (var: {2:0.4f}), avg coll free dur: {3:0.4f} (var: {4:0.4f}), avg imitation loss: {5:0.4f} (var: {6:0.4f}), avg depth loss: {7:0.4f} (var: {8:0.4f})".format(os.path.basename(folder),
					np.mean(distances),
					np.var(distances),
					np.mean(durations),
					np.var(durations),
					np.mean(imitations),
					np.var(imitations),
					np.mean(scans),
					np.var(scans)
					)	
				if FLAGS.overleaf:
					print "{1:0.2f} ({2:0.1f}) & {3:0.2f} ({4:0.1f}) & {5:0.2f} ({6:0.1f}) & {7:0.2E} ({8:0.1E})".format(os.path.basename(folder),
						np.mean(distances),
						np.var(distances),
						np.mean(durations),
						np.var(durations),
						np.mean(imitations),
						np.var(imitations),
						np.mean(scans),
						np.var(scans)
						)	
			else:	
				print "average: "+str(np.mean(distances))+", min: "+str(np.min(distances))+", max "+str(np.max(distances))+", variance: "+str(np.var(distances))+", success_rate: "+str(success_rate)
				print  "{8} arverage {0:0.2f} & var {3:0.2f} & success {4} / {5} & avg delay {6:.2E} & host {9}".format(np.mean(distances),
					np.min(distances),
					np.max(distances),
					np.var(distances),
					success_rate,
					len(distances),
					np.mean(delays),
					np.var(delays),
					os.path.basename(folder),
					hosts)
	
