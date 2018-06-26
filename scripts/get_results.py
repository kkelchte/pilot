#!/usr/bin/python

import os,shutil,sys
import numpy as np
import subprocess, shlex


import argparse
#-------------------------------------------------------------------------------
# Parse results from group of log folders with min, max and variance over 5 runs
#-------------------------------------------------------------------------------

log_root = '/esat/opal/kkelchte/docker_home/tensorflow/log'

parser = argparse.ArgumentParser(description='Get results')
parser.add_argument('--mother_dir', default='', type=str, help='if all runs are grouped in one mother directory in log: e.g. depth_q_net')
parser.add_argument('--startswith', default='', type=str, help='if all runs start with a tag: e.g. base')
parser.add_argument('--endswith', default='', type=str, help='if all runs ends with a tag: e.g. eva')
args = parser.parse_args()


log_folders = sorted([ log_root+'/'+args.mother_dir+'/'+d for d in os.listdir(log_root+'/'+args.mother_dir) if d.startswith(args.startswith) and d.endswith(args.endswith)])

if len(log_folders)==0:
	print "Woops, could not find anythin ing "+log_root+"/"+args.mother_dir+" that startswith "+args.startswith+" and endswith "+args.endswith
	sys.exit(1)

for folder in log_folders:
	# print folder
	try:
		distances=[float(l.split(',')[2].split(':')[1]) for lf in os.listdir(folder+'/xterm_python') for l in open(folder+'/xterm_python/'+lf,'r').readlines() if 'Distance_furthest' in l]
		delays=[float(l.split(',')[6].split(':')[1]) for lf in os.listdir(folder+'/xterm_python') for l in open(folder+'/xterm_python/'+lf,'r').readlines() if 'Distance_furthest' in l]
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
			#print "average: "+str(np.mean(distances))+", min: "+str(np.min(distances))+", max "+str(np.max(distances))+", variance: "+str(np.var(distances))+", success_rate: "+str(success_rate)
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
			# print  "{8} arverage {0:0.2f} & min {1:0.2f} & max {2:0.2f} & var {3:0.2f} & success {4} / {5} & avg delay {6:.2E} & var delay {7:.2E} & host {9}".format(np.mean(distances),
			# 	np.min(distances),
			# 	np.max(distances),
			# 	np.var(distances),
			# 	success_rate,
			# 	len(distances),
			# 	np.mean(delays),
			# 	np.var(delays),
			# 	os.path.basename(folder),
			# 	hosts)
		# latex format
		# print  "{0:0.2f} & {1:0.2f} & {2}/{3}".format(np.mean(distances),np.var(distances),success_rate,len(distances))
