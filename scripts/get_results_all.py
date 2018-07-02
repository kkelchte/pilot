#!/usr/bin/python

import os,shutil,sys
import numpy as np
import subprocess, shlex


import argparse
#-------------------------------------------------------------------------------
# Parse results from group of log folders with min, max and variance over 5 runs
#-------------------------------------------------------------------------------
# extract online evaluation results
results={}
root_dir='/esat/opal/kkelchte/docker_home/tensorflow/log'
for n in [50, 100, 200, 500, 700, 900] :
	results[n]={}
	for net in ['depth_q_net_no_coll', 'coll_q_net']:
		results[n][net]={}
		results[n][net]['distances']=[]
		results[n][net]['success_rate']=[]
		results[n][net]['imitations']=[]
		for i in [0,1,2]:
			folder=root_dir+'/'+net+'/ds'+str(n)+'_'+str(i)+'_eva'
			distances = [float(e.split(':')[1]) for lf in os.listdir(folder+'/xterm_python') for l in open(folder+'/xterm_python/'+lf,'r').readlines() for e in l.split(',') if 'Distance_furthest' in e ]
			imitations = [float(e.split(':')[1]) for lf in os.listdir(folder+'/xterm_python') for l in open(folder+'/xterm_python/'+lf,'r').readlines() for e in l.split(',') if 'imitation_loss' in e ]
			assert len(distances)==20, "couldn't find 20 runs in {0}".format(root_dir+'/'+net+'/ds'+str(n)+'_'+str(i)+'_eva')
			results[n][net]['distances'].append(np.mean(distances))
			results[n][net]['success_rate'].append(len([d for d in distances if d > 15]))
			results[n][net]['imitations'].append(np.mean(imitations))
# extract offline test loss
for n in [50, 100, 200, 500, 700, 900] :
	for net in ['depth_q_net_no_coll', 'coll_q_net']:
		results[n][net]['testloss']=[]
		for i in [0,1,2]:
			try:
				folder=root_dir+'/'+net+'/ds'+str(n)+'_'+str(i)
				if not os.path.isfile(folder+'/tf_log'):
					logfile = folder+'/'+[r for r in os.listdir(folder) if r.startswith('2018')][0]+'/tf_log'
				else:
					logfile = folder+'/tf_log'
				results[n][net]['testloss'].append(float(open(logfile,'r').readlines()[-1].split(',')[0].split(':')[1][:-1]))
			except:
				print("problem parsing: {}".format(folder))
# print results in a nice way:
for n in [50, 100, 200, 500, 700, 900] :
	print("{0} & {1:0.2f} ({2:0.2E}) & {3:0.2f} ({4:0.2E}) & {5:0.1f} ({6:0.1f}) & {7:0.1f} ({8:0.1f}) & {9:0.2f} ({10:0.2E}) & {11:0.2f} ({12:0.2E}) & {13:0.2f} ({14:0.2E}) & {15:0.2f} ({16:0.2E})".format(n,
							np.mean(results[n]['coll_q_net']['distances']),
							np.std(results[n]['coll_q_net']['distances']),
							np.mean(results[n]['depth_q_net_no_coll']['distances']),
							np.std(results[n]['depth_q_net_no_coll']['distances']),
							np.mean(results[n]['coll_q_net']['success_rate']),
							np.std(results[n]['coll_q_net']['success_rate']),
							np.mean(results[n]['depth_q_net_no_coll']['success_rate']),
							np.std(results[n]['depth_q_net_no_coll']['success_rate']),
							np.mean(results[n]['coll_q_net']['imitations']),
							np.std(results[n]['coll_q_net']['imitations']),
							np.mean(results[n]['depth_q_net_no_coll']['imitations']),
							np.std(results[n]['depth_q_net_no_coll']['imitations']),
							np.mean(results[n]['coll_q_net']['testloss']),
							np.std(results[n]['coll_q_net']['testloss']),
							np.mean(results[n]['depth_q_net_no_coll']['testloss']),
							np.std(results[n]['depth_q_net_no_coll']['testloss'])))

			

# log_root = args.root_dir
# log_folders = sorted([ log_root+'/'+args.mother_dir+'/'+d for d in os.listdir(log_root+'/'+args.mother_dir) if d.startswith(args.startswith) and d.endswith(args.endswith)])

# if len(log_folders)==0:
# 	print "Woops, could not find anythin ing "+log_root+"/"+args.mother_dir+" that startswith "+args.startswith+" and endswith "+args.endswith
# 	sys.exit(1)

# for folder in log_folders:
# 	# print folder
# 	try:
# 		distances = [float(e.split(':')[1]) for lf in os.listdir(folder+'/xterm_python') for l in open(folder+'/xterm_python/'+lf,'r').readlines() for e in l.split(',') if 'Distance_furthest' in e ]
# 		delays = [float(e.split(':')[1]) for lf in os.listdir(folder+'/xterm_python') for l in open(folder+'/xterm_python/'+lf,'r').readlines() for e in l.split(',') if 'avg_delay' in e ]
# 		durations = [float(e.split(':')[1]) for lf in os.listdir(folder+'/xterm_python') for l in open(folder+'/xterm_python/'+lf,'r').readlines() for e in l.split(',') if 'driving_duration' in e ]
# 		imitations = [float(e.split(':')[1]) for lf in os.listdir(folder+'/xterm_python') for l in open(folder+'/xterm_python/'+lf,'r').readlines() for e in l.split(',') if 'imitation_loss' in e ]
# 	except:
# 		print 'Failed to parse: '+folder
# 	else:
# 		success_rate=len([d for d in distances if d > 15])
# 		# extract host
# 		hosts=[]
# 		try:
# 			ipadresses_lines=[l for l in open([folder+'/condor/'+f for f in os.listdir(folder+'/condor') if f.endswith('.log')][0],'r').readlines() if 'executing on host' in l]
# 			hosts = [subprocess.check_output(shlex.split("host "+l.split(' ')[8][1:].split(':')[0])).split(' ')[4].split('.')[0] for l in ipadresses_lines]
# 			# print 'hosts: '+str(hosts)
# 		except:
# 			pass
# 		if len(distances) != 0 and len(delays) != 0:
# 			if args.real:
# 				print "{0}: avg coll free dis: {1:0.4f} (var: {2:0.4f}), avg coll free dur: {3:0.4f} (var: {4:0.4f}), avg imitation loss: {5:0.4f} (var: {6:0.4f})".format(os.path.basename(folder),
# 					np.mean(distances),
# 					np.var(distances),
# 					np.mean(durations),
# 					np.var(durations),
# 					np.mean(imitations),
# 					np.var(imitations)
# 					)	
# 				if args.overleaf:
# 					print "{1:0.2f} ({2:0.1f}) & {3:0.2f} ({4:0.1f}) & {5:0.2f} ({6:0.1f})".format(os.path.basename(folder),
# 						np.mean(distances),
# 						np.var(distances),
# 						np.mean(durations),
# 						np.var(durations),
# 						np.mean(imitations),
# 						np.var(imitations)
# 						)
# 			else:	
# 				print "average: "+str(np.mean(distances))+", min: "+str(np.min(distances))+", max "+str(np.max(distances))+", variance: "+str(np.var(distances))+", success_rate: "+str(success_rate)
# 				print  "{8} arverage {0:0.2f} & var {3:0.2f} & success {4} / {5} & avg delay {6:.2E} & host {9}".format(np.mean(distances),
# 					np.min(distances),
# 					np.max(distances),
# 					np.var(distances),
# 					success_rate,
# 					len(distances),
# 					np.mean(delays),
# 					np.var(delays),
# 					os.path.basename(folder),
# 					hosts)
# 			# print  "{8} arverage {0:0.2f} & min {1:0.2f} & max {2:0.2f} & var {3:0.2f} & success {4} / {5} & avg delay {6:.2E} & var delay {7:.2E} & host {9}".format(np.mean(distances),
# 			# 	np.min(distances),
# 			# 	np.max(distances),
# 			# 	np.var(distances),
# 			# 	success_rate,
# 			# 	len(distances),
# 			# 	np.mean(delays),
# 			# 	np.var(delays),
# 			# 	os.path.basename(folder),
# 			# 	hosts)
# 		# latex format
# 		# print  "{0:0.2f} & {1:0.2f} & {2}/{3}".format(np.mean(distances),np.var(distances),success_rate,len(distances))

