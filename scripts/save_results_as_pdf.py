#!/usr/bin/python

import os,shutil,sys, time
import numpy as np
import subprocess, shlex
import json

import argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# util function
# add results for each run to the table
def save_append(dic, k, v):
	"""Append a value to dictionary (dic)
	if it gives a key error, create a new list.
	"""
	try:
		dic[k].append(v)
	except KeyError:
			dic[k]=[v]

"""
This script is used at the end of a DAG condor train_and_evaluate series.
It expects a motherdir with a number of models as well as a results.json file within that motherdir.
This scripts copies an example pdf into the report folder of the mother_dir.
Creates matplotlib images from the offline data and adds them to the copied template.
Add tables of the online results.
Possibly sends the pdf with email.
"""

parser = argparse.ArgumentParser(description='Parse offline results and online results as well as some images. Combine them in pdf. Send it to me.')
parser.add_argument('--home', default='/esat/opal/kkelchte/docker_home', type=str, help='Define the root directory: default is /esat/opal/kkelchte/docker_home/tensorflow/log')
parser.add_argument('--summary_dir', default='tensorflow/log/', type=str, help='Define the root directory: default is /esat/opal/kkelchte/docker_home/tensorflow/log')
parser.add_argument('--mother_dir', default='', type=str, help='if all runs are grouped in one mother directory in log: e.g. depth_q_net')

FLAGS, others = parser.parse_known_args()

mother_dir=FLAGS.home+'/'+FLAGS.summary_dir+FLAGS.mother_dir

# Step 1: copy template pdf-latex into projects log folder
current_dir=os.path.dirname(os.path.realpath(__file__))
if not os.path.isdir(mother_dir+'/report'): 
	os.mkdir(mother_dir+'/report')

for f in os.listdir(current_dir+'/template'):
	shutil.copyfile(current_dir+'/template/'+f, mother_dir+'/report/'+f)
# subprocess.call(shlex.split("cp -r {0}/template/ {1}/report/".format(current_dir,mother_dir)))

latex_file=open("{0}/report/report.tex".format(mother_dir), 'r')
report = latex_file.readlines()
latex_file.close()

# Step 2: extract offline training convergence and create images
# get_results is made to extract the online results so quickly extracting offline results here.
offline_models=[d for d in os.listdir(mother_dir) if not 'eva' in d and not d=='DAG' and not d=='results' and not d=='report' and os.path.isdir(mother_dir+'/'+d)]
offline_results={}
saliency_maps=""
control_dream_maps=""
for m in offline_models:
	offline_results[m]={}
	try:
		log_dirs=sorted([d for d in os.listdir(mother_dir+'/'+m) if '2018' in d])
		log_dir = '' if len(log_dirs) == 0 else '/'+log_dirs[-1]
		tf_log=open(mother_dir+'/'+m+log_dir+'/tf_log','r').readlines()
		saliency_maps = mother_dir+'/'+m+log_dir+'/saliency_maps.jpg'
		control_dream_maps = mother_dir+'/'+m+log_dir+'/control_dream_maps.jpg'
		# saliency_maps = sorted([d for d in os.listdir(mother_dir+'/'+m) if '2018' in d])[-1]+'/saliency_maps.jpg'
		# control_dream_maps = sorted([d for d in os.listdir(mother_dir+'/'+m) if '2018' in d])[-1]+'/control_dream_maps.jpg'
	except:
		print("[save_results_as_pdf]: could not find offline tf_log in {0}/{1}".format(FLAGS.mother_dir, m))
		continue
	else:
		for line in tf_log:
			# ['Loss_val_total : 4.69217729568', 'Loss_train_control : 4.55608940125', 'Loss_train_total : 4.61403799057', 'Loss_val_control : 4.63422870636']
			for v in line.split('[\'')[1].split('\']')[0].split('\', \''):
				try:
					key = v.split(' :')[0]
					val = float(v.split(':')[1])
					save_append(offline_results[m], key, val)					
				except:
					pass
# create matplotlib images of offline data
graph_keys=list(set([k for m in offline_results.keys() for k in offline_results[m].keys()]))
if len(graph_keys) > 4:
	fig, axes = plt.subplots(int(np.ceil(len(graph_keys)/4.)), 4, figsize=(23, 5*int(np.ceil(len(graph_keys)/4.))))
else:
	fig, axes = plt.subplots(1,len(graph_keys))

for i, gk in enumerate(graph_keys):
	axes[i/4][i%4].set_title(gk)
	for m in offline_results.keys():
		try:
			axes[i/4][i%4].scatter(range(len(offline_results[m][gk])), offline_results[m][gk])
		except KeyError:
			pass
plt.savefig(mother_dir+'/report/offline_results.jpg',bbox_inches='tight')

# Find line in report to add offline results
for l in report:
	if 'INSERTOFFLINERESULTS' in l:
		line_index=report.index(l)
report[line_index] = ""
report.insert(line_index, "\\begin{figure}[ht] \n")
line_index+=1
report.insert(line_index, "\\includegraphics[width=1.2\\textwidth]{"+mother_dir+'/report/offline_results.jpg'+"}\n")
line_index+=1
report.insert(line_index, "\\end{figure} \n")
line_index+=1

# In case there are saliency maps or deep dream maps in the offline folder, add them to the report
def add_figure(report, line_index, image_path):
	if os.path.isfile(image_path):
		report.insert(line_index, "\\begin{figure}[ht] \n")
		line_index+=1
		report.insert(line_index, "\\includegraphics[width=\\textwidth]{"+image_path+"}\n")
		line_index+=1
		report.insert(line_index, "\\end{figure} \n")
		line_index+=1
	return report, line_index

report, line_index = add_figure(report, line_index, saliency_maps)
report, line_index = add_figure(report, line_index, control_dream_maps)


# Step 3: extract online results from json and add tables if json file with results is there
if os.path.isfile("{0}/results.json".format(mother_dir)):
	with open('{0}/results.json'.format(mother_dir), 'r') as data_file:    
	    data = json.load(data_file)

	# Adjust the title
	for l in report:
		if 'TEMPLATE REPORT' in l:
			report[report.index(l)]=l.replace('TEMPLATE REPORT',FLAGS.mother_dir.replace('_',' '))

	# Find line in report to add online results
	for l in report:
		if 'INSERTONLINERESULTS' in l:
			line_index=report.index(l)
	report[line_index] = ""

	# Define each key for which a separate table is created
	# results[table_name][run_name(with hosts)][row information]
	run_images={}
	table_keys=['success', 'Distance_furthest']
	results={}
	for k in table_keys: # go over tables and add them to report		
		results[k]={}
		results[k]['total']={}
		# go over runs and add them to results dictionary
		for r in sorted(data.keys()): 
			if '_eva' in r:
				name=str(r).replace("_"," ")+" "+str([str(i) for i in list(set(data[r]['host']))])
				results[k][name]={}
				if k in data[r].keys():
					if "worldnames" in data[r].keys():
						for i, w in enumerate(data[r]["worldnames"]):
							save_append(results[k][name],w,data[r][k][i])
							save_append(results[k][name],'total',data[r][k][i])
							save_append(results[k]['total'],w,data[r][k][i])
					else:
						for v in data[r][k]: 
							save_append(results[k][name],'total',v)
							save_append(results[k]['total'],'total',v)
				if 'run_img' in data[r].keys(): run_images[r]=data[r]['run_img']			
		# start filling the table with column names
		worldnames=list(set([wn for name in results[k].keys() for wn in results[k][name].keys()]))
		worldnames.remove('total') # ensure total is in the end
		worldnames.append('total')
		start_table="\\begin{tabular}{|l|"+len(worldnames)*'c'+"|}\n"
		report.insert(line_index, start_table)
		line_index+=1
		report.insert(line_index, "\\hline\n")
		line_index+=1
		
		table_row="{0} ".format(k.replace('_',' '))
		for w in worldnames: table_row="{0} & {1} ".format(table_row, w)
		table_row="{0} \\\\ \n".format(table_row)
		report.insert(line_index, table_row)
		line_index+=1
		report.insert(line_index, "\\hline \n")
		line_index+=1
		
		def add_table_val(row, key):
			"""takes the row-results, 
			checks for a key and a list of results,
			returns the mean and std (func) of the list as string."""
			if key in row.keys(): 
				return "{0:0.2f} ({1:0.2f})".format(np.mean(row[key]), np.std(row[key]))
			else:
				return ''
		
		# go over runs (names) and add values in table if its there
		for name in sorted(results[k].keys()):
			table_row="{0} ".format(name)
			for w in worldnames: table_row="{0} & {1} ".format(table_row, add_table_val(results[k][name],w))
			table_row="{0} \\\\ \n".format(table_row)
			if name == 'total':
				report.insert(line_index, "\\hline \n")
				line_index+=1			
			report.insert(line_index, table_row)
			line_index+=1
		report.insert(line_index, "\\hline \n")
		line_index+=1
		# insert 
		report.insert(line_index, "\\end{tabular} \n")
		line_index+=1
		
	# Add for each model one trajectory
	report.insert(line_index, "\\section{First Run of Each Model} \n")
	line_index+=1

	for model in sorted(run_images.keys()):
		report.insert(line_index, "\\begin{figure}[ht] \n")
		line_index+=1
		report.insert(line_index, "\\includegraphics[width=0.5\\textwidth]{"+run_images[model][0]+"}\n")
		line_index+=1
		report.insert(line_index, "\\caption{"+model.replace('_',' ')+"} \n")
		line_index+=1
		report.insert(line_index, "\\end{figure} \n")
		line_index+=1




# Step 4: build pdf
latex_file=open("{0}/report/report.tex".format(mother_dir), 'w')
for l in report: latex_file.write(l)
latex_file.close()

subprocess.call(shlex.split("pdflatex -output-directory {0}/report {0}/report/report.tex".format(mother_dir)))	

# Step 5: send it with mailx
p_msg = subprocess.Popen(shlex.split("echo {0} : {1} is finished.".format(time.strftime("%Y-%m-%d_%I:%M:%S"), FLAGS.mother_dir)), stdout=subprocess.PIPE)
p_mail = subprocess.Popen(shlex.split("mailx -s {0} -a {1} klaas.kelchtermans@esat.kuleuven.be".format(FLAGS.mother_dir, mother_dir+'/report/report.pdf')),stdin=p_msg.stdout, stdout=subprocess.PIPE)
print(p_mail.communicate())


