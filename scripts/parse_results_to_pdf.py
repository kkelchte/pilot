#!/usr/bin/python

import os,shutil,sys, time
import numpy as np
import subprocess, shlex
import json

import collections

import argparse
import xml.etree.cElementTree as ET


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import tablib

"""
Parse results from group of log folders with min, max and variance over 5 runs
It expects a motherdir with a number of models as well as a results.json file within that motherdir.
This scripts copies an example pdf into the report folder of the mother_dir.
Creates matplotlib images from the offline data and adds them to the copied template.
Add tables of the online results.
Possibly sends the pdf with email.

 
 Exit codes:
 1: no log folders found
"""

#--------------------------------------------------------------
#
# Utility functions
#
#--------------------------------------------------------------

def save_append(dic, k, v):
  """Append a value to dictionary (dic)
  if it gives a key error, create a new list.
  """
  try:
    v=float(v)
  except: 
    pass
  try:
    dic[k].append(v)
  except KeyError:
    dic[k]=[v]

def add_figure(report, line_index, image_path, caption=""):
  if os.path.isfile(image_path):
    report.insert(line_index, "\\begin{figure}[ht] \n")
    line_index+=1
    report.insert(line_index, "\\includegraphics[width=\\textwidth]{"+image_path+"}\n")
    line_index+=1
    if len(caption)==0: caption=image_path.split('/')[-3].replace('_',' ')
    report.insert(line_index, "\\caption{"+caption.replace('_',' ')+"} \n")
    # report.insert(line_index, "\\caption{"+image_path.split('/')[-3].replace('_',' ')+"/"+image_path.split('/')[-2].replace('_',' ')+": "+os.path.basename(image_path).replace('_',' ').split('.')[0]+"} \n")
    line_index+=1   
    report.insert(line_index, "\\end{figure} \n")
    line_index+=1
  else:
    print("figure not found: {0}".format(image_path))
  return report, line_index

def add_table_val(row, key):
  """takes the row-results, 
  checks for a key and a list of results,
  returns the mean and std (func) of the list as string."""
  if key in row.keys():
    if k == 'success': # show percentage
      return "{0:0.0f} \% ({1:0.2f})".format(np.mean(row[key])*100, np.std(row[key]))
    else:
      return "{0:0.2f} ({1:0.2f})".format(np.mean(row[key]), np.std(row[key]))
  else:
    return ''

#--------------------------------------------------------------
#
# STEP 1: parse arguments and get log folders
#
#--------------------------------------------------------------

parser = argparse.ArgumentParser(description='Get results, combine them and save them in a pdf send to me.')
parser.add_argument('--home', default='/esat/opal/kkelchte/docker_home', type=str, help='Define the root directory: default is /esat/opal/kkelchte/docker_home/tensorflow/log')
parser.add_argument('--summary_dir', default='tensorflow/log/', type=str, help='Define the root directory: default is /esat/opal/kkelchte/docker_home/tensorflow/log')
parser.add_argument('--mother_dir', default='', type=str, help='if all runs are grouped in one mother directory in log: e.g. depth_q_net')
parser.add_argument('--startswith', default='', type=str, help='Define sub folders in motherdir to parse.')
parser.add_argument('--endswith', default='', type=str, help='Define sub folders in motherdir to parse.')

FLAGS, others = parser.parse_known_args()

if len(FLAGS.mother_dir) == len(FLAGS.startswith) == len(FLAGS.endswith) == 0:
  print("Missing log folder instructions. Please specify mother_dir / startswith / endswith argument.")
  sys.exit(1)
print("\nSettings:")
for f in FLAGS.__dict__: print("{0}: {1}".format( f, FLAGS.__dict__[f]))
print("Others: {0}".format(others))

log_root = FLAGS.home+'/'+FLAGS.summary_dir
if 'tf_log' in os.listdir(log_root+FLAGS.mother_dir) or ('nn_ready' in os.listdir(log_root+FLAGS.mother_dir) and 'fsm_log' in os.listdir(log_root+'/'+FLAGS.mother_dir)):
  # mother_dir is the logfolde to parse
  log_folders = [log_root+FLAGS.mother_dir]
else:
  # subfolders of mother dir should be parsed
  log_folders = sorted([ log_root+FLAGS.mother_dir+'/'+d for d in os.listdir(log_root+FLAGS.mother_dir) if (len(d) == 1 or d.startswith(FLAGS.startswith) or d.endswith(FLAGS.endswith)) and (os.path.isfile(log_root+FLAGS.mother_dir+'/'+d+'/nn_ready') or os.path.isfile(log_root+FLAGS.mother_dir+'/'+d+'/tf_log'))])

if len(log_folders)==0:
  print("Woops, could not find anything "+log_root+FLAGS.mother_dir+" that startswith "+FLAGS.startswith+" and endswith "+FLAGS.endswith+" and has an nn_ready log.")
  sys.exit(1)
else:
  print("Parsing "+str(len(log_folders))+" log_folders.")

# store keys related at run frequency and not at epoch frequency
run_keys=[]

#--------------------------------------------------------------------------------
#
# STEP 2: parse all info from nn_log and nn_ready files and save it in dictionary
#
#--------------------------------------------------------------------------------

results = {}

run_images={}

for folder_index, folder in enumerate(sorted(log_folders)):
  print("\n {0}/{1}: {2} \n".format(folder_index+1, len(log_folders),folder))
  results[folder] = {}
  run_images[folder]=[]
  raw_log=tablib.Databook()
  # Parse online log_files: nn_ready and nn_log
  for file in ['nn_ready','nn_log','tf_log']:
    try:
      log_file=open(folder+'/'+file,'r').readlines()
    except:
      continue
    headers=[]
    for line in log_file:
      if len(line.split(',')) > 2:
        for term in line.split(','):
          try:
            if len(term.split(":")) == 2:
              save_append(results[folder], term.split(':')[0].strip(), term.split(':')[1].strip())
              if term.split(':')[0].strip() not in headers: headers.append(term.split(':')[0].strip())
              if file=='nn_ready':
                run_keys.append(term.split(':')[0].strip())
          except:
            print("[{0}] failed to parse {1} term: {2}".format(os.path.basename(folder), log_file, term))
    # save data in csv
    data = tablib.Dataset(headers=headers)

    # loop again over data and fill it in table
    if len(headers) == 0: continue
    for i in range(max([len(results[folder][h]) for h in headers])):
      datapoint=[]
      for h in headers:
        try:
          datapoint.append(results[folder][h][i])
        except:
          datapoint.append('')
      data.append(datapoint)
    raw_log.add_sheet(data)
    # open(folder+'/'+file+'.xls','wb').write(data.xls)
  if raw_log.size != 0:
    open(folder+'/log.xls','wb').write(raw_log.xls)


  # parse current condor host from events.file.name
  try:
    host=[f.split('.')[4] for f in os.listdir(folder) if f.startswith('events')][0]
    save_append(results[folder], 'host', host)
  except:
    pass
    
  # todo: add visualization maps
  # ...

  # add run images
  if os.path.isdir(folder+'/runs'):
    run_images[folder].extend([folder+'/runs/'+f for f in sorted(os.listdir(folder+'/runs')) if f.endswith('png')])

  print("Overview parsed information: ")
  for k in sorted(results[folder].keys()):
    print("{0}: {1} values".format(k, len(results[folder][k]) if len(results[folder][k]) != 1 else results[folder][k]))

# write result dictionary in json file
if os.path.isfile(log_root+FLAGS.mother_dir+'/results.json'):
  os.rename(log_root+FLAGS.mother_dir+'/results.json', log_root+FLAGS.mother_dir+'/_old_results.json')
with open(log_root+FLAGS.mother_dir+'/results.json','w') as out:
  json.dump(results,out,indent=2, sort_keys=True)


#--------------------------------------------------------------------------------
#
# STEP 3: prepare pdf file 
#
#--------------------------------------------------------------------------------
# Step 1: copy template pdf-latex into projects log folder
current_dir=os.path.dirname(os.path.realpath(__file__))
if not os.path.isdir(log_root+FLAGS.mother_dir+'/report'): 
  os.mkdir(log_root+FLAGS.mother_dir+'/report')

# if there is an older report, move this to _old_ (and overwrite even older one)
if os.path.isfile(log_root+FLAGS.mother_dir+'/report/report.pdf'):
  os.rename(log_root+FLAGS.mother_dir+'/report/report.pdf', log_root+FLAGS.mother_dir+'/report/_old_report.pdf')

for f in os.listdir(current_dir+'/template'):
  shutil.copyfile(current_dir+'/template/'+f, log_root+FLAGS.mother_dir+'/report/'+f)

latex_file=open("{0}/report/report.tex".format(log_root+FLAGS.mother_dir), 'r')
report = latex_file.readlines()
latex_file.close()

for l in report:
  if 'TEMPLATE REPORT' in l:
    report[report.index(l)]=l.replace('TEMPLATE REPORT',FLAGS.mother_dir.replace('_',' '))


#--------------------------------------------------------------------------------
#
# STEP 4: fill in figures
#
#--------------------------------------------------------------------------------

# find correct location to fill in the figures
line_index=0
for l in report:
  if 'INSERTFIGURES' in l: 
    line_index=report.index(l)
report[line_index] = ""

# merge all keys together
all_keys=[]
for f in log_folders:
  all_keys+=results[f].keys()
all_keys=list(set(all_keys))

# group interesting keys and leave out some keys to avoid an overdose of information
black_keys=["run_delay_std_control", "run_delay_std_image", 'Distance_current_test_esatv3', 'Distance_furthest_test_esatv3', 'run']
for k in black_keys:
  if k in all_keys:
    all_keys.remove(k)

for key in sorted(all_keys):
  # add one plot of offline training with validation accuracy against training accuracy
  plt.clf()
  plt.cla()
  plt.close()
  fig=plt.figure(figsize=(10,10))
  legend=[]
  all_fail=True
  for i,l in enumerate(log_folders): #loop over log_folders
    try:
      color=(1.-(i+0.)/len(log_folders), 0.1, (i+0.)/len(log_folders))
      plt.plot(range(len(results[l][key])),results[l][key],color=color)
      legend.append(mpatches.Patch(color=color, label=os.path.basename(l)))
      if len(results[l][key]) > 2 and type(results[l][key][0]) == float: 
        all_fail=False # in case all models have only 2 values or no float values don't show
    except Exception as e:
      print e
      pass
  if not all_fail:
    plt.xlabel("Run" if key in run_keys else "Epoch")
    plt.ylabel(key)
    plt.legend(handles=legend)
    fig_name=log_root+FLAGS.mother_dir+'/report/'+key+'.jpg'
    plt.savefig(fig_name,bbox_inches='tight')
    report, line_index = add_figure(report, line_index, fig_name, FLAGS.mother_dir)

# add runs if they are available:
report.insert(line_index,"\\section{RUNS}\n")
for folder in run_images.keys():
  report.insert(line_index,"\\section{RUNS}\n")
  for im in run_images[folder]:
    report, line_index = add_figure(report, line_index, im, caption=os.path.basename(im).replace('_',' '))

#--------------------------------------------------------------------------------
#
# STEP 5: fill in tables
#
#--------------------------------------------------------------------------------

for l in report:
  if 'INSERTTABLES' in l: 
    line_index=report.index(l)
report[line_index] = ""

# table mainly will contain those variables of which only one or few are available
table_keys=['Distance_current_test_esatv3', 'Distance_furthest_test_esatv3','host']

start_table="\\begin{tabular}{|l|"+len(log_folders)*'c'+"|}\n"
report.insert(line_index, start_table)
line_index+=1
report.insert(line_index, "\\hline\n")
line_index+=1

table_row="model: "
for m in log_folders: table_row="{0} & {1} ".format(table_row, os.path.basename(m).replace('_', ' '))
table_row="{0} \\\\ \n".format(table_row)
report.insert(line_index, table_row)
line_index+=1
report.insert(line_index, "\\hline \n")
line_index+=1

for key in sorted(table_keys):
  # add keys at each row with filling in model's value in each column
  table_row="{0} ".format(key.replace('_',' '))
  for m in log_folders:
    try:
      if isinstance(results[m][key], collections.Iterable):
        if type(results[m][key][0]) == float: #multiple floats --> take mean
          table_row="{0} & {1:0.3f} ({2:0.3f}) ".format(table_row, np.mean(results[m][key]), np.std(results[m][key]))
        else: #multiple strings
          for v in results[m][key]:  table_row="{0} & {1} ".format(table_row, v)
      else: #one value
        table_row="{0} & {1} ".format(table_row, results[m][key])
    except KeyError:
      pass
  if len(table_row) == len(key)+1: #don't add line if no information is there
    continue
  else:
    table_row="{0} \\\\ \n".format(table_row)
    report.insert(line_index, table_row)
    line_index+=1
  
report.insert(line_index, "\\hline \n")
line_index+=1
# insert 
report.insert(line_index, "\\end{tabular} \n")
line_index+=1
report.insert(line_index, "\n")
line_index+=1
# Add for each model one trajectory
report.insert(line_index, "\\newpage \n")
line_index+=1


#--------------------------------------------------------------------------------
#
# STEP 6: compile pdf and send version by mail
#
#--------------------------------------------------------------------------------

latex_file=open("{0}/report/report.tex".format(log_root+FLAGS.mother_dir), 'w')
for l in report: latex_file.write(l)
latex_file.close()

exit=subprocess.call(shlex.split("pdflatex -output-directory {0}/report {0}/report/report.tex".format(log_root+FLAGS.mother_dir))) 
# if not exit == 0: #in case pdf creation fails quit and start somewhere else
#   sys.exit(exit)

# Step 5: send it with mailx
mailcommand="mailx -s {0} -a {1} ".format(log_root+FLAGS.mother_dir, log_root+FLAGS.mother_dir+'/report/report.pdf')
for f in log_folders: 
  if os.path.isfile(f+'/log.xls'): mailcommand+=" -a {0}/log.xls".format(f)
p_msg = subprocess.Popen(shlex.split("echo {0} : {1} is finished.".format(time.strftime("%Y-%m-%d_%I:%M:%S"), log_root+FLAGS.mother_dir)), stdout=subprocess.PIPE)
p_mail = subprocess.Popen(shlex.split(mailcommand+" klaas.kelchtermans@esat.kuleuven.be"),stdin=p_msg.stdout, stdout=subprocess.PIPE)
print(p_mail.communicate())

# wait a second.
time.sleep(5)

# Step 6: put report also in archive
shutil.copyfile(log_root+FLAGS.mother_dir+'/report/report.pdf', '{0}/{1}archive/{2}.pdf'.format(FLAGS.home,FLAGS.summary_dir, FLAGS.mother_dir.replace('/','_')) )

