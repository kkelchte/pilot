#!/usr/bin/python

import os,shutil,sys, time
import numpy as np
import subprocess, shlex
import json

import collections

import argparse
import xml.etree.cElementTree as ET


import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tablib

"""
Parse results from all subfolders of mother_dir

 
 Exit codes:
 1: no log folders found
"""

#--------------------------------------------------------------
#
# Utility functions
#
#--------------------------------------------------------------
def clean_values(x,y, cutend=-1):
  """
  Avoid doubles x-values by ensuring a monotonic increase in x, and ignoring the corresponding y.
  Loop from back to front, if x-value is not lower than previous value, ignore it.
  Working from back to front ensures correct (latest) values are used.
  Return clean x and y without doubles.
  """
  # cut end:
  shortest_list=x if len(x) < len(y) else y
  if cutend != -1:
    new_x=[]
    new_y=[]
    for i in range(len(shortest_list)):
      if x[i]<cutend:
        new_x.append(x[i])
        new_y.append(y[i])
  else:
    new_x=x
    new_y=y
  clean_y=[]
  clean_x=[]
  x=list(reversed(new_x))
  y=list(reversed(new_y))
  shortest_list=x if len(x) < len(y) else y
  previous_x=999999999
  # for i,v in enumerate(x):
  for i in range(len(shortest_list)):
    if x[i]<previous_x:
        clean_x.append(x[i])
        clean_y.append(y[i])
        previous_x=x[i]
    # else:
    #     print("ignore {}".format(v))
  return list(reversed(clean_x)), list(reversed(clean_y))


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


#--------------------------------------------------------------
#
# STEP 1: parse arguments and list folders
#
#--------------------------------------------------------------

parser = argparse.ArgumentParser(description='Get results, combine them and save them in a pdf send to me.')
parser.add_argument('--home', default='/esat/opal/kkelchte/docker_home', type=str, help='Define the root directory: default is /esat/opal/kkelchte/docker_home/tensorflow/log')
parser.add_argument('--summary_dir', default='tensorflow/log/', type=str, help='Define the root directory: default is /esat/opal/kkelchte/docker_home/tensorflow/log')
parser.add_argument('--mother_dir', default='', type=str, help='if all runs are grouped in one mother directory in log: e.g. depth_q_net')
parser.add_argument('--blog_destination', default='', type=str, help='if image should be copied to blog imgs, specify the name.')
parser.add_argument('--log_folders', default=[],nargs='+', help="Define sub folders in motherdir to parse.")
parser.add_argument('--legend_names', default=[],nargs='+', help="Define the folder legends.")
parser.add_argument('--tags', default=[],nargs='+', help="Select certain tag within log file that needs to be combined.")
parser.add_argument('--subsample', default=1, type=int, help='To avoid cluttered images, subsample data making graph more smooth.')
parser.add_argument('--cutend', default=-1, type=int, help='Cut list of data earlier to cut the convergence tail.')
parser.add_argument('--title', default='', type=str, help='Define title of graph.')

FLAGS, others = parser.parse_known_args()

if len(FLAGS.mother_dir) == len(FLAGS.log_folders) == 0:
  print("Missing log folder instructions. Please specify mother_dir / startswith / endswith argument.")
  sys.exit(1)
print("\nSettings:")
for f in FLAGS.__dict__: print("{0}: {1}".format( f, FLAGS.__dict__[f]))
print("Others: {0}".format(others))

log_root = FLAGS.home+'/'+FLAGS.summary_dir
log_folders=[log_root+f for f in FLAGS.log_folders]

if len(FLAGS.log_folders)==0:
  log_folders=[]
  for root, dirs, files in os.walk(log_root+FLAGS.mother_dir):
    dirs[:]=[d for d in dirs if not d[0]=='.']
    if ('tf_log' in os.listdir(root) or ('nn_ready' in os.listdir(root) and 'fsm_log' in os.listdir(root))):
      log_folders.append(root)
  log_folders=sorted(log_folders)
  # log_folders=sorted([d[0] for d in os.walk(log_root+FLAGS.mother_dir) if not os.path.basename(d[0]).startswith('.') and ('tf_log' in os.listdir(d[0]) or ('nn_ready' in os.listdir(d[0]) and 'fsm_log' in os.listdir(d[0])))])
elif sum([os.path.isfile(f+'/tf_log') for f in log_folders]) != len(log_folders):
  log_folders=[d[0] for folder in FLAGS.log_folders for d in os.walk(log_root+folder) if 'tf_log' in os.listdir(d[0]) or ('nn_ready' in os.listdir(d[0]) and 'fsm_log' in os.listdir(d[0]))]

if len(log_folders)==0:
  print("Woops, could not find anything "+log_root+FLAGS.mother_dir+" and has an nn_ready log.")
  sys.exit(1)
else:
  print("Parsing "+str(len(log_folders))+" log_folders.")

colors=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
#------------------------------------------------------------------------------------------
#
# STEP 2: parse all info from tf_log, nn_log and nn_ready files and save it in dictionary
#
#------------------------------------------------------------------------------------------

results = {}

for folder_index, folder in enumerate(sorted(log_folders)):
  print("\n {0}/{1}: {2} \n".format(folder_index+1, len(log_folders),folder))
  results[folder] = {}
  # Parse online log_files: nn_ready and nn_log
  for file in ['nn_ready','nn_log','tf_log']:
    try:
      log_file=open(folder+'/'+file,'r').readlines()
    except:
      continue
    for line in log_file:
      if len(line.split(',')) > 2:
        for term in line.split(','):
          try:
            if len(term.split(":")) == 2:
              save_append(results[folder], term.split(':')[0].strip(), term.split(':')[1].strip())
              if file=='nn_ready':
                run_keys.append(term.split(':')[0].strip())
          except:
            print("[{0}] failed to parse {1} term: {2}".format(os.path.basename(folder), log_file, term))

  print("Overview parsed information: ")
  for k in sorted(results[folder].keys()):
    print("{0}: {1} values".format(k, len(results[folder][k]) if len(results[folder][k]) != 1 else results[folder][k]))

# write result dictionary in json file
# if os.path.isfile(log_root+FLAGS.mother_dir+'/results.json'):
#   os.rename(log_root+FLAGS.mother_dir+'/results.json', log_root+FLAGS.mother_dir+'/_old_results.json')
# with open(log_root+FLAGS.mother_dir+'/results.json','w') as out:
#   json.dump(results,out,indent=2, sort_keys=True)


#--------------------------------------------------------------------------------
#
# STEP 3: fill in xls
#
#--------------------------------------------------------------------------------
print FLAGS.tags

# loop again over data and fill it in table
for t in FLAGS.tags:
  print("saving {0}".format(t))
  # save data in csv
  raw_log=tablib.Databook()
  data = tablib.Dataset(headers=log_folders)

  # detect longest row:
  for r in range(max([len(results[folder][t]) for folder in log_folders if t in results[folder].keys()])):
    datapoint=[]
    for folder in log_folders:
      try:
        datapoint.append(results[folder][t][r])
      except:
        datapoint.append('')
    data.append(datapoint)
  raw_log.add_sheet(data)

  if raw_log.size != 0:
    open(log_root+FLAGS.mother_dir+'/combined_'+t+'.xls','wb').write(raw_log.xls)


#--------------------------------------------------------------------------------
#
# STEP 4: create figure
#
#--------------------------------------------------------------------------------

# group interesting keys and leave out some keys to avoid an overdose of information
# fig_name=''
for tag in sorted(FLAGS.tags):
  # add one plot of offline training with validation accuracy against training accuracy
  plt.clf()
  plt.cla()
  plt.close()
  fig=plt.figure(figsize=(5,5))
  legend=[]
  all_fail=True
  for folder_index,l in enumerate(log_folders): #loop over log_folders
    print l
    try:
      # color=(1.-(folder_index+0.)/len(log_folders), 0.1, (folder_index+0.)/len(log_folders))
      color=colors[folder_index%len(colors)]
      if len(FLAGS.legend_names) == len(log_folders):
        label=FLAGS.legend_names[folder_index]
      else:
        label=os.path.basename(l)
      legend.append(mpatches.Patch(color=color, label=label.replace('_', ' ')))
      if 'run' in results[l].keys():
        x,y=clean_values(list(results[l]['run']),list(results[l][tag]), cutend=FLAGS.cutend)
        plt.plot(x[::FLAGS.subsample],y[::FLAGS.subsample],color=color,linewidth=1, linestyle='--' if label=='ref' else '-')
      else:
        # plt.plot(range(len(results[l][key])),results[l][key],color=color)
        plt.plot(range(len(results[l][tag][:FLAGS.cutend]))[::FLAGS.subsample],results[l][tag][:FLAGS.cutend][::FLAGS.subsample],color=color)
      if len(results[l][tag][:FLAGS.cutend]) > 2 and type(results[l][tag][0]) == float: 
        all_fail=False # in case all models have only 2 values or no float values don't show
    except Exception as e:
      print e
      pass
  if not all_fail:
    plt.xlabel("Step")
    # plt.xlabel("Run" if tag in FLAGS.tags else "Epoch")
    if FLAGS.title: plt.title(FLAGS.title.replace('_',' '))
    ylabel=tag
    if 'imitation_learning' in tag:
      ylabel=tag.replace('imitation_learning', 'MSE')
    plt.ylabel(ylabel.replace('_',' '))
    if 'accuracy' in tag and np.amin(results[l][tag][:FLAGS.cutend]) > 0.5:
      # plt.ylabel('Accuracy')
      plt.ylim((0.5,1))
      # plt.ylim((0.7,1))
    
    # plt.ylim((0.,0.5))

    plt.legend(handles=legend)
    if FLAGS.mother_dir!='':
      fig_name=log_root+FLAGS.mother_dir+'/'+tag+'.jpg'  
    else:
      union_folder=''
      for p in log_folders[0].split('/'):
        if p in log_folders[-1] and os.path.isdir(union_folder+'/'+p):
          union_folder+='/'+p if len(p)!=0 else ''
        else: break
      fig_name=union_folder+'/'+tag+'.jpg'  
    
    plt.savefig(fig_name,bbox_inches='tight')
    command="display {0}".format(fig_name)
    print(command)
    subprocess.call(shlex.split(command))
    if FLAGS.blog_destination != "":
      command="cp {0} /users/visics/kkelchte/blogs/kkelchte.github.io/imgs/{1}_{2}.jpg".format(fig_name, FLAGS.blog_destination, tag)
      subprocess.call(shlex.split(command))
      print(command)
  else:
    print("all failed for tag {}".format(tag))
