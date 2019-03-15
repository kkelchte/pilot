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
Parse results from all subfolders of mother_dir

 
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
# STEP 1: parse arguments and list folders
#
#--------------------------------------------------------------

parser = argparse.ArgumentParser(description='Get results, combine them and save them in a pdf send to me.')
parser.add_argument('--home', default='/esat/opal/kkelchte/docker_home', type=str, help='Define the root directory: default is /esat/opal/kkelchte/docker_home/tensorflow/log')
parser.add_argument('--summary_dir', default='tensorflow/log/', type=str, help='Define the root directory: default is /esat/opal/kkelchte/docker_home/tensorflow/log')
parser.add_argument('--mother_dir', default='', type=str, help='if all runs are grouped in one mother directory in log: e.g. depth_q_net')
parser.add_argument('--log_folders', default=[],nargs='+', help="Define sub folders in motherdir to parse.")
parser.add_argument('--legend_names', default=[],nargs='+', help="Define the folder legends.")
parser.add_argument('--tags', default=[],nargs='+', help="Select certain tag within log file that needs to be combined.")
parser.add_argument('--subsample', default=1, type=int, help='To avoid cluttered images, subsample data making graph more smooth.')
parser.add_argument('--cutend', default=-1, type=int, help='Cut list of data earlier to cut the convergence tail.')

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
  log_folders=sorted([d[0] for d in os.walk(log_root+FLAGS.mother_dir) if 'tf_log' in os.listdir(d[0]) or ('nn_ready' in os.listdir(d[0]) and 'fsm_log' in os.listdir(d[0]))])


if len(log_folders)==0:
  print("Woops, could not find anything "+log_root+FLAGS.mother_dir+" that startswith "+FLAGS.startswith+" and endswith "+FLAGS.endswith+" and has an nn_ready log.")
  sys.exit(1)
else:
  print("Parsing "+str(len(log_folders))+" log_folders.")


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
for tag in sorted(FLAGS.tags):
  # add one plot of offline training with validation accuracy against training accuracy
  plt.clf()
  plt.cla()
  plt.close()
  fig=plt.figure(figsize=(5,5))
  legend=[]
  all_fail=True
  for folder_index,l in enumerate(log_folders): #loop over log_folders
    try:
      color=(1.-(folder_index+0.)/len(log_folders), 0.1, (folder_index+0.)/len(log_folders))
      plt.plot(range(len(results[l][tag][:FLAGS.cutend]))[::FLAGS.subsample],results[l][tag][:FLAGS.cutend][::FLAGS.subsample],color=color)
      if len(FLAGS.legend_names) == len(log_folders):
        label=FLAGS.legend_names[folder_index]
      else:
        label=l
      legend.append(mpatches.Patch(color=color, label=label.replace('_', ' ')))
      if len(results[l][tag][:FLAGS.cutend]) > 2 and type(results[l][tag][0]) == float: 
        all_fail=False # in case all models have only 2 values or no float values don't show
    except Exception as e:
      print e
      pass
  if not all_fail:
    plt.xlabel("Step")
    # plt.xlabel("Run" if tag in FLAGS.tags else "Epoch")
    plt.ylabel(tag.replace('_',' '))
    if 'accuracy' in tag and np.amin(results[l][tag][:FLAGS.cutend]) > 0.5:
      # plt.ylabel('Accuracy')
      plt.ylim((0.5,1))
    plt.legend(handles=legend)
    fig_name=log_root+FLAGS.mother_dir+'/'+tag+'.jpg'
    plt.savefig(fig_name,bbox_inches='tight')

  print("display {0}".format(fig_name))
