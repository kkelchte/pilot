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
import matplotlib.image as mpimg
from numpy.linalg import inv

import tablib

import numpy as np

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

def clean_values(x,y):
  """
  Avoid doubles x-values by ensuring a monotonic increase in x, and ignoring the corresponding y.
  Loop from back to front, if x-value is not lower than previous value, ignore it.
  Working from back to front ensures correct (latest) values are used.
  Return clean x and y without doubles.
  """
  clean_y=[]
  clean_x=[]
  x=list(reversed(x))
  y=list(reversed(y))
  previous_x=999999999
  for i,v in enumerate(x):
      if v<previous_x:
          clean_x.append(v)
          clean_y.append(y[i])
          previous_x=v
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

def add_figure(report, line_index, image_path, caption=""):
  if os.path.isfile(image_path):
    report.insert(line_index, "\\begin{figure}[ht] \n")
    line_index+=1
    report.insert(line_index, "\\includegraphics[width=1.2\\textwidth]{"+image_path+"}\n")
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

def combine_runs_map(motherdirs,destination):
  logroot='/esat/opal/kkelchte/docker_home/tensorflow/log'
  if not motherdirs[0].startswith('/'):  motherdirs=[logroot+'/'+md for md in motherdirs]
  origin_arrow_map = np.asarray([[0.,0.],[7.,0.],[7.,1.5],[9.,0.],[7.,-1.5],[7.,0.]])
  transformed_arrow = origin_arrow_map[:]
  rotation_gazebo_map = np.asarray([[-1,0],[0,1]])
  img_type='esatv3'
  plt.cla()
  plt.clf()
  fig,ax = plt.subplots(1,figsize=(30,30))
  ax.set_title('Position Display')

  current_image = np.zeros((1069,1322))
  implot=ax.imshow(current_image)
  
  img_file=logroot+'/../../simsup_ws/src/simulation_supervised/simulation_supervised_demo/worlds/esatv3.png'
  current_image=mpimg.imread(img_file)
  implot=ax.imshow(current_image)
  ax.axis('off')
  legend=[]
  colors=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
  success=False
  for mdindex, md in enumerate(motherdirs):
    # print md
    posfiles=sorted([os.path.join(d[0],f) for d in os.walk(md) for f in os.listdir(d[0]) if f.startswith('gt') and f.endswith('txt')])
    color=colors[mdindex%len(colors)]
    for p in posfiles:
      #     print p
      # get positions of run
      positions=[[float(t) for t in l.strip().split(',')] for l in open(p,'r').readlines()]
      for posindex, pos in enumerate(positions):
        if posindex > 270: break
        assert len(pos) == 3
        x,y=pos[0],pos[1]
        # transform arrow
        drone_gazebo_orientation = np.asarray([[np.cos(pos[2]), -np.sin(pos[2])],[np.sin(pos[2]), np.cos(pos[2])]])
        # combine with rotation gazebo_map to get orientation from drone to map
        # combine with translation
        transformation_map_to_drone = np.zeros((3,3))
        transformation_map_to_drone[2,2] = 1
        transformation_map_to_drone[0:2,2] = x,y
        transformation_map_to_drone[0:2,0:2] = inv(np.matmul(rotation_gazebo_map, drone_gazebo_orientation))
        # transformation_map_to_drone[0:2,0:2] = np.identity(2)
        # apply transformation to points in arrow
        transformed_arrow=np.transpose(np.matmul(transformation_map_to_drone,np.concatenate([np.transpose(origin_arrow_map),np.ones((1,origin_arrow_map.shape[0]))],axis=0)))
        transformed_arrow=transformed_arrow[:,:2]

        # add patch
        ax.add_patch(mpatches.Polygon(transformed_arrow,linewidth=1,edgecolor=color,facecolor='None'))
        success=True
    legend.append(mpatches.Patch(color=color, label=os.path.basename(md).replace('_', ' ')))
  plt.legend(handles=legend)
  plt.savefig(destination)
  return success

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
parser.add_argument("--dont_mail", action='store_true', help="In case no mail is necessary, add this flag.")

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
CAM_images={}

for folder_index, folder in enumerate(sorted(log_folders)):
  print("\n {0}/{1}: {2} \n".format(folder_index+1, len(log_folders),folder))
  results[folder] = {}
  run_images[folder]=[]
  CAM_images[folder]=[]
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
    save_append(results[folder], 'host', '')
    pass
    
  # add run images
  # if os.path.isdir(folder+'/runs'):
  #   run_images[folder].extend([folder+'/runs/'+f for f in sorted(os.listdir(folder+'/runs')) if f.endswith('jpg') or f.endswith('png')])

  if os.path.isdir(folder+'/CAM'):
    number_CAM_images=3
    CAM_images[folder].extend(np.random.choice([folder+'/CAM/'+f for f in sorted(os.listdir(folder+'/CAM')) if f.endswith('jpg') or f.endswith('png')]), number_CAM_images, replace=False)

    
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

# Combine run images of different runs 
fig_name=log_root+FLAGS.mother_dir+'/report/runs_combined.png'
if combine_runs_map(log_folders, fig_name):
  report, line_index = add_figure(report, line_index, fig_name, FLAGS.mother_dir)

# merge all keys together
all_keys=[]
for f in log_folders:
  all_keys+=results[f].keys()
all_keys=list(set(all_keys))

# group interesting keys and leave out some keys to avoid an overdose of information
black_keys=["run_delay_std_control", 
            "run_delay_std_image", 
            'Distance_current_test_esatv3', 
            'Distance_furthest_test_esatv3',
            'run_number']
for k in black_keys:
  if k in all_keys:
    all_keys.remove(k)

for key in sorted(all_keys):
  # skip the run index
  if key =='run': continue
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
      if 'run' in all_keys:
        x,y=clean_values(list(results[l]['run']),list(results[l][key]))
        plt.plot(x,y,color=color)
      else:
        plt.plot(range(len(results[l][key])),results[l][key],color=color)
      legend.append(mpatches.Patch(color=color, label=os.path.basename(l)))
      if len(results[l][key]) > 2 and type(results[l][key][0]) == float: 
        all_fail=False # in case all models have only 2 values or no float values don't show
    except Exception as e:
      # print e
      pass
  if not all_fail:
    plt.xlabel("Run" if key in run_keys else "Epoch")
    plt.ylabel(key)
    plt.legend(handles=legend)
    fig_name=log_root+FLAGS.mother_dir+'/report/'+key+'.png'
    plt.savefig(fig_name,bbox_inches='tight')
    report, line_index = add_figure(report, line_index, fig_name, FLAGS.mother_dir)

# add CAM images
report.insert(line_index,"\\section\{CAM\}\n")
# image_count=0
for folder in CAM_images.keys():
  # report.insert(line_index,"\\section{RUNS}\n")
  for im in CAM_images[folder]:
    report, line_index = add_figure(report, line_index, im, caption=os.path.basename(im).replace('_',' '))
  #   image_count+=1
  #   if image_count > 10: break
  # if image_count > 10: break

# add runs if they are available:
# report.insert(line_index,"\\section{RUNS}\n")
# image_count=0
# for folder in run_images.keys():
#   # report.insert(line_index,"\\section{RUNS}\n")
#   for im in run_images[folder]:
#     report, line_index = add_figure(report, line_index, im, caption=os.path.basename(im).replace('_',' '))
#     image_count+=1
#     if image_count > 10: break
#   if image_count > 10: break
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
table_keys=['Distance_current_test_esatv3', 
            'Distance_furthest_test_esatv3',
            'test_success',
            'run_imitation_loss',
            'test_accuracy',
            'validation_accuracy',
            'validation_imitation_learning',
            'host']


start_table="\\begin{tabular}{|l|"+len(log_folders)*'c'+'c'+"|}\n"
report.insert(line_index, start_table)
line_index+=1
report.insert(line_index, "\\hline\n")
line_index+=1

table_row="model: "
for m in log_folders: table_row="{0} & {1} ".format(table_row, os.path.basename(m).replace('_', ' '))
table_row="{0} & total \\\\ \n".format(table_row)
report.insert(line_index, table_row)
line_index+=1
report.insert(line_index, "\\hline \n")
line_index+=1

for key in sorted(table_keys):
  # add keys at each row with filling in model's value in each column
  table_row="{0} ".format(key.replace('_',' '))
  total_vals=[]
  for m in log_folders:
    try:
      if isinstance(results[m][key], collections.Iterable):
        if type(results[m][key][-1]) in [float,int,bool]: #multiple floats --> take mean
          table_row="{0} & {1:0.3f} ({2:0.3f}) ".format(table_row, np.mean(results[m][key]), np.std(results[m][key]))
          total_vals.append(np.mean(results[m][key]))
        else: #multiple strings
          for v in results[m][key]:  table_row="{0} & {1} ".format(table_row, v)
      else: #one value
        table_row="{0} & {1} ".format(table_row, results[m][key])
        if not isinstance(results[m][key], str):
          total_vals.append(results[m][key])
    except KeyError:
      pass
  if len(table_row) == len(key)+1: #don't add line if no information is there
    continue
  else:
    # add total column info
    if len(total_vals)!=0:
      try:
        table_row="{0} & {1:0.3f} ({2:0.3f}) ".format(table_row, np.mean(total_vals), np.std(total_vals))
      except KeyError:
        table_row="{0} & ".format(table_row)
    else:
      table_row="{0} & ".format(table_row)
      
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


# Specific offline training table:

table_keys=['test_accuracy',
            'validation_accuracy',
            'validation_imitation_learning',
            'host']

eva_folders=[f for f in log_folders if f.endswith('eva')]
train_folders=[f for f in log_folders if not f.endswith('eva')]
# if 'test_success' in results[log_folders[0]].keys() and 'run_imitation_loss' in results[log_folders[0]].keys():
good_keys=[k for k in table_keys if k in results[train_folders[0]].keys()]
for l in report:
  if 'INSERTTABLES' in l: 
    line_index=report.index(l)
report[line_index] = ""
start_table="\\begin{tabular}{|l|"
for i in range(len(good_keys)): start_table+="c|"
start_table+="}\n"
report.insert(line_index, start_table)
line_index+=1
report.insert(line_index, "\\hline\n")
line_index+=1
table_row="model"
for k in good_keys: table_row+=" & "+k.replace("_"," ")
table_row+=" \\\\ \n"
report.insert(line_index, table_row)
line_index+=1
report.insert(line_index, "\\hline\n")
line_index+=1
total_vals={}
for m in train_folders:
  table_row="{0}".format(os.path.basename(m).replace('_', ' '))
  for k in good_keys:
    try:
      if k == 'validation_accuracy': # take last value
        table_row+=" & {0}".format(results[m][k][-1])
        value=results[m][k][-1]
      elif isinstance(results[m][k], collections.Iterable):
        if type(results[m][k][-1]) in [float,int,bool]: #multiple floats --> take mean
          table_row="{0} & {1:0.3f} ({2:0.3f}) ".format(table_row, np.mean(results[m][k]), np.std(results[m][k]))
          value=np.mean(results[m][k])
        else: #multiple strings
          value=results[m][k][0]
          table_row="{0} & {1} ".format(table_row, value)
          # for v in results[m][k]:  
          #   table_row="{0} & {1} ".format(table_row, v)
      else: #one value
        table_row="{0} & {1} ".format(table_row, results[m][k])
        value=results[m][k]
    except KeyError:
      table_row+=" & "
      pass
    else:
      if not isinstance(value, str):
        if k in total_vals.keys():
          total_vals[k].append(value)
        else:
          total_vals[k]=[value]
  table_row+="\\\\ \n"
  report.insert(line_index, table_row)
  line_index+=1
  report.insert(line_index, "\\hline \n")
  line_index+=1
# insert total vals
table_row="total"
for k in good_keys:
  try:
    table_row+=" & {0:0.3f} ({1:0.3f})".format(np.mean(total_vals[k]),np.std(total_vals[k]))
  except KeyError:
    table_row+="&"
table_row+="\\\\ \n"
report.insert(line_index, table_row)
line_index+=1
report.insert(line_index, "\\hline \n")
line_index+=1
report.insert(line_index, "\\end{tabular} \n")
line_index+=1
report.insert(line_index, "\n")
line_index+=1

# specific on-policy performance table


table_keys=['Distance_current_test_esatv3',
            'test_success',
            'run_imitation_loss',
            'host']

good_keys=[k for k in table_keys if k in results[eva_folders[0]].keys()]
for l in report:
  if 'INSERTTABLES' in l: 
    line_index=report.index(l)
report[line_index] = ""
start_table="\\begin{tabular}{|l|"
for i in range(len(good_keys)): start_table+="c|"
start_table+="}\n"
report.insert(line_index, start_table)
line_index+=1
report.insert(line_index, "\\hline\n")
line_index+=1
table_row="model"
for k in good_keys: table_row+=" & "+k.replace("_"," ")
table_row+=" \\\\ \n"
report.insert(line_index, table_row)
line_index+=1
report.insert(line_index, "\\hline\n")
line_index+=1
total_vals={}
for m in eva_folders:
  table_row="{0}".format(os.path.basename(m).replace('_', ' '))
  for k in good_keys:
    try:
      if k == 'validation_accuracy': # take last value
        table_row+=" & {0}".format(results[m][k][-1])
        value=results[m][k][-1]
      elif isinstance(results[m][k], collections.Iterable):
        if type(results[m][k][-1]) in [float,int,bool]: #multiple floats --> take mean
          table_row="{0} & {1:0.3f} ({2:0.3f}) ".format(table_row, np.mean(results[m][k]), np.std(results[m][k]))
          value=np.mean(results[m][k])
        else: #multiple strings
          value=results[m][k][0]
          table_row="{0} & {1} ".format(table_row, value)
          # for v in results[m][k]:  
          #   table_row="{0} & {1} ".format(table_row, v)
      else: #one value
        table_row="{0} & {1} ".format(table_row, results[m][k])
        value=results[m][k]
    except KeyError:
      table_row+=" & "
      pass
    else:
      if not isinstance(value, str):
        if k in total_vals.keys():
          total_vals[k].append(value)
        else:
          total_vals[k]=[value]
  table_row+="\\\\ \n"
  report.insert(line_index, table_row)
  line_index+=1
  report.insert(line_index, "\\hline \n")
  line_index+=1
# insert total vals
table_row="total"
for k in good_keys:
  try:
    table_row+=" & {0:0.3f} ({1:0.3f})".format(np.mean(total_vals[k]),np.std(total_vals[k]))
  except KeyError:
    table_row+="&"
table_row+="\\\\ \n"
report.insert(line_index, table_row)
line_index+=1
report.insert(line_index, "\\hline \n")
line_index+=1
report.insert(line_index, "\\end{tabular} \n")
line_index+=1
report.insert(line_index, "\n")
line_index+=1


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

print("create pdf to {0}/report {0}/report/report.tex".format(log_root+FLAGS.mother_dir))
exit=subprocess.call(shlex.split("pdflatex -output-directory {0}/report {0}/report/report.tex".format(log_root+FLAGS.mother_dir))) 
if not exit == 0: #in case pdf creation fails quit and start somewhere else
  sys.exit(exit)

if not FLAGS.dont_mail:
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

