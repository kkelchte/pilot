#!/usr/bin/python
import numpy as np
import os,sys, time
import argparse
import shutil
import subprocess, shlex
import json
'''
Clean_dataset.py: 
used in DAG form with condor_online jobs to clean up a dataset after its created.
The dataset is to be used for offline training in pilot for doshico for instance.
First arguments are parsed:
  Define the rec folders in pilot_data
    --startswith
    --endswith
  Define the destination folder in pilot data
    --destination
  Define requirements for success
    --min_distance
    --min_rgb
    --min_depth
    --min_depth_rgb_diff
    --min_successes_before_merge
Second:
  - Travelled distance is parsed and check and deleted
  - RGB - Depth images are checked and possibly deleted
Third:
  - stats are printed 
  - in case of enough success merge dataset
  - create train, val and test set

exit code:
2: not enough success runs so shutting down.

'''
print("\n {0} Clean dataset.py: started.".format(time.strftime("%Y-%m-%d_%I%M")))

# 1. Parse arguments and make settings
parser = argparse.ArgumentParser(description='Clean up dataset collected by a group of recordings that loop over different runs.')
parser.add_argument("--data_root", default="pilot_data/",type=str, help="Define the root folder of the different datasets.")
parser.add_argument("--startswith", default='', type=str, help="Define how the recorders are taged at the start in pilot_data.")
parser.add_argument("--endswith", default='', type=str, help="Define how the recorders are taged at the end in pilot_data.")
parser.add_argument("--destination", default='new_dataset', type=str, help="Define the name of the final dataset.")
parser.add_argument("--min_distance", default=-1, type=int, help="minimum allowed distance travelled: {'default':10,'sandbox':10, 'canyon':35, 'forest': 50} ")
parser.add_argument("--min_rgb", default=-1, type=int, help="minimum number of rgb images")
parser.add_argument("--max_rgb", default=-1, type=int, help="maximum number of rgb images")
parser.add_argument("--min_depth", default=-1, type=int, help="minimum number of depth images")
parser.add_argument("--max_depth_rgb_difference", default=-1, type=int, help="maximum difference between number of rgb images and depth images.")
parser.add_argument("--minimum_number_of_success", default=1, type=int, help="minimum number of success runs for continuing the data creation.")
parser.add_argument("--val_len", default=10, type=int, help="length of the validation set in number of runs.")
parser.add_argument("--test_len", default=10, type=int, help="length of the test set in number of runs.")

FLAGS, others = parser.parse_known_args()

min_rgb={'default':10,'sandbox':10, 'canyon':100, 'forest':100}
max_rgb={'default':5000,'sandbox':2000, 'canyon':3000, 'forest': 3000}

min_distance={'default':1,'sandbox':1, 'canyon':3, 'forest': 3}

if FLAGS.data_root[0] != '/':  # 2. Pilot_data directory for saving data
  FLAGS.data_root=os.environ['HOME']+'/'+FLAGS.data_root

print("\nSettings:")
for f in FLAGS.__dict__: print("{0}: {1}".format( f, FLAGS.__dict__[f]))

# 2. Go over all runs and check if all is well, if not delete runs
data_dirs=sorted([FLAGS.data_root + d for d in os.listdir(FLAGS.data_root) if d.startswith(FLAGS.startswith) and d.endswith(FLAGS.endswith)])
runs={d:sorted([d+'/'+r for r in os.listdir(d) if r.startswith('0') and os.path.isdir(d+'/'+r)]) for d in data_dirs}

print("\n {0} Clean dataset.py: cleaning {1} runs from {2} dirs.".format(time.strftime("%Y-%m-%d_%I%M"), sum([len(runs[d]) for d in runs.keys()]), len(runs.keys())))


for d_i, d in enumerate(runs.keys()):
  print("\n{0} directory: {1}  ( {2}/{3} ) ".format(time.strftime("%Y-%m-%d_%I%M"), os.path.basename(d), d_i+1, len(runs.keys())))
  to_be_removed_from_runs=[]    
  for r_i, r in enumerate(runs[d]):
    # if r_i%10==0:
    #   print("run: {0}/{1}".format(r_i+1,len(runs[d])))
    print("run: {0}/{1} {2}".format(r_i+1,len(runs[d]), r))
    # get positions
    # positions={int(l.split('[')[0]):np.array([float(l.split('[')[1][:-2].split(',')[0]),float(l.split('[')[1][:-2].split(',')[1]),float(l.split('[')[1][:-2].split(',')[2])]) for l in open(r+'/position_info.txt','r').readlines()[2:]}
    positions={int(l.split('[')[0]):np.array([float(l.split('[')[1][:-2].split(',')[0]),float(l.split('[')[1][:-2].split(',')[1])]) for l in open(r+'/position_info.txt','r').readlines()[2:]}

    # get world name
    world_name=os.path.basename(r).split('_')[1]
    if world_name not in min_rgb.keys():
      world_name='default'
    ## Parse total flying distance and remove folder if it is too low
    travelled_distance = 0
    last_position=np.array([0,0])
    for p in positions.keys():
      travelled_distance+=(np.sqrt((positions[p]-last_position)**2).mean())
      last_position = positions[p]
    if FLAGS.min_distance!= -1:
      minimum = FLAGS.min_distance
    else:
      minimum=min_distance[world_name]
    print("travelled_distance: {}".format(travelled_distance))
    if travelled_distance < minimum:
      # remove run folder
      print('removed: {0} due to no far enough flying distance: {1:0.2f} < {2:0.2f}'.format(r, travelled_distance, minimum))
      shutil.rmtree(r)
      to_be_removed_from_runs.append(r_i)
      continue

    ## Ensure number of RGB and Depth images is within range and don't differ too much
    num_rgb=len([f for f in os.listdir(r+'/RGB') if f.endswith('jpg')])
    num_depth=len([f for f in os.listdir(r+'/Depth') if f.endswith('jpg')])
    
    rgb_min=FLAGS.min_rgb if FLAGS.min_rgb != -1 else min_rgb[world_name] 
    rgb_max=FLAGS.max_rgb if FLAGS.max_rgb != -1 else max_rgb[world_name]


    if (num_rgb < rgb_min) or (num_rgb > rgb_max) or (abs(num_depth-num_rgb) > FLAGS.max_depth_rgb_difference and FLAGS.max_depth_rgb_difference != -1):
      # remove run folder
      print("Removed {5} due to: {2} > {0} imgs > {1} or {3} < {4}".format(num_rgb, rgb_min, rgb_max, abs(num_depth-num_rgb),  FLAGS.max_depth_rgb_difference,r))
      shutil.rmtree(r)
      to_be_removed_from_runs.append(r_i)
      continue

    ## Ensure control_info.txt exists
    if not os.path.isfile("{0}/control_info.txt".format(r)):
      # remove run folder
      print('removed: {0} due to no control info.'.format(r))
      shutil.rmtree(r)
      to_be_removed_from_runs.append(r_i)
      continue

    # Delete images taken without flying for more than 0.25m
    # parse first image taken at 0.25m from startingposition (0,0)
    travelled_distance = 0
    last_position=np.array([0,0])
    index=0
    while travelled_distance < 0.25:
      try:
        travelled_distance+=(np.sqrt((positions[index]-last_position)**2).mean())
      except:
          pass
      else:
        last_position=positions[index]
      index+=1
    # remove all images before
    for i in positions.keys():
      if i < index:
        # print("removing image:{0}/RGB/{1:010d}.jpg".format(r,i))
        try: os.remove("{0}/RGB/{1:010d}.jpg".format(r,i))
        except: pass
        try: os.remove("{0}/Depth/{1:010d}.jpg".format(r,i))
        except: pass
  # remove the runs that were deleted in reverse order not to mess up the indices
  for r_i in reversed(to_be_removed_from_runs): del runs[d][r_i]



# 3. Print stats and decide whether to merge and create train/val/test sets.
for w in '', 'canyon', 'forest', 'sandbox':
  print('{}'.format(w if w!='' else 'total'))
  num_runs=sum([len([r for r in runs[d] if w in r]) for d in runs.keys()])
  print("Runs in {1}: {0}".format(num_runs, w if w!='' else 'total'))
  num_rgb=sum([len(os.listdir(r+'/RGB')) for d in runs.keys() for r in runs[d] if w in r])
  print("RGBs in {1}: {0}".format(num_rgb,w if w!='' else 'total'))
  

if sum([len(runs[d]) for d in runs.keys()]) < FLAGS.minimum_number_of_success:
  print("Had only {0} success runs instead of {1} so shutting down.".format(sum([len(runs[d]) for d in runs.keys()]), FLAGS.minimum_number_of_success))
  sys.exit(2)

# 4. Merge datasets together.
print("\n {0} Clean dataset.py: merging data in {1}.".format(time.strftime("%Y-%m-%d_%I%M"), FLAGS.destination))


# merge different datasets together
# if dataset already exists but has default name, delete:
dataset=FLAGS.data_root+FLAGS.destination
if os.path.isdir(dataset) and FLAGS.destination == 'new_dataset':
  print("remove old dataset.")
  shutil.rmtree(dataset)
  time.sleep(1)
# if dataset already exists add up runs from index
index=0
if os.path.isdir(dataset):
  index=len([r for r in os.listdir(dataset) if r.startswith('0') and os.path.isdir(dataset+'/'+r)])
  print("Found dataset with {0} runs.".format(index))
else: # if it does not exist, create
  try:
    os.makedirs(dataset)
    print("Created new dataset.")
  except:
    pass
for d_i, d in enumerate(runs.keys()):
  print("Copy: {0}/{1} directories.".format(d_i, len(runs.keys())))
  for r in runs[d]:
    world_name=os.path.basename(r).split('_')[1]
    dest="{0}/{1:05d}_{2}".format(dataset, index, world_name)
    # print dest
    subprocess.call(shlex.split('mv '+r+' '+dest))
    index+=1

# clean up all previous dirs
for d in runs.keys(): shutil.rmtree(d)

# 5. Create train, val and test sets.
# if there is already a train/test/val set delete and resample
print("\n {0} Clean dataset.py: creating train/val/test_set.txt in {1}.".format(time.strftime("%Y-%m-%d_%I%M"), FLAGS.destination))
try:
  os.remove("{0}/train_set.txt".format(dataset))
except: pass
try:
  os.remove("{0}/val_set.txt".format(dataset))
except: pass
try:
  os.remove("{0}/test_set.txt".format(dataset))
except: pass
else:
  print("removed txt files in {}".format(dataset))

total_set=sorted(["{0}/{1}".format(dataset,r) for r in os.listdir(dataset) if r.startswith('0') and os.path.isdir("{0}/{1}".format(dataset,r))])
val_set=[]
test_set=[]
train_set=[]
# get updated stats over total set
stats={"Date":time.strftime("%Y-%m-%d_%I%M")}
stats["total"]={}
for w in '', 'canyon', 'forest', 'sandbox':
  num_runs=len([r for r in total_set if w in r])
  num_rgb=sum([len(os.listdir(r+'/RGB')) for r in total_set if w in r])
  w_key = w if len(w) != 0 else 'all'
  stats["total"][w_key]={"RGB":num_rgb, "runs":num_runs}

# select from total list
if len(total_set) < FLAGS.val_len + FLAGS.test_len + 10:
  FLAGS.val_len = 1
  FLAGS.test_len = 1
for vn in range(FLAGS.val_len):
  selected=np.random.choice(total_set)
  total_set.remove(selected)
  val_set.append(selected)
for tn in range(FLAGS.test_len):
  selected=np.random.choice(total_set)
  total_set.remove(selected)
  test_set.append(selected)
train_set=total_set
# write away:
train_file=open(dataset+'/train_set.txt','w')
for l in train_set: train_file.write(l+'\n')
train_file.close()
val_file=open(dataset+'/val_set.txt','w')
for l in val_set: val_file.write(l+'\n')
val_file.close()
test_file=open(dataset+'/test_set.txt','w')
for l in test_set: test_file.write(l+'\n')
test_file.close()
# add stats for train, val, test
data_set={"train":train_set, "val":val_set, "test":test_set}
for data in "train", "val", "test":
  stats[data]={}
  for w in '', 'canyon', 'forest', 'sandbox':
    num_runs=len([r for r in data_set[data] if w in r])
    num_rgb=sum([len(os.listdir(r+'/RGB')) for r in data_set[data] if w in r])
    w_key = w if len(w) != 0 else 'all'
    stats[data][w_key]={"RGB":num_rgb, "runs":num_runs}
# write stats file
with open(dataset+'/stats.json','w') as out:
  json.dump(stats,out,indent=2, sort_keys=True)


print("\n {0} Clean dataset.py: finished.".format(time.strftime("%Y-%m-%d_%I%M")))
