#!/bin/bash

# choose number of steps to take: best one by one
# in case of no step specified, all steps are done.
step="$1"

# Clean data
dataset='esat'
# worlds=('canyon' 'forest' 'sandbox')
worlds=('esat_v1' 'esat_v2')

cd /esat/opal/kkelchte/docker_home/pilot_data/$dataset

echo "STEP 1 delete runs that are not a directory or have no images"
# echo "deleting $(for d in 0* ; do if [ ! -d $d ] ; then echo $d; fi; done | wc -l) directories."
if [[ $step == 1 || -z $step ]] ; then
	for d in 0* ; do if [ ! -d $d ] ; then rm -r $d; fi; done 
	for d in 0* ; do if [ $(ls $d/RGB | wc -l) -eq 0 ] ; then rm -r $d; echo "deleted $d"; fi; done 
	if [ $step == 1 ] ; then exit; fi
fi

echo "STEP 2: Rank runs according to number of images:"
if [[ $step == 2 || -z $step ]] ; then
	for w in ${worlds[@]}; do 
		echo $w; 
		for d in 0*_$w ; do echo $d $(ls $d/RGB | wc -l); done | sort -k 2 -n | head -1
		display $(for d in 0*_$w ; do echo $d $(ls $d/RGB | wc -l); done | sort -k 2 -n | head -1 | cut -d ' ' -f 1)/runs.png
		for d in 0*_$w ; do echo $d $(ls $d/RGB | wc -l); done | sort -k 2 -n | tail -1
		display $(for d in 0*_$w ; do echo $d $(ls $d/RGB | wc -l); done | sort -k 2 -n | tail -1 | cut -d ' ' -f 1)/runs.png
	done
	if [ $step == 2 ] ; then exit; fi
fi

echo "STEP 2B remove too short trajectories or too long trajectories [NOT IMPLEMENTED]"

echo "STEP 3,4,5  mv this directory to tmp name and create new directory in pilot_data with original name"
cd ..
if [[ $step == 3 || -z $step ]] ; then
	if [ -d tmp ] ; then rm -r tmp; fi
	mv $dataset tmp
	mkdir $dataset
	if [ $step == 3 ] ; then exit; fi
fi
if [[ $step == 4 || -z $step ]] ; then
	i=0;
	for d in tmp/0* ; do
		world="$(echo $d | cut -d '_' -f 2)"
		echo "mv $d $dataset/$(printf %05d $i)_$world"
		i=$((i+1))
	done
	if [ $step == 4 ] ; then exit; fi
fi
if [[ $step == 5 || -z $step ]] ; then
	i=0;
	for d in tmp/0* ; do
		world="$(echo $d | cut -d '_' -f 2)"
		mv $d $dataset/$(printf %05d $i)_$world
		i=$((i+1))
	done
	rm -r tmp
	if [ $step == 5 ] ; then exit; fi
fi
cd $dataset

echo "STEP 6 create train, validate test files"
if [[ $step == 6 || -z $step ]] ; then
	for d in 0* ; do echo $PWD/$d >> all_set.txt; done
	cat all_set.txt | head -3 >> test_set.txt
	cat all_set.txt | head -12 | tail -9 >> val_set.txt
	tot=$(cat all_set.txt | wc -l)
	N=$((tot-12))
	cat all_set.txt | tail -$N >> train_set.txt
	rm all_set.txt
	for f in train_set.txt val_set.txt test_set.txt ; do 
		echo $f
		head -1 $f
	done
	if [ $step == 6 ] ; then exit; fi
fi

echo "STEP 7 make spread numbers over depth and RGB directory start from 0 and increment with 1"
echo
echo "DONT FORGET TO ADJUST CONTROL INFO RGB INDICES WITH CLEAN CONTROL COLLISION FILES"
echo 
if [[ $step == 7 || -z $step ]] ; then
	for r in 0*; do 
		echo $(date +%F_%H:%M) $r; 
		cd $r; 
		for d in Depth RGB Depth_predicted ; do 
			mv $d old_$d; 
			mkdir $d; 
			i=0; 
			for f in old_$d/* ; do 
				mv $f $d/$(printf %010d $i).jpg; 
				i=$((i+1)); 
			done; 
			rm -r old_$d; 
		done; c
		d ..; 
	done
	if [ $step == 7 ] ; then exit; fi
fi