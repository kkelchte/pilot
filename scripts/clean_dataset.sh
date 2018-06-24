# Clean data

dataset='canyon_turtle_scan'

cd /esat/opal/kkelchte/docker_home/pilot_data/$dataset


# Delete all runs with less than 10 images
for d in 0* ; do
	if [ $(ls $d/RGB | wc -l) -lt 20 ] ; then 
		rm -r $d; 
		echo "deleted $d"; 
	else
		if [ ! -e $d/control_info.txt ] ; then
			rm -r $d;
			echo "deleted $d";
		else
			# make sure the applied control of the model is set as control info and loaded in data
			# echo "mv control / predicted action info"
			# mv $d/control_info.txt $d/ba_info.txt
			# mv $d/predicted_info.txt $d/control_info.txt

			# get rid of the last 15 frames due to the delay of bump detection up until saving images is finished
			# echo "remove final images"
			# for i in $(ls $d/RGB/ | sort -r | head -15) ; do rm $d/RGB/$i; done

			# create collision labels
			echo "create collision labels"
			if [ -e $d/collision_info.txt ] ; then rm $d/collision_info.txt; fi
			for i in $(ls $d/RGB | cut -d '.' -f 1) ; do echo "$i 0" >> $d/collision_info.txt; done
			for l in $(tail -5 $d/collision_info.txt | cut -d ' ' -f 1); do sed -i "s/$l 0/$l 1/" $d/collision_info.txt; done
		fi
	fi
done

#________________________________________________________________
#
# In order to make a collision free dataset: rm the last 5 images
#
#________________________________________________________________

cd /esat/opal/kkelchte/docker_home/pilot_data
cp -r $dataset ${dataset}_collision_free_tmp
rm ${dataset}_collision_free_tmp/*.txt
cd ${dataset}_collision_free_tmp

# Parse the expected number of left over images to know if not too much is deleted
init_num="$(for d in 0* ; do ls $d/RGB; done | wc -l)"
run_num="$(for d in 0* ; do echo $d; done | wc -l)"
expected_num=$((init_num-run_num*5))

# Delete last 5 images
for d in 0*; do echo $d; for f in $(ls $d/RGB | tail -5); do rm $d/RGB/$f; done; done

# Check if it was successfull
current_num="$(for d in 0* ; do ls $d/RGB; done | wc -l)"
if [ $current_num != $expected_num ] ; then
	echo "Failed to extract collision free dataset."
	exit 1
else
	echo "successfully extracted collision free dataset."
fi

# Copy to correct location
cd ..
mkdir ${dataset}_collision_free
i=0 
for d in ${dataset}_collision_free_tmp/0* ; do echo "mv $d ${dataset}_collision_free/$(printf %05d $i)_canyon"; mv $d ${dataset}_collision_free/$(printf %05d $i)_canyon; i=$((i+1)); donefor d in ${dataset}_collision_free_tmp/0* ; do echo "mv $d ${dataset}_collision_free/$(printf %05d $i)_canyon"; mv $d ${dataset}_collision_free/$(printf %05d $i)_canyon; i=$((i+1)); done

# Create train, test and val set....


# create train_set.txt, test_set.txt, val_set.txt
# num_tot=$(ls | grep 0 | wc -l)
# num_train=$((num_tot-5))
# num_val=3
# num_test=2
# for d in $(ls | grep 00 | sort -r | head -$num_train); do
# 	echo "$PWD/$d" >> train_set.txt
# done

# for d in $(ls | grep 00 | head -$num_val); do 
# 	echo "$PWD/$d" >> val_set.txt
# done

# for d in $(ls | grep 00 | head -$((num_test+num_val)) | tail -$num_test); do 
# 	echo "$PWD/$d" >> test_set.txt
# done

