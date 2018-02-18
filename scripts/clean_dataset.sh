# Clean data

dataset='canyon_random'

cd /esat/qayd/kkelchte/docker_home/pilot_data/$dataset


# Delete all runs with less than 20 images
for d in 00* ; do
	if [ $(ls $d/RGB | wc -l) -lt 30 ] ; then 
		rm -r $d; 
		echo "deleted $d"; 
	else
		# make sure the applied control of the model is set as control info and loaded in data
		echo "mv control / predicted action info"
		mv $d/control_info.txt $d/ba_info.txt
		mv $d/predicted_info.txt $d/control_info.txt

		# get rid of the last 15 frames due to the delay of bump detection up until saving images is finished
		echo "remove final images"
		for i in $(ls $d/RGB/ | sort -r | head -15) ; do rm $d/RGB/$i; done

		# create collision labels
		echo "create collision labels"
		for i in $(ls $d/RGB | cut -d '.' -f 1) ; do echo "$i 0" >> $d/collision_info.txt; done
		for l in $(tail -10 $d/collision_info.txt | cut -d ' ' -f 1); do sed -i "s/$l 0/$l 1/" $d/collision_info.txt; done
	fi
done

# create train_set.txt, test_set.txt, val_set.txt
num_tot=$(ls | grep 00 | wc -l)
num_train=$((num_tot-5))
num_val=3
num_test=2
for d in $(ls | grep 00 | sort -r | head -$num_train); do
	echo "$PWD/$d" >> train_set.txt
done

for d in $(ls | grep 00 | head -$num_val); do 
	echo "$PWD/$d" >> val_set.txt
done

for d in $(ls | grep 00 | head -$((num_test+num_val)) | tail -$num_test); do 
	echo "$PWD/$d" >> test_set.txt
done

