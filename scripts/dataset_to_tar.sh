#!/bin/bash
dataset="$1"
dataroot="/esat/opal/kkelchte/docker_home/pilot_data"
tmpdataset=${dataset}_tmp
mypwd="$PWD"
if [ -z $dataset ] ; then
  echo "Error: no dataset found: $dataset"
  exit 1
fi
echo "tarring dataset: $dataroot/$dataset"
if [ -d $dataroot/$tmpdataset ] ; then
  rm -r $dataroot/$tmpdataset
fi
mkdir $dataroot/$tmpdataset
for t in train val test ; do
  while read line ; do 
    cp -r $line $dataroot/$tmpdataset
    echo "/tmp/$dataset/$(basename $line)" >> $dataroot/$tmpdataset/${t}_set.txt
  done < $dataroot/$dataset/${t}_set.txt
  # cp $dataroot/$dataset/${t}_set.txt $dataroot/$tmpdataset
  # sed -i 's/esat\/opal\/kkelchte\/docker_home\/pilot_data/tmp/' $dataroot/$tmpdataset/${t}_set.txt
done
cd $dataroot
mv $dataset ${dataset}_old
mv $tmpdataset $dataset
tar cf ${dataset}.tar $dataset
cp ${dataset}.tar /gluster/visics/kkelchte/pilot_data
rm -r $dataset
mv ${dataset}_old $dataset
cd $mypwd
