#!/bin/sh

module add openmind/singularity

SCRIPT="/om/user/belledon/git/hdf5_dataset/hdf5.py"
SOURCE=$1
DEST="/om/user/ilkery/DATASETS/"
CONT="/om/user/belledon/singularity_imported/chainer_v2.img"

echo "Creating database from $SOURCE in $DEST"

singularity exec -B /om:/om -B /om2:/om2 $CONT python3 $SCRIPT $SOURCE -o $DEST
