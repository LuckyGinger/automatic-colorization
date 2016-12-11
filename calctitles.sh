#!/bin/bash
CLASSIFY_IMAGE="/mnt/6TB-WD-Black/cs450/automatic-colorization/./classify_image.py"

if [ -n "$1" ]
then
	echo "Running folder $1"
else
	echo "USAGE: calctitles [folder_of_tars]"
	exit
fi

FOLDER=$1
TAR_FILES=$(ls $FOLDER)

# Current directory is always this folder.
cd $FOLDER
echo > OUT.txt

# For each tar file in the directory.
for tf in $TAR_FILES;
do
	mkdir -p /tmp/$tf
	tar -xf $tf -C /tmp/$tf
	FILES=$(ls /tmp/$tf)
	TMP_TXT=/tmp/$tf/tmp.txt
	echo "Working on file $tf."

	# For each file in the tar file.
	for f in $FILES;
	do
		$CLASSIFY_IMAGE --image_file /tmp/$tf/$f 2> /dev/null > $TMP_TXT;
		echo -en "$f," >> OUT.txt
		ls $TMP_TXT
		cat $TMP_TXT | head -n 1 | sed 's/[,\(].*$//' >> OUT.txt
	done

	# Remove the extracted images.
	rm -r /tmp/$tf
done
