#!/bin/bash
# this script runs the program that colorizes images

for i in $(seq $1 $2) 
do
    echo $i
    iter=$((i * 1000))
    echo $iter
    /usr/bin/python2.7 /mnt/6TB-WD-Black/cs450/automatic-colorization/colorize.py -i $iter -m 100 -d /mnt/6TB-WD-Black/cs450/data/gargantuan/color_224x224
done
