#! /bin/bash

source /home/ravindu/.virtualenvs/cv/bin/activate

rm -rf /media/ravindu/Games/V-Num/Final_Product/cropped_plates/*

gnome-terminal -- python3 -u "/media/ravindu/Games/V-Num/Final_Product/plate_detector_script.py" 
sleep 3
gnome-terminal -- python3 -u "/media/ravindu/Games/V-Num/Final_Product/listener.py" 