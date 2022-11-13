
# Set folder and file paths
weight_folder = '../YOLOv4_trained_weights'
train_images_folder = '../Yolo train data'

# Create object names file with class names
!echo -e 'license-plate' > data/obj.names 

# Create train.txt file with file paths for train images
import glob

images_list = glob.glob(train_images_folder+'/*.jpg')
with open('data/train.txt', 'w') as f:
    f.write('\n'.join(images_list))



# Create data file with relevant text files
!echo -e 'classes = 1\ntrain = data/train.txt\nvalid = data/test.txt\nnames = data/obj.names\nbackup = '+ weight_folder > data/obj.data

# Train darknet model for Yolov3
./darknet detector train data/obj.data cfg/yolov3-train.cfg darknet53.conv.74 -dont_show


# Train darknet model for Yolov4
./darknet detector train data/obj.data cfg/yolov4-train.cfg yolov4.conv.137 -dont_show