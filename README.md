# **V-Num**

## An enhanced direct vehicle number extraction system for moving high speed vehicles.

### Application Scenario

- The module is developed in order to implement an entirely automated tolling system for expressways in Sri Lanka.
- In order to achieve such a feat precise identification of vehicles are needed at entrance and exit.
- One of the main methods to uniquely identify a vehicle is to use its number plate ID.
- The project is a sub module of the automated system and it focuses on correctly identifying the ID of the vehicle while it's moving at 60kmph+ speed.

![1_inbGrTzQZ9ZvhZTwIYMxzA](https://user-images.githubusercontent.com/80534358/191065587-50dc0cc0-aa51-47f4-a872-ce3fb337a6e6.png)

### Software algorithms

In order to achieve high success rates in uncontrolled environments efficient and reliable software algorithms have to be implemented. This project will consist of three such major software modules.

#### 1. Plate Localization

- Since the cameras installed at toll gates cover a lot of area, the approach to apply character recognition to the entire video frame becomes inefficient and requires a tremendous amount of computational power to acheive a reliable success rate. Therefore as the first module of the pipeline was dedicated for localizing and cropping out the vehicle number plate from the frame.
- An object detection algorithm was used for this purpose. Both YOLO v3 and v4 algorithms were both tested for this purpose.
- YOLO v4 slowed a slightly better performance than v3.
- Darknet framework was used to train the model.
- Manual built OpenCV with GPU support was used to run predictions on video file.

- Application
  - The detection algorithm was run on the frames of the video input using the trained weights.
  - Since it is computationally expensive to run detection on each and every frame, tuning is done. More details are given below.
  - The detected plate was cropped out and saved to a directory.

#### 2. Character Localization

Due to the nature of the license plate it is inefficient to read the entire number at once. Therefore relevant characters were localized first.

- Upsampling
  - The cropped out plates were of low quality and therefoe image upsampling was carried out to enhance image quality.
  - Several methods including ESRGAN, ESPCN, LapSRN was used. The method which provided best outcome was the image super rsolution technique EDSR.
  - This is not computationally expensive.
- Localization
  - Achieved using the following image processing techniques.
  - Grayscaling
  - Adaptive Gaussian Thresholding with 25 pixel kernel
  - Noise reduction
  - Dilate to fill small gaps
  - Erode to isolate letters
  - Contour detection was done
  - Smaller and larger contours were filtered out. The aspect ratios were calculated based on dimensions of the plate
  - Using the bounding box of the contour, the character from the original image was cropped out.

#### 3. Character Recognition

- A custom CNN was trained to recognize the characters from the license plate.
- CNN model was trained on 20,000 images from the target dataset and was tested on 1000 images from the target distribution.
- Neual network consisted of 4 Convolution and Pooling layers with 2 dense layes and softmax activation at the end.
- SparseCategoricalCrossentropy was used as the loss function and adam optimizer was used.
- The weights were stored in HDF5 format.

### Hardware Modules and other platforms

The processing was done using a computer with a CUDA enabled GTX 1650 graphics processing unit and a 9th gen Intel core i7 processor.
The training of CNN models was be done using a computer with a CUDA enabled RTX 2080 graphics processing unit for time efficiency.

Linux Mint and Ubuntu were mainly used as the operating systems for the project.
A manual built GPU support Opencv build was used for image processing.
Cuda 11.8 and CuDNN were used to get GPU support for processing.
Tensorflow-GPU libary was as the framework for deep learning.

By enabling CUDA and using GPU for processing, I was able to increase the time efficiency for all the training, preprocessing, image processing and prediction modules.
As an example, CuDNN reduced the training time of the YOLo model by a factor of 6 on GTX 1650 and 12 on RTX 2080 compared to default CPU processing.

### Accuracy

Accuracy of the models for both Yolo v3 and v4 was tested using the IoUs for the ground truth and predict bounding boxes.

IoU values should be closer to 1 as much as possible.

![Yolov3_IOU](https://user-images.githubusercontent.com/80534358/201886496-2175e4fe-fab2-4933-ae0d-7ac0fcbce56a.png)

![Yolov4_IOU](https://user-images.githubusercontent.com/80534358/201886523-c51a5bf1-de88-482f-a573-9e78ee7c4b91.png)

### Performance Tuning

One of the main decisions in setting up this module is setting up how to run predictions on the video stream. Running a prediction is a computationally expensive task and has to be done minimally. On the other hand as this is aimed towards high speed vehicles, the predictions has to be run frequently as well.
I tested the fps output when

1. No prediction is done
2. Prediction is done for every video frame
3. Prediction is done for every 4 video frames
4. Prediction is done for every 10 video frames

Folliwng are the results for Yolov3 and v4.

![Yolov3](https://user-images.githubusercontent.com/80534358/201687814-8244204f-635b-4673-b2dc-dfb43e29e17d.png)

![Yolov4](https://user-images.githubusercontent.com/80534358/201687865-8870d744-1c49-4a1f-988f-72d56cc8847d.png)

### Pipeline Architecture

All three modules mentioned above must be implemented together in the final product.

This is done via executing two scripts simultaneously.

1. Detector script - Runs the plate detector algorithm on the video and save the detected plates to the local storage as jpgs.
2. Listener - The python API watchdog is used to listen for file additions in the specific directory. When the detector saves a new jpg in the directory the listener is triggered and event handler gets executed. The event handler feeds the new image to the character localization and character recognition algorithms.
   The characters read by the algorithms are currently written to a text file.

To avoid duplicates, the text values are checked before writing to the text file.

### References

1. ralhad Gavali, J. Saira Banu,
   Chapter 6 - Deep Convolutional Neural Network for Image Classification on CUDA Platform

2. J. Shashirangana, H. Padmasiri, D. Meedeniya and C. Perera, "Automated License Plate Recognition: A Survey on Methods and Techniques," in IEEE Access, vol. 9, pp. 11203-11225, 2021, doi: 10.1109/ACCESS.2020.3047929.

3. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788).

4. Lim, B., Son, S., Kim, H., Nah, S. and Mu Lee, K., 2017. Enhanced deep residual networks for single image super-resolution. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops (pp. 136-144)

5. Kang, Dong. (2009). Dynamic programming-based method for extraction of license plate numbers of speeding vehicles on the highway. International Journal of Automotive Technology. 10. 205-210. 10.1007/s12239-009-0024-2.

6. Lim, B., Son, S., Kim, H., Nah, S. and Mu Lee, K., 2017. Enhanced deep residual networks for single image super-resolution. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops (pp. 136-144)
