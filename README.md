# **V-Num**

## An enhanced direct vehicle number extraction system for moving high speed vehicles. 

### Application Scenario

- The module is developed in order to implement an entirely automated tolling system for expressways in Sri Lanka. 
- In order to achieve such a feat precise identification of vehicles are needed at entrance and exit. 
- One of the main methods to uniquely identify a vehicle is to use its number plate ID.
- The project is a sub module of the automated system and it focuses on correctly identifying the ID of the vehicle while it's moving at 60kmph+ speed.

![1_inbGrTzQZ9ZvhZTwIYMxzA](https://user-images.githubusercontent.com/80534358/191065587-50dc0cc0-aa51-47f4-a872-ce3fb337a6e6.png)

### Software algorithms

In order to achieve high success rates in uncontrolled environments efficient and reliable software algorithms have to be implemented. This project will consist of two such major software modules.
#### 1. Plate Localization
Object detection algorithm has to be implemented in order to identify the number plate area of the image. Recurrent Convolutional Neural Network algorithms can be used efficiently for this.
#### 2. OCR
The localized characters needs to be identified and this will be done using OCR ( Optical Charact Recognition) algorithms. For this purpose I have currently implemented a CRNN model which consists of convolutional layers followed by LSTM layers which combined can etxract information from the image. A CTC layer is used for prediction.

### Hardware Modules

High quality cameras are used to gather the video stream.
The sequent processing will be done using a computer with a CUDA enabled GTX 1650 graphics processing unit. 

The training of CNN models will be done using a computer with a CUDA enabled RTX 2080 graphics processing unit for time efficiency.

### Timeline

![V-Num timeline](https://user-images.githubusercontent.com/80534358/191070849-5564ed50-0b47-43e5-802b-36cc15e1a764.jpg)

### References

1. Kang, Dong. (2009). Dynamic programming-based method for extraction of license plate numbers of speeding vehicles on the highway. International Journal of Automotive Technology. 10. 205-210. 10.1007/s12239-009-0024-2. 

2. W. Wang, Q. Jiang, X. Zhou and W. Wan, "Car license plate detection based on MSER," 2011 International Conference on Consumer Electronics, Communications and Networks (CECNet), 2011, pp. 3973-3976, doi: 10.1109/CECNET.2011.5768335.

3. Islam, Md & Mondal, Chayan & Azam, Md & Islam, Abu. (2016). Text detection and recognition using enhanced MSER detection and a novel OCR technique. 15-20. 10.1109/ICIEV.2016.7760054. 

4. Jiuxiang Gu, Zhenhua Wang, Jason Kuen, Lianyang Ma, Amir Shahroudy, Bing Shuai, Ting Liu, Xingxing Wang, Gang Wang, Jianfei Cai, Tsuhan Chen,
Recent advances in convolutional neural networks,
Pattern Recognition,
