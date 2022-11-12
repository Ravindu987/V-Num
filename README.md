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
The localized characters needs to be identified and this will be done using OCR ( Optical Character Recognition) algorithms. For this purpose I have currently implemented a CRNN model which consists of convolutional layers followed by LSTM RNN layers which combined can etxract information from the image. 
- CNN is used to extract features from the image
- RNN (LSTM) to predict characters
- Finally a CTC layer is used for the classification.

### Hardware Modules

High quality cameras are used to gather the video stream.
The sequent processing will be done using a computer with a CUDA enabled GTX 1650 graphics processing unit. 

The training of CNN models will be done using a computer with a CUDA enabled RTX 2080 graphics processing unit for time efficiency.

By enabling CUDA, I was able to reduce the training time by a factor of 6 on GTX 1650 and 12 on RTX 2080.

### Timeline

![Timline_OCT_11](https://user-images.githubusercontent.com/80534358/195003300-9e98222d-2b3f-4397-a92d-bfb7c6ac8d3e.jpg)

### References

1. Kang, Dong. (2009). Dynamic programming-based method for extraction of license plate numbers of speeding vehicles on the highway. International Journal of Automotive Technology. 10. 205-210. 10.1007/s12239-009-0024-2. 

2. Drobac, Senka; LindÃ©n, Krister (2020). Optical character recognition with neural networks and post-correction with finite state methods. International Journal on Document Analysis and Recognition (IJDAR), (), –. doi:10.1007/s10032-020-00359-9 

3. Graves, Alex & Fernández, Santiago & Gomez, Faustino & Schmidhuber, Jürgen. (2006). Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural 'networks. ICML 2006 - Proceedings of the 23rd International Conference on Machine Learning. 2006. 369-376. 10.1145/1143844.1143891. 

4. Wick, Christoph & Reul, Christian & Puppe, Frank. (2018). Calamari - A High-Performance Tensorflow-based Deep Learning Package for Optical Character Recognition

5. Jiuxiang Gu, Zhenhua Wang, Jason Kuen, Lianyang Ma, Amir Shahroudy, Bing Shuai, Ting Liu, Xingxing Wang, Gang Wang, Jianfei Cai, Tsuhan Chen,
Recent advances in convolutional neural networks,
Pattern Recognition,
