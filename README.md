# Handwritten_Digits_Detection
## Neural Network Model 

### Description
Handwritten digit detection involves recognizing and classifying handwritten digits from an image(28*28 pixels used in model).
A handwritten digit detection model is a machine learning algorithm that is trained on a dataset of handwritten digits to recognize and classify them accurately.

This model consists of several layers of artificial neural networks that use various mathematical functions to process the input image and predict the handwritten digit. The output layer of the model consists of a set of neurons, each corresponding to a particular digit class (0-9), which produces a probability distribution over the classes.

During training, the model is fed with a dataset of labeled images and adjusts its parameters to minimize the difference between its predictions and the actual labels. The process is typically done using stochastic gradient descent optimization algorithms.

After training, the model is used to classify new handwritten digits that are not present in the training data. The image is first preprocessed to normalize the pixel values and size, and then fed into the model for classification. The output of the model provides the predicted class label of the digit in the image.

### Installation and Running
**Language Used** : Python

**Libraries Used** : tensorflow, numpy, matplotlib, seaborn

**Data set** : mnist from keras

### Result Images
<p align="center">
  <img width="300" height="300" src="https://user-images.githubusercontent.com/83869332/229929262-8788290d-735a-4ed0-bd68-b6ce5a7e95d7.png">
</p>
<p align="center">
Demo Test Image
</p>
<p align="center">
  <img width="300" height="300" src="https://user-images.githubusercontent.com/83869332/229929903-44b0f050-ed73-422e-9488-7172528c1e09.png">
</p>

### Real World Uses
Handwritten digit detection models have many applications, including **optical character recognition**, **document processing**, and **digit recognition in postal**. services.

### Good Thoughts
Deep learning is like cooking - you need the right ingredients (data), the right recipe (algorithm), and the right tools (computational resources) to make something delicious (accurate predictions)." - **_Fei-Fei Li_**
