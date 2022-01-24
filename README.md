# object-detection-yolo
* Util.py and drawboxes.py should be in Utils folder which will be present in yad2k folder.
* The keras_yolo and keras_darknet should be in models folder which will be present in yad2k folder.
* We trained our model on yolov2 weights took from yolo official website.
* The test images should be in folder created Images.
* The coco and pascal classes and yolo.h5 model(yolov2 pretrained weights file) and yolo anchors.txt is saved in model_data folder which is present in directory where yad2k folder is present and the yad2k.py is present.

**summary**
The pretrained weights are used for this project( Transfer Learning )
You only look once (YOLO) is a state-of-the-art, real-time object detection system. On a Pascal Titan X it processes images at 30 FPS and has a mAP of 57.9% on COCO test-dev.
"You Only Look Once" (YOLO) is a popular algorithm because it achieves high accuracy while also being able to run in real-time. This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.


**Model details**
* Inputs and outputs
  * The input is a batch of images, and each image has the shape (m, 608, 608, 3)
  * The output is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers  as explained above. If you expand  into an 80-         dimensional vector, each bounding box is then represented by 85 numbers.
* Anchor Boxes
  * Anchor boxes are chosen by exploring the training data to choose reasonable height/width ratios that represent the different classes. For this assignment, 5 anchor boxes       were chosen for you (to cover the 80 classes), and stored in the file './model_data/yolo_anchors.txt'
  * The dimension for anchor boxes is the second to last dimension in the encoding: .
  * The YOLO architecture is: IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85).
* Encoding
Let's look in greater detail at what this encoding represents.

If the center/midpoint of an object falls into a grid cell, that grid cell is responsible for detecting that object.

Since we are using 5 anchor boxes, each of the 19 x19 cells thus encodes information about 5 boxes. Anchor boxes are defined only by their width and height.

For simplicity, we will flatten the last two last dimensions of the shape (19, 19, 5, 85) encoding. So the output of the Deep CNN is (19, 19, 425).

* Class score
 * Now, for each box (of each cell) we will compute the following element-wise product and extract a probability that the box contains a certain class.
  The class score is : the probability that there is an object  times the probability that the object is a certain class .


* Visualizing classes
 * Here's one way to visualize what YOLO is predicting on an image:

 * For each of the 19x19 grid cells, find the maximum of the probability scores (taking a max across the 80 classes, one maximum for each of the 5 anchor boxes).
   Color that grid cell according to what object that grid cell considers the most likely.
   Doing this results in this picture:

* Note that this visualization isn't a core part of the YOLO algorithm itself for making predictions; it's just a nice way of visualizing an intermediate result of the algorithm.

* Visualizing bounding boxes
Another way to visualize YOLO's output is to plot the bounding boxes that it outputs. Doing that results in a visualization like this:

* Non-Max suppression
In the figure above, we plotted only boxes for which the model had assigned a high probability, but this is still too many boxes. You'd like to reduce the algorithm's output to a much smaller number of detected objects.

* To do so, you'll use non-max suppression. Specifically, you'll carry out these steps:

* Get rid of boxes with a low score (meaning, the box is not very confident about detecting a class; either due to the low probability of any object, or low probability of this particular class).
* Select only one box when several boxes overlap with each other and detect the same object.
 * Filtering with a threshold on class scores
   You are going to first apply a filter by thresholding. You would like to get rid of any box for which the class "score" is less than a chosen threshold.

**The model gives you a total of 19x19x5x85 numbers, with each box described by 85 numbers. It is convenient to rearrange the (19,19,5,85) (or (19,19,425)) dimensional tensor into the following variables:**

box_confidence: tensor of shape  containing  (confidence probability that there's some object) for each of the 5 boxes predicted in each of the 19x19 cells.
boxes: tensor of shape  containing the midpoint and dimensions  for each of the 5 boxes in each cell.
box_class_probs: tensor of shape  containing the "class probabilities"  for each of the 80 classes for each of the 5 boxes per cell.

Predicting
![Screenshot (522)](https://user-images.githubusercontent.com/90260133/150759109-b0ebb2df-6741-4158-9824-e60d62dc9973.png)

