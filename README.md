# Object-Detection(Computer Vision)
The Goal of this project is to detect a banana in the group of images and to draw a red rectangle around the region that is detected as a banana.  In the following, I will describe the datasets used in this project and also procedure of improving the accuracy of our learning model. The procedure of this project includes following steps: 1-Three different datasets are created by opencv c++, namely simple rgb, grayscale images and gaussian blurred images 2-CNN model implemented by Keras in python and also the best patches and its neighbour are determined by python and a final bounding box coordinates will be stored in a file 3-The bounding box coordinates file is used by opencv C++ to draw bounding box

# Datasets and Preprocessing(C++)
Datset Address: https://d2l.ai/chapter_computer-vision/object-detection-dataset.html

There is a one base database in this project with 1000 training images and 100 test images with size 256x256 pixels that have different backgrounds and a bananas placed in specific coordinates,(xmin,ymin,xmax,ymax), of them. Some steps are done on this database to generate 4 new different databases to use them as a input of Convolutional Neural Network(CNN). There is a general procedure for creating datasets including: 1-dividing each training image into 32*32 patches with step of 16. 2-check the intersection of each patches with banana patch. 3- if there is a intersection, calculate the common area between two patches. 4- if the common area is bigger than 25% of banana patch area, then the label will be 1, smaller than 5% will be 0 and between 5% and 25% has not any label because if we assign 0 or 1 to them they can increase the fault of our model. After these four steps, 5852 32*32 patches with label=1 and 219149 32*32 patches with label=0 is generated. In first database, 2000 patches from 1’s and 2000 patches from 0’s is randomly selected and converted to csv file for using as input of CNN. Second dataset is same as one but number of patches with label=0 has been doubled, 2000 patches from 1’s and 4000 patches from 0’s. In third dataset, we have 2000 1’s and 4000 0’s but images are first converted to grayscale and then converted to csv file. In fourth dataset, we have 2000 1’s and 4000 0’s but images are first blured by gaussian blur with kernel size 7 and sigma 2 and then converted to csv file.

# Learning Phase(Python)
CNN model have been implemented by tensorflow and keras. Different strategies and architectures was tested such as Kfold and finally the best architecture was selected, according to its results and scores. After running CNN on training data the generated model will be saved and 20 random images are given to the model to see the results. The procedure includes following steps: 1- Executing CNN on training data 2- Giving 20 images as test data to the trained model 3- preprocess each image based on our model for example for the model that is trained by grayscale images the test images should be converted to grayscale and for the model that is trained by blurred images the test images should be blurred by gaussian blur 4- divide each test image to a grid of 32*32 patches with step 16 to get 225 patches and normalize each patches 5- Execute model prediction on each patch and store the generated score and the coordinates of patch if the score is higher than a specified threshold. The threshold can be different for each model. 6- The patch with the highest score is likely contained a banana, So we sort the score of patches. 7- Analyze neighbours of the patches with the highest score and each neighbour that has score higher than a threshold will be added to a list of final patches 8- Analyze the final patches and find the lowest xmin, lowest ymin, highest xmax and highest ymax between them 9- Now we have two points with coordinates of (xmin,ymin) and (xmax,ymax) and we can draw bounding box on each test image. - If the algorithm can not find any patch on a image with score higher than specified threshold then it will return ”no banana” and no bounding box will be drawn on image - There is two threshold, first one for selecting patches and second one for selecting neighbours
