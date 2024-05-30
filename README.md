# README
Name - **KIRTI KSHIRSAGAR** 
- Using Windows 10, VScode, CMake,OpenCV 4.9(recent one) and C++.
- Not using any Time Travel Days
- Extension is video record capability so: [Short Video](https://drive.google.com/file/d/1HuC4J5c5XWaNXUuzbflr6IzGDnlZ7wMe/view?usp=sharing) 

## This is a Readme file for vidDisplay.cpp and imgDisplay.cpp file and this includes all the details that are necessary to run the program.

### Task-1
- This task needs to create a imgDisplay.cpp file. This has been included in a different folder known as Project-1-Task1. 
    * Keypress **q** leads to closing of the image window.
    * Keypress **s** saves a screenshot in the Project folder.

## All the tasks from now on have been created in the new folder known as Project-1
### Task-2
- This task opens a video channel, create a window, and then loop, capturing a new frame and displaying it each time through the loop.
    * Keypress **q** leads to quiting the program.
    * Keypress **s** leads to saving the frame to the folder.

### Task-3
- This task displays a greyscale version of the video stream instead of color.
    * Keypress **g** leads to grayscaling the video frames from the time it is pressed.

### Task-4
- The greyscale function implements the greyscale transformation of the live video.
    * Keypress **h** leads to this Custom Greyscale version of the frame.

### Task-5
- The sepiaTone function applies sepia tone filter and makes an image look like it was taken by an antique camera.
    * Keypress **e** leads to applying this function on live camera feed from the moment **e** is pressed.

### Task-6
- **Task A :** In this task, 5x5 blur filter has been implemented. This function uses at<> method
- **Task B :** In this task, again blur filter has been implemented, but this implementation uses two separable 1x5 filters. Also, this function uses the ptr<> method. 
    * Keypress **b** leads to application of Task-B blur filter on the live camera feed.

### Task-7
- In this task, two functions are created, sobelX3x3 and sobelY3x3. Each function implements a 3x3 Sobel filter, either horizontal (X) or vertical (Y). The X filter is positive right and the Y filter is positive up.
    * Keypress **x** leads to calling and application of sobel-x function.
    * Keypress **y** leads to calling and application of sobel-y function.

### Task-8
- This function generates a gradient magnitude image using Euclidean distance for magnitude.
    * keypress **m** calls the magnitude function, which in turn calls the sobel function as it needs 3-channel signed short images as input from x and y sobel.

### Task-9
- The blurQuantize function takes in a color image, blurs the image, and then quantizes the image into a fixed number of levels as specified by a parameter.
    * keypress **l** displys the filter applied frame by calling the blur function and applying more calculations on it.

### Task-10
- This task links the faceDetect.cpp file to our video stream program i.e. vidDisplay.cpp and turns on the face detection if the user hits the **f** key.

### Task-11
- For a single-step pixel-wise modification, I have implemented createNegative function. This function creates a negative of the input image.
    * keypress **n** displays the negative of the frame.
- For effect that needs to make use of an area effect, applyBoxFilter function is implemented, which applies box filter to the live camera feed.
    * keypress **a** applies this area effect and gives the filtered output as the display.
- For effect that needs to be build on the face detector, I implemented the 'Make the face colorful, while the rest of the image is greyscale' effect.
    * keypress **k** gives the desired output by detecting the faces first and then applying the filter.



# Extension:
For the extensions, I have applied two tasks.
- Task 1 : I have created a sharpenImage function that applies Laplacian kernel in order to generate sharpened Image from the live feed.
    * Keypress **j** executed the image sharpening filter.

- Task 2 : In the vidDisplay.cpp file, I've created a videoWriter object, thereby giving the user chance to save videos with special effects.
    * Keypress **r** starts the recording of the feed displayed in the window including the effects being tested out.
    * Keypress **p** stops this recording.


