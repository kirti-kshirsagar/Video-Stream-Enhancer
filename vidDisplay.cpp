/* Kirti Kshirsagar | 26-01-2024 
 Program to read and display live video from the device camera in a window. 
 This file accesses all the functions from the filter.cpp and displays the effects on the live video.
 More detailed info written in the README file*/

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <cstdio> // a bunch of standard C/C++ functions like printf, scanf
#include <cstring> // C/C++ functions for working with strings
#include <cmath>
// #include <sys/time.h> // for gettimeofday()
#include "filter.h" // Include the filter header file
#include "faceDetect.h" // Include the face detection header file

// returns a double which gives time in seconds
// double getTime() {
//   struct timeval cur;

//   gettimeofday( &cur, NULL );
//   return( cur.tv_sec + cur.tv_usec / 1000000.0 );
// }

int main(int argc, char *argv[]) {
    cv::VideoCapture *capdev;

    // opens the video device
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return (-1);
    }

    // Creating VideoWriter object
    cv::VideoWriter videoWriter;
    int frame_width = static_cast<int>(capdev->get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

    // properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    // identifies a window
    cv::namedWindow("Video", 1); 
    cv::Mat frame, grey; // Matrix to store the frame and greyscale image
    std::vector<cv::Rect> faces; // Vector to store faces detected
    cv::Rect last(0, 0, 0, 0); // Variable to store the last face detected

    // Load the cascade
    cv::CascadeClassifier faceCascade; 
    std :: string cascadePath = FACE_CASCADE_FILE;
    faceCascade.load(FACE_CASCADE_FILE);

    //char lastKey = '\0'; // Variable to store the last key pressed
    bool convertToGrayscale = false; // Variable to track grayscale conversion state
    bool customGreyscale = false; // Variable to track custom greyscale state
    bool sepia = false; // Variable to track sepia state
    bool blur = false; // Variable to track blur state
    bool sobelX = false; // Variable to track sobelX state
    bool sobelY = false; // Variable to track sobelY state
    bool mag = false; // Variable to track magnitude state
    bool blurQ = false; // Variable to track blurQuantize state
    bool face = false; // Variable to track face detection state
    bool neg = false; // Variable to track negative state
    bool box = false; // Variable to track box filter state
    bool faceColor = false; // Variable to track face detection color state
    bool sharpen = false; // Variable to track sharpen state
    const int Ntimes = 10; // Number of times to run the loop for timing purposes

    // Loop to run the program
    for (;;) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        // Check if the frame is empty
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }
        
        char key = cv::waitKey(10);

        // Convert to greyscale if 'g' is pressed
        if (key == 'g') {
            convertToGrayscale = true; 
            customGreyscale = false;  
            sepia = false;      
            blur = false;  
            sobelX = false; 
            sobelY = false; 
            mag = false; 
            blurQ = false; 
            face = false; 
            neg = false; 
            box = false; 
            faceColor = false; 
            sharpen = false; 
        } 
        else if (key =='h') {
            // Convert to custom greyscale if 'h' is pressed
            customGreyscale = true;
            convertToGrayscale = false;
            sepia = false;
            blur = false;
            sobelX = false;
            sobelY = false;
            mag = false;
            blurQ = false;
            face = false;
            neg = false;
            box = false;
            faceColor = false;
            sharpen = false;
        } 
        else if (key == 'e') {
            // Convert to sepia if 'e' is pressed
            sepia = true;
            convertToGrayscale = false;
            customGreyscale = false;
            blur = false;
            sobelX = false;
            sobelY = false;
            mag = false;
            blurQ = false;
            face = false;
            neg = false;
            box = false;
            faceColor = false;
            sharpen = false;
        }
        else if (key == 'c') {
            //Convert to original color if 'c' is pressed
            convertToGrayscale = false;
            customGreyscale = false;
            sepia = false;
            blur = false;
            sobelX = false;
            sobelY = false;
            mag = false;
            blurQ = false;
            face = false;
            neg = false;
            box = false;
            faceColor = false;
            sharpen = false;
        }
        else if (key == 'b') {
            // Blur the image if 'b' is pressed
            blur = true;
            convertToGrayscale = false;
            customGreyscale = false;
            sepia = false;
            sobelX = false;
            sobelY = false;
            mag = false;
            blurQ = false;
            face = false;
            neg = false;
            box = false;
            faceColor = false;
            sharpen = false;
        }
        else if (key == 'x') {
            // Apply sobelX if 'x' is pressed
            sobelX = true;
            convertToGrayscale = false;
            customGreyscale = false;
            sepia = false;
            blur = false;
            sobelY = false;
            mag = false;
            blurQ = false;
            face = false;
            neg = false;
            box = false;
            faceColor = false;
            sharpen = false;
        }
        else if (key == 'y') {
            // Apply sobelY if 'y' is pressed
            sobelY = true;
            convertToGrayscale = false;
            customGreyscale = false;
            sepia = false;
            blur = false;
            sobelX = false;
            mag = false;
            blurQ = false;
            face = false;
            neg = false;
            box = false;
            faceColor = false;
            sharpen = false;
        }
        else if(key == 'm') {
            // Apply magnitude if 'm' is pressed
            mag = true;
            convertToGrayscale = false;
            customGreyscale = false;
            sepia = false;
            blur = false;
            sobelX = false;
            sobelY = false;
            blurQ = false;
            face = false;
            neg = false;
            box = false;  
            faceColor = false; 
            sharpen = false;
        }
        else if( key == 'l') {
            // Apply blurQuantize if 'l' is pressed
            blurQ = true;
            convertToGrayscale = false;
            customGreyscale = false;
            sepia = false;
            blur = false;
            sobelX = false;
            sobelY = false;
            mag = false;
            face = false;
            neg = false;
            box = false;   
            faceColor = false; 
            sharpen = false;
        }
        else if(key == 'f') {
            // Apply face detection if 'f' is pressed
            face = true;
            convertToGrayscale = false;
            customGreyscale = false;
            sepia = false;
            blur = false;
            sobelX = false;
            sobelY = false;
            mag = false;
            blurQ = false;
            neg = false;
            box = false;
            faceColor = false;
            sharpen = false;
        }
        else if(key == 'n') {
            // Apply negative if 'n' is pressed
            neg = true;
            convertToGrayscale = false;
            customGreyscale = false;
            sepia = false;
            blur = false;
            sobelX = false;
            sobelY = false;
            mag = false;
            blurQ = false;
            face = false;
            box = false;
            faceColor = false;
            sharpen = false;
        }
        else if(key == 'a'){
            // Apply box filter if 'a' is pressed
            box = true;
            convertToGrayscale = false;
            customGreyscale = false;
            sepia = false;
            blur = false;
            sobelX = false;
            sobelY = false;
            mag = false;
            blurQ = false;
            face = false;
            neg = false;
            faceColor = false;
            sharpen = false;
        } 
        else if(key == 'k'){
            // Apply face detection color and remaining greyscale, if 'k' is pressed
            faceColor = true;
            convertToGrayscale = false;
            customGreyscale = false;
            sepia = false;
            blur = false;
            sobelX = false;
            sobelY = false;
            mag = false;
            blurQ = false;
            face = false;
            neg = false;
            box = false;
            sharpen = false;
        }
        else if(key =='j') {
            // Apply sharpening if 'j' is pressed
            sharpen = true;
            convertToGrayscale = false;
            customGreyscale = false;
            sepia = false;
            blur = false;
            sobelX = false;
            sobelY = false;
            mag = false;
            blurQ = false;
            face = false;
            neg = false;
            box = false;
            faceColor = false;
        }
        
        // Greyscale conversion function is called from filter.cpp
        if(convertToGrayscale) {
            cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
            frame = grey;
        } 
        // Custom greyscale conversion 
        else if(customGreyscale) {
            // Calling the custom greyscale function from filter.cpp
            cv::Mat customGreyscaleFrame;
            greyscale(frame, customGreyscaleFrame);
            frame = customGreyscaleFrame;
        }
        else if(sepia) {
            // Calling the sepia function from filter.cpp
            cv::Mat sepiaFrame;
            sepiaTone(frame, sepiaFrame);
            frame = sepiaFrame;
        }
        else if(blur) {
            // Calling the blur function from filter.cpp
            cv::Mat blurFrame;
            blur5x5_2(frame, blurFrame);
            frame = blurFrame;
        }
        else if(sobelX) {
            // Calling the sobelX function from filter.cpp
            cv::Mat sobelXFrame,cvtImg;
            sobelX3x3(frame, sobelXFrame);
            // Converting signed short image to unsigned char for display
            cv::convertScaleAbs(sobelXFrame, cvtImg);
            frame = cvtImg;
        }
        else if(sobelY) {
            // Calling the sobelX function from filter.cpp
            cv::Mat sobelYFrame,cvtImgY;
            sobelY3x3(frame, sobelYFrame);
            // Converting signed short image to unsigned char for display
            cv::convertScaleAbs(sobelYFrame, cvtImgY);
            frame = cvtImgY;
        }
        else if(mag) {
            // Calling the magnitude function from filter.cpp
            cv::Mat magImage,sobelYFrame, sobelXFrame;
            sobelX3x3(frame, sobelXFrame);
            sobelY3x3(frame, sobelYFrame);
            magnitude(sobelXFrame, sobelYFrame, magImage);
            frame = magImage;
        }
        else if(blurQ) {
            // Calling the blurQuantize function from filter.cpp
            cv::Mat blurQFrame;
            blurQuantize(frame, blurQFrame, 10);
            frame = blurQFrame;
        }
        else if(face) {
            // Calling the face detection function from faceDetect.cpp
            // convert the image to greyscale
            cv::cvtColor( frame, grey, cv::COLOR_BGR2GRAY, 0);
            // detect faces
            detectFaces( grey, faces );
            // draw boxes around the faces
            drawBoxes( frame, faces );

            // add a little smoothing by averaging the last two detections
            if( faces.size() > 0 ) {
            last.x = (faces[0].x + last.x)/2;
            last.y = (faces[0].y + last.y)/2;
            last.width = (faces[0].width + last.width)/2;
            last.height = (faces[0].height + last.height)/2;
            }
        }
        else if(neg){
            // Calling the createNegative function from filter.cpp
            cv::Mat negFrame;
            createNegative(frame, negFrame);
            frame = negFrame;
        }
        else if(box){
            // Calling the applyBoxFilter function from filter.cpp
            cv::Mat boxFrame;
            applyBoxFilter(frame, boxFrame, 5);
            frame = boxFrame;
        }
        else if(faceColor){
            // Calling the makeFaceColorful function from filter.cpp
            cv::Mat faceColorFrame;
            colorfulFace(frame, faceColorFrame,faceCascade);
            frame = faceColorFrame;
        }
        else if(sharpen){
            // Calling the sharpenImage function from filter.cpp
            cv::Mat sharpenFrame;
            sharpenImage(frame, sharpenFrame);
            frame = sharpenFrame;
        }
        // Displays the frame in the window
        cv::imshow("Video", frame);
        
        // Exit the program if 'q' is pressed
        if (key == 'q') {
            break;
        } 
        else if (key == 's') {
            // Save a screenshot if 's' is pressed 
            cv::imwrite("D:/PRCV/k.jpg", frame);
            std::cout << "Screenshot saved!" << std::endl;
        } 
        // Extension 2: Lets the user save a video recording of the effects applied to the live video
        else if (key == 'r') {
            // Start recording
            videoWriter.open("Recording.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25, cv::Size(frame_width, frame_height));
        }
        else if (key == 'p') {
            // Stop recording
            if (videoWriter.isOpened()) {
                videoWriter.release();
                std::cout << "Recording stopped." << std::endl;
            }
        }
        // Record processed frame
        if (videoWriter.isOpened()) {
            videoWriter.write(frame);
        }
    }
    // release the video capture resources
    delete capdev;
    if (videoWriter.isOpened()) {
        videoWriter.release();
    }
    return (0); 
}



