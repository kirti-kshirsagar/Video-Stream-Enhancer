/* Kirti Kshirsagar | 21-01-2024 
Program to read an image and display it in a window. Press 'q' to quit and 's' to save a screenshot. */
#include <opencv2/opencv.hpp>
#include<stdio.h>
int main() {
    // Read an image file
    cv::Mat image = cv::imread("D:/PRCV/test1.jpg");

    // Check if the image was successfully loaded
    if (image.empty()) {
        std::cerr << "Can't read the image file !" << std::endl;
        return -1;
    }
    // Create a window for display
    cv::namedWindow("Display Image", cv::WINDOW_NORMAL);
    // Resize the window 
    cv::resizeWindow("Display Image", 600, 900);

    // Display the image in a window
    cv::imshow("Display Image", image);

    // Enter a loop to check for keypress
    while (true) {
        // Wait for a key event (0 means wait indefinitely)
        int key = cv::waitKey(0);

        // Check if the key is 'q' 
        if (key == 'q') {
            break;  // Quit the program if 'q' is pressed
        } else if (key == 's') {
            // Save a screenshot if 's' is pressed (replace "screenshot.jpg" with the desired file name)
            cv::imwrite("D:/PRCV/Project-1-Task1/screenshot.jpg", image);
            std::cout << "Screenshot saved!" << std::endl;
        }
    }

    return 0;
}

