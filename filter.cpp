/*
  Kirti Kshirsagar | 26-01-2024

  Functions for adding filters and effects to the frames of a video stream

*/
#include <opencv2/opencv.hpp>
#include "filter.h"

/*
 Function to convert a color image to a custom greyscale
 Arguments:
 cv::Mat &src - Input color image
 cv::Mat &dst - Output greyscale image
*/
int greyscale(cv::Mat &src, cv::Mat &dst) {
    // destination image should be of the same size and type as the source image, hence copying the source image
    src.copyTo(dst);

    // Iterate through each pixel
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            // Get pixel values for each channel
            uchar blue = src.at<cv::Vec3b>(i, j)[0];
            uchar green = src.at<cv::Vec3b>(i, j)[1];
            uchar red = src.at<cv::Vec3b>(i, j)[2];

            // Custom greyscale transformation (e.g., subtract red channel from 255)
            uchar greyValue = 255 - red;

            // Assign the same greyscale value to all three channels
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(greyValue, greyValue, greyValue);
        }
    }
    return 0;
}

/*
 Function to convert a color image to sepia tone
 Arguments:
 cv::Mat& src - Input image
 cv::Mat& dst - Output image with sepia tone effect
 */
void sepiaTone(cv::Mat& src, cv::Mat& dst) {
    CV_Assert(src.depth() == CV_8U);
    // Create a copy of the source image
    dst = src.clone();
    // Iterate through each pixel
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            // Get the pixel value
            cv::Vec3b& pixel = dst.at<cv::Vec3b>(i, j);
            // Apply the sepia tone formula
            int newRed = 0.393 * pixel[2] + 0.769 * pixel[1] + 0.189 * pixel[0];
            int newGreen = 0.349 * pixel[2] + 0.686 * pixel[1] + 0.168 * pixel[0];
            int newBlue = 0.272 * pixel[2] + 0.534 * pixel[1] + 0.131 * pixel[0];
            // Adjust pixel channel values while ensuring they stay within the valid uchar range [0, 255]
            pixel[0] = cv::saturate_cast<uchar>(std::min(newBlue, 255));
            pixel[1] = cv::saturate_cast<uchar>(std::min(newGreen, 255));
            pixel[2] = cv::saturate_cast<uchar>(std::min(newRed, 255));
        }
    }
}


/*Implement a 5x5 blur filter using a 
Gaussian kernel:
[1 2 4 2 1; 
2 4 8 4 2; 
4 8 16 8 4; 
2 4 8 4 2; 
1 2 4 2 1]
Arguments:
cv::Mat &src - Input image
cv::Mat &dst - Output image with 5x5 blur effect
*/
int blur5x5_1(cv::Mat &src, cv::Mat &dst) {
    src.copyTo(dst);
    // Guassian kernel
    int kernel[5][5] = {{1, 2, 4, 2, 1}, {2, 4, 8, 4, 2}, {4, 8, 16, 8, 4}, {2, 4, 8, 4, 2}, {1, 2, 4, 2, 1}};
    int sum = 100; // Normalization factor
    // Iterate over each pixel in the inner region
    for (int i = 2; i < src.rows - 2; i++) {
        for (int j = 2; j < src.cols - 2; j++) {
            // Separate channels
            int sb = 0, sg = 0, sr = 0;
            // Apply the kernel
            for (int ki = -2; ki <= 2; ki++) {
                for (int kj = -2; kj <= 2; kj++) {
                    cv::Vec3b pixel = src.at<cv::Vec3b>(i + ki, j + kj);
                    sb += pixel[0] * kernel[ki + 2][kj + 2];
                    sg += pixel[1] * kernel[ki + 2][kj + 2];
                    sr += pixel[2] * kernel[ki + 2][kj + 2];
                }
            }
            // Normalize and assign to the destination pixel
            dst.at<cv::Vec3b>(i, j)[0] = sb / sum;
            dst.at<cv::Vec3b>(i, j)[1] = sg / sum;
            dst.at<cv::Vec3b>(i, j)[2] = sr / sum;
        }
    }
    return 0;
}

/*
 Implement a 5x5 blur filter using two 1D kernels (seperable filters)
 Arguments:
 cv::Mat &src - Input image
 cv::Mat &dst - Output image with 5x5 blur effect, color image (type vec3b
*/
int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;
    }
    // Ensure the destination matrix has the same size and type as the source
    dst = cv::Mat(src.size(), src.type());
    // 1D kernel
    int kernel[5] = {1, 2, 4, 2, 1};
    int norm = 10; // Sum of 1D kernel values
    // Horizontal pass
    // Iterate over each pixel in the inner region
    for (int i = 0; i < src.rows; i++) {
        for (int j = 2; j < src.cols - 2; j++) {
            // Separate channels
            int blueSum = 0, greenSum = 0, redSum = 0;
            // Apply the kernel
            for (int kj = -2; kj <= 2; kj++) {
                int colIndex = j + kj;
                cv::Vec3b pixel = src.ptr<cv::Vec3b>(i)[colIndex];
                // Accumulate the values from the neighborhood
                blueSum += pixel[0] * kernel[kj + 2];
                greenSum += pixel[1] * kernel[kj + 2];
                redSum += pixel[2] * kernel[kj + 2];
            }
            // Normalize and assign to the destination pixel
            dst.ptr<cv::Vec3b>(i)[j][0] = blueSum / norm;
            dst.ptr<cv::Vec3b>(i)[j][1] = greenSum / norm;
            dst.ptr<cv::Vec3b>(i)[j][2] = redSum / norm;
        }
    }

    // Vertical pass
    // Iterate over each pixel in the inner region
    for (int j = 0; j < src.cols; j++) {
        for (int i = 2; i < src.rows - 2; i++) {
            // Separate channels
            int blueSum = 0, greenSum = 0, redSum = 0;
            // Apply the kernel
            for (int ki = -2; ki <= 2; ki++) {
                int rowIndex = i + ki;
                // Accumulate the values from the neighborhood
                cv::Vec3b pixel = src.ptr<cv::Vec3b>(rowIndex)[j];
                blueSum += pixel[0] * kernel[ki + 2];
                greenSum += pixel[1] * kernel[ki + 2];
                redSum += pixel[2] * kernel[ki + 2];
            }
            // Normalize and assign to the destination pixel
            dst.ptr<cv::Vec3b>(i)[j][0] = blueSum / norm;
            dst.ptr<cv::Vec3b>(i)[j][1] = greenSum / norm;
            dst.ptr<cv::Vec3b>(i)[j][2] = redSum / norm;
        }
    }
    return 0;
}



/*
Function to apply 3x3 Sobel filter for horizontal edges (X direction)
Arguments:
cv::Mat &src - Input image with 3 channels
cv::Mat &dst - Output image with 3x3 Sobel filter applied, needs to be of type 16SC3 (signed short)
*/
int sobelX3x3( cv::Mat &src, cv::Mat &dst ){
    // Destination image needs to be of type 16SC3 (signed short)
    dst.create(src.size(), CV_16SC3);
    // Iterate through each pixel
    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            // Sobel filter for horizontal edges
            cv::Vec3s gradient;
            // Compute the gradient for each channel
            gradient[0] = src.at<cv::Vec3b>(i, j + 1)[0] - src.at<cv::Vec3b>(i, j - 1)[0];
            gradient[1] = src.at<cv::Vec3b>(i, j + 1)[1] - src.at<cv::Vec3b>(i, j - 1)[1];
            gradient[2] = src.at<cv::Vec3b>(i, j + 1)[2] - src.at<cv::Vec3b>(i, j - 1)[2];
            // Assign the gradient to the destination pixel
            dst.at<cv::Vec3s>(i, j) = gradient;
        }
    }
    return 0; // Success
}

/*
Function to apply 3x3 Sobel filter for horizontal edges (Y direction)
Arguments:
cv::Mat &src - Input image with 3 channels
cv::Mat &dst - Output image with 3x3 Sobel filter applied, needs to be of type 16SC3 (signed short)
*/
int sobelY3x3( cv::Mat &src, cv::Mat &dst ){
    // Destination image needs to be of type 16SC3 (signed short)
    dst.create(src.size(), CV_16SC3);
    // Iterate through each pixel
    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            // Sobel filter for vertical edges
            cv::Vec3s gradient;
            // Compute the gradient for each channel
            gradient[0] = src.at<cv::Vec3b>(i + 1, j)[0] - src.at<cv::Vec3b>(i - 1, j)[0];
            gradient[1] = src.at<cv::Vec3b>(i + 1, j)[1] - src.at<cv::Vec3b>(i - 1, j)[1];
            gradient[2] = src.at<cv::Vec3b>(i + 1, j)[2] - src.at<cv::Vec3b>(i - 1, j)[2];
            // Assign the gradient to the destination pixel
            dst.at<cv::Vec3s>(i, j) = gradient;
        }
    }
    return 0; // Success
}


/*
Arguments:
sx and sy need to be 3-channel signed short images
cv::Mat &sx - Input image with 3x3 Sobel filter applied in X direction
cv::Mat &sy - Input image with 3x3 Sobel filter applied in Y direction
cv::Mat &dst - Output image with magnitude of the gradient of format uchar color image
*/
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
    // Ensure input images have the same size and type
    CV_Assert(sx.size() == sy.size() && sx.type() == CV_16SC3 && sy.type() == CV_16SC3);
    dst.create(sx.size(), CV_8UC3); // Output as uchar color image
    // Iterate through each pixel
    for (int i = 0; i < dst.rows; i++) {
        for (int j = 0; j < dst.cols; j++) {
            // Compute the Euclidean distance for magnitude: I = sqrt( sx*sx + sy*sy )
            float magnitude = std::sqrt(static_cast<float>(
                sx.at<cv::Vec3s>(i, j)[0] * sx.at<cv::Vec3s>(i, j)[0] +
                sy.at<cv::Vec3s>(i, j)[0] * sy.at<cv::Vec3s>(i, j)[0] +
                sx.at<cv::Vec3s>(i, j)[1] * sx.at<cv::Vec3s>(i, j)[1] +
                sy.at<cv::Vec3s>(i, j)[1] * sy.at<cv::Vec3s>(i, j)[1] +
                sx.at<cv::Vec3s>(i, j)[2] * sx.at<cv::Vec3s>(i, j)[2] +
                sy.at<cv::Vec3s>(i, j)[2] * sy.at<cv::Vec3s>(i, j)[2]));
            // Normalize the magnitude to fit in the uchar range [0, 255]
            uchar mg = static_cast<uchar>(magnitude);
            // Set the color channels of the output image
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(mg, mg, mg);
        }
    }
    return 0; // Success
}


/*
 This function blurs an image and quantizes it to a specified number of levels
 Arguments:
 cv::Mat &src - Input image
 cv::Mat &dst - Output quantized image
 int levels - Number of quantization levels
*/
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
    // Check if the source image is empty
    if (src.empty()) {
        return -1; // Error: Empty source image
    }
    // Blur the image
    blur5x5_2(src, dst);
    // Quantize the image
    double bucketSize = 255.0 / levels;
    // Iterate through each pixel 
    for (int i = 0; i < dst.rows; i++) {
        for (int j = 0; j < dst.cols; j++) {
            // Iterate through each channel
            for (int c = 0; c < dst.channels(); c++) {
                // Quantize each color channel
                double xt = dst.at<cv::Vec3b>(i, j)[c] / bucketSize;
                double xf = std::round(xt) * bucketSize;
                dst.at<cv::Vec3b>(i, j)[c] = static_cast<uchar>(xf);
            }
        }
    }
    return 0; // Success
}

/*
TASK-11 : Single-step pixel-wise modification
This function creates a negative of the input image
Arguments:
const cv::Mat& src - 3 channel input image
cv::Mat& dst - Output image with negative effect
*/
void createNegative(const cv::Mat& src, cv::Mat& dst) {
    // Ensure the destination matrix has the same size and type as the source
    dst.create(src.size(), src.type());
    // Iterate through each pixel
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            // Get the pixel value
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            // Invert each channel value and assign to the destination pixel
            pixel[0] = 255 - pixel[0]; // Blue
            pixel[1] = 255 - pixel[1]; // Green
            pixel[2] = 255 - pixel[2]; // Red
            // Set the inverted pixel value in the destination matrix
            dst.at<cv::Vec3b>(i, j) = pixel;
        }
    }
}


/*
TASK-11 : Effect makes use of an area effect
This function applies a box filter to the input image
Arguments:
cv::Mat& src - Input image
cv::Mat& dst - Output image with box filter effect
int kernelSize - Size of the kernel (must be odd)
*/
void applyBoxFilter(cv::Mat& src, cv::Mat& dst, int kernelSize) {
    dst.create(src.size(), src.type());
    // Ensure the kernel size is odd
    if (kernelSize % 2 == 0) {
        std::cerr << "Kernel size must be odd." << std::endl;
        return;
    }
    // Calculate the border size based on the kernel size
    int borderSize = kernelSize / 2;
    // Iterate through each pixel in the source image
    for (int x = borderSize; x < src.rows - borderSize; x++) {
        for (int y = borderSize; y < src.cols - borderSize; y++) {
            // Initialize the sum to 0 for each channel
            cv::Vec3f sum(0, 0, 0);
            // Iterate over the kernel
            for (int kx = -borderSize; kx <= borderSize; kx++) {
                for (int ky = -borderSize; ky <= borderSize; ky++) {
                    // Accumulate the values from the neighborhood
                    sum += src.at<cv::Vec3b>(x + kx, y + ky);
                }
            }
            // Averages the values and assign to the destination pixel
            dst.at<cv::Vec3b>(x, y) = sum / (kernelSize * kernelSize);
        }
    }
}

/*
TASK-11 : Effect build on the face detector.
This function applies a color filter to the faces in the input image and the backgraound is converted to grayscale
Arguments:
cv::Mat &src - Input image
cv::Mat &dst - Output image with color filter effect
cv::CascadeClassifier &faceCascade - Face cascade classifier
*/
void colorfulFace(cv::Mat &src, cv::Mat &dst, cv::CascadeClassifier &faceCascade) {
    std::vector<cv::Rect> faces;
    cv::Mat grey;
    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY); // Convert to grayscale
    cv::cvtColor(grey, dst, cv::COLOR_GRAY2BGR);  // Convert to BGR format for overlay
    // Detect faces
    faceCascade.detectMultiScale(grey, faces);
    // Iterate through each face
    for (const auto &face : faces) {
        // Apply a color filter to the face
        src(face).copyTo(dst(face));
    }
}

/*
Extension 1: Sharpening filter
This function sharpens the input image using a convolution operation
Arguments:
cv::Mat &src - Input image
cv::Mat &dst - Output image with sharpening effect
*/
void sharpenImage(cv::Mat &src, cv::Mat &dst) {
    // Ensure both source and destination have the same size and type
    src.copyTo(dst);
    // Sharpening kernel : Laplacian kernel
    int kernel[3][3] = {{0, -1, 0},
                        {-1, 5, -1},
                        {0, -1, 0}};

    // Iterate through each pixel in the inner region
    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            // Separate channels
            int sb = 0, sg = 0, sr = 0;
            // Apply the kernel
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    // Accumulate the values from the neighborhood
                    cv::Vec3b pixel = src.at<cv::Vec3b>(i + ki, j + kj);
                    sb += pixel[0] * kernel[ki + 1][kj + 1];
                    sg += pixel[1] * kernel[ki + 1][kj + 1];
                    sr += pixel[2] * kernel[ki + 1][kj + 1];
                }
            }
            // Assign the sharpened values to the destination pixel
            dst.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(sb);
            dst.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(sg);
            dst.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(sr);
        }
    }
}




