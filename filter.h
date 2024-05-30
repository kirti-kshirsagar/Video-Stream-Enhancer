/* 
  Kirti Kshirsagar | 23-01-2024 

  Include file for filter.cpp, filtering functions and face detection functions
*/

#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>

// prototypes
int greyscale(cv::Mat &src, cv::Mat &dst);
void sepiaTone(cv::Mat& src, cv::Mat& dst);
int blur5x5_1(cv::Mat &src, cv::Mat &dst);
int blur5x5_2(cv::Mat &src, cv::Mat &dst);
int sobelX3x3( cv::Mat &src, cv::Mat &dst );
int sobelY3x3( cv::Mat &src, cv::Mat &dst );
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );
int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels );
void createNegative(const cv::Mat& src, cv::Mat& dst);
void applyBoxFilter(cv::Mat& src, cv::Mat& dst, int kernelSize);
void colorfulFace(cv::Mat &src, cv::Mat &dst, cv::CascadeClassifier &faceCascade);
void sharpenImage(cv::Mat &src, cv::Mat &dst);


#endif 
