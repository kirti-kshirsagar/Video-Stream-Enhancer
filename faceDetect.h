/*
  Kirti Kshirsagar | 24-01-2024

  Include file for faceDetect.cpp, face detection and drawing functions
*/
#ifndef FACEDETECT_H
#define FACEDETECT_H

// put the path to the haar cascade file here
#define FACE_CASCADE_FILE "D:/PRCV/faceDetect/haarcascade_frontalface_alt2.xml"

// prototypes
int detectFaces( cv::Mat &grey, std::vector<cv::Rect> &faces );
int drawBoxes( cv::Mat &frame, std::vector<cv::Rect> &faces, int minWidth = 50, float scale = 1.0  );

#endif
