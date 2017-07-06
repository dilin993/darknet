//
// Created by dilin on 5/15/17.
//

#ifndef PEOPLETRACKER_PERSONTRACK_H
#define PEOPLETRACKER_PERSONTRACK_H

#define STATE_SIZE 6
#define MEAS_SIZE 4

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class PersonTrack
{
public:
    PersonTrack();

    virtual ~PersonTrack();
    PersonTrack(const Rect &bbox, int age, int totalVisibleCount, int consectiveInvisibleCount);
    PersonTrack(const Rect &bbox);

    //int id;
    Rect bbox;
    void predict(double dT);
    void take_measure(Rect detection);
    bool should_keep(bool detected);
    int rejectionTolerance = 10;
private:
    static const int TYPE = CV_32F;
    KalmanFilter kf;
    cv::Mat state;  // [x,y,v_x,v_y,w,h]
    cv::Mat meas;    // [z_x,z_y,z_w,z_h]
    bool found;
    bool prevDetected;
    int age;
    int totalVisibleCount;
    int consectiveInvisibleCount;
};


#endif //PEOPLETRACKER_PERSONTRACK_H
