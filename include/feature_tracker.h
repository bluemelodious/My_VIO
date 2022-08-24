#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tictoc.h"

using namespace std;
using namespace Eigen;

class FeatureTracker{
public:
    FeatureTracker() {}
    void readImg(const cv::Mat& img, double time, const Matrix3d& _relative_R = Matrix3d::Identity());
    cv::Mat prev_img_, cur_img_;
    vector<cv::Point2f> prev_pts_, cur_pts_;
    vector<cv::Point2f> prev_un_pts_, cur_un_pts_;
    vector<cv::Point2f> new_pts_;
    double cur_time_;
    double prev_time_;
    vector<int> ids_;//当前帧跟踪到的特征的id
    vector<int> track_cnt_;//当前帧跟踪到的特征的累计追踪次数
    cv::Mat mask_;
    camodocal::CameraPtr my_camera_;
private:
    int global_feature_id = 0;
    void reduceVector(vector<int>& Vec, vector<uchar>& status);
    void reduceVector(vector<cv::Point2f>& Vec, vector<uchar>& status);
    void PredictPtsWithIMU(const Matrix3d& _relative_R);
    void rejectWithF();
    void undistortedPoints();
    void setMask();
    void addPoints();
    // void updateId(int i);
    bool inBorder(const cv::Point2f& pt);
    
};