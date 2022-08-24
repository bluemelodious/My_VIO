#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <thread>
#include <iomanip>

#include "feature_tracker.h"
#include "tictoc.h"

using namespace std;
using namespace Eigen;
using namespace cv;


int main() {
    // 检查光流
    // string img_name_path = "/home/alen/Documents/dataset/TUM/rgbd_dataset_freiburg1_xyz/associations.txt";
    // string img_file_path = "/home/alen/Documents/dataset/TUM/rgbd_dataset_freiburg1_xyz/";
    // ifstream ins_img;
    // ins_img.open(img_name_path);
    // if (!ins_img.is_open()) {
    //     cout << "can't open the file" <<endl;
    //     return 0;
    // }
    // shared_ptr<FeatureTracker> tracker(new FeatureTracker());
    // string s, time_stamp, img_name, img_path;
    // while (getline(ins_img, s) && !s.empty()) {
    //     istringstream iss(s);
    //     iss >> time_stamp>> img_name;
    //     img_path = img_file_path + img_name;
    //     // cout << "img_path" <<img_path<<endl;
    //     Mat img = imread(img_path, 0);
    //     if (img.empty()) {
    //         cout << "image is empty!" <<endl;
    //         return 0;
    //     }
    //     tracker->readImg(img, stod(time_stamp));
    //     Mat show_img;
    //     cvtColor(img, show_img, CV_GRAY2RGB);
    //     for (auto pt : tracker->cur_pts_) {
    //         circle(show_img, pt, 3, Scalar(0, 255, 120), 2);
    //     }
    //     for (auto pt : tracker->new_pts_) {
    //         circle(show_img, pt, 3, Scalar(0, 0, 0), 2);
    //     }
    //     cv::namedWindow("tracker");                      
    //     cv::imshow("image", show_img);
    //     cv::waitKey(50);
    // }
    // 对比不同角点提取效果和时间
    string img_path = "/home/alen/Documents/dataset/TUM/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png";
    Mat img = imread(img_path, 0);
    vector<cv::Point2f> pts;
    TicToc m;
    cv::goodFeaturesToTrack(img,pts, 150, 0.01, 30);
    cout << "cost " << m.toc() << "ms" <<endl;
    Mat show_img1;
    cvtColor(img, show_img1, CV_GRAY2RGB);
    for (auto pt : pts) {
        circle(show_img1, pt, 3, Scalar(0, 255, 120), 2);
    }
    cv::namedWindow("tracker");
    cv::imshow("image", show_img1);
    cv::waitKey(0);
    vector<cv::KeyPoint> pts_;
    TicToc n;
    cv::FAST(img, pts_, 80, true);
    cout << "cost " << n.toc() << "ms" <<endl;
    Mat show_img2;
    cvtColor(img, show_img2, CV_GRAY2RGB);
    for (auto pt : pts_) {
        circle(show_img2, pt.pt, 3, Scalar(0, 255, 120), 2);
    }
    cv::namedWindow("tracker");
    cv::imshow("image", show_img2);
    cv::waitKey(0);
}