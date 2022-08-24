#include "feature_tracker.h"


const int MAX_CNT = 150;
const int ROW = 480;
const int COL = 640;
const int MIN_DIST = 30;

bool FeatureTracker::inBorder(const cv::Point2f& pt) {
    int x = cvRound(pt.x);
    int y = cvRound(pt.y);
    const int border = 1;
    return x-border>0 && x<COL-border && y-border>0 && y<ROW-border;
}

void FeatureTracker::reduceVector(vector<int>& Vec, vector<uchar>& status) {
    int j = 0; 
    for (int i = 0; i < Vec.size(); i++) {
        if (status[i]) Vec[j++] = Vec[i];
    }
}

void FeatureTracker::reduceVector(vector<cv::Point2f>& Vec, vector<uchar>& status) {
    int j = 0; 
    for (int i = 0; i < Vec.size(); i++) {
        if (status[i]) Vec[j++] = Vec[i];
    }
}

void FeatureTracker::readImg(const cv::Mat& input_img, double time, const Matrix3d &_relative_R) {
    cv::Mat img;
    img = input_img;
    cur_img_ = img;

    cur_pts_.clear();
    if (prev_pts_.size() > 0) {
        vector<uchar> status(prev_pts_.size());
        vector<float> err;
        if (USE_IMU) {
            PredictPtsWithIMU(_relative_R);
            cv::calcOpticalFlowPyrLK(prev_img_, cur_img_, prev_pts_, cur_pts_, status, err, cv::Size(21, 21), 1, 
                                    cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
        } else {
            cv::calcOpticalFlowPyrLK(prev_img_, cur_img_, prev_pts_, cur_pts_, status, err, cv::Size(21, 21), 3);
        }
        for (int i = 0; i < cur_pts_.size(); i++) {
            if (status[i] && !inBorder(cur_pts_[i])) 
                status[i] = 0;
        }
        reduceVector(cur_pts_, status);
        reduceVector(prev_pts_, status);//这里为什么要更新上一帧的特征的原因是之后要对两帧做ransac剔除外点，相同的特征必须一一对应
        reduceVector(track_cnt_, status);
        reduceVector(ids_, status);
    }
    for (auto cnt : track_cnt_) {
        cnt++;
    }
    // if (PUB_THIS_FRAME) {
        rejectWithF();//极线约束
        setMask();
        new_pts_.clear();
        int need_new_pts = MAX_CNT - cur_pts_.size();
        if (need_new_pts > 0) {
            if (mask_.size() != cur_img_.size()) cout << "mask's size is wrong"<< endl;
            cv::goodFeaturesToTrack(cur_img_, new_pts_, need_new_pts, 0.01, MIN_DIST, mask_);
        }
        addPoints();
    // }
    prev_pts_ = cur_pts_;
    prev_img_ = cur_img_;
    prev_time_ = cur_time_;
    undistortedPoints();
    prev_un_pts_ = cur_un_pts_;
}

void FeatureTracker::rejectWithF() {
    if (cur_pts_.size() > 8) {
        int pts_size = cur_pts_.size();
        // vector<cv::Point2f> prev_un_pts(pts_size), cur_un_pts(pts_size);
        // for (int i = 0; i < pts_size; i++) {
        //     Vector3d tmp;
        //     my_camera_->liftProjective(Vector2d(cur_pts_[i].x, cur_pts_[i].y), tmp);
        //     tmp.x() = FOCAL_LENGTH * tmp.x() / tmp.z() + COL / 2;
        //     tmp.y() = FOCAL_LENGTH * tmp.y() / tmp.z() + ROW / 2;
        //     cur_un_pts[i] = cv::Point2d(tmp.x(), tmp.y());

        //     my_camera_->liftProjective(Vector2d(prev_pts_[i].x, prev_pts_[i].y), tmp);
        //     tmp.x() = FOCAL_LENGTH * tmp.x() / tmp.z() + COL / 2;
        //     tmp.y() = FOCAL_LENGTH * tmp.y() / tmp.z() + ROW / 2;
        //     prev_un_pts[i] = cv::Point2d(tmp.x(), tmp.y());
        // }
        vector<uchar> status(pts_size);
        cv::findFundamentalMat(prev_pts_, cur_pts_, cv::FM_RANSAC, 1.0, 0.99, status);
        // cv::findFundamentalMat(prev_un_pts, cur_un_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        reduceVector(cur_pts_, status);
        reduceVector(track_cnt_, status);
        reduceVector(ids_, status);

    }
}

void FeatureTracker::setMask() {
    mask_ = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    vector<pair<int, pair<int, cv::Point2f>>> track_cnt_pts(cur_pts_.size());
    for (int i = 0; i < cur_pts_.size(); i++) {
        track_cnt_pts.push_back(pair<int, pair<int, cv::Point2f>>(track_cnt_[i], pair<int, cv::Point2f>(ids_[i], cur_pts_[i])));
    }
    sort(track_cnt_pts.begin(),track_cnt_pts.end(),[](const pair<int, pair<int, cv::Point2f>>& x,const pair<int, pair<int, cv::Point2f>>& y){
        return x.first>y.first;});
    cur_pts_.clear();
    track_cnt_.clear();
    ids_.clear();
    for (auto &track_cnt_pt : track_cnt_pts) {
        if (mask_.at<uchar>(track_cnt_pt.second.second) != 0) {
            cv::circle(mask_, track_cnt_pt.second.second, MIN_DIST, 0, -1);
            cur_pts_.push_back(track_cnt_pt.second.second);
            track_cnt_.push_back(track_cnt_pt.first);
            ids_.push_back(track_cnt_pt.second.first);
        }
    }
}

void FeatureTracker::addPoints() {
    for (auto &pt : new_pts_) {
        cur_pts_.push_back(pt);
        track_cnt_.push_back(1);
        ids_.push_back(global_feature_id++);//这里有修改
    }
}

void FeatureTracker::undistortedPoints() {
    // for (int i = 0; i < cur_pts_.size(); i++) {
    //     Vector3d tmp;
    //     my_camera_->liftProjective(Vector2d(cur_pts_[i].x, cur_pts_[i].y), tmp);
    //     cur_un_pts_[i] = cv::Point2d(tmp.x()/ tmp.z(), tmp.y() / tmp.z());
    // }
    cur_un_pts_ = cur_pts_;

}

void FeatureTracker::PredictPtsWithIMU(const Matrix3d& _relative_R) {
    Vector3d tmp_3d1;
    Vector3d tmp_3d2;
    Vector2d tmp_p;
    for (auto i : prev_pts_) {
        my_camera_->liftProjective(Vector2d(i.x, i.y), tmp_3d1);
        tmp_3d2 = _relative_R * tmp_3d1;
        my_camera_->spaceToPlane(tmp_3d2, tmp_p);
        cur_pts_.push_back(cv::Point2f(tmp_p.x(), tmp_p.y()));
    }
}
// void FeatureTracker::updateId(int i) {
//     if (ids_[i] == -1) {
//         ids_[i] = global_feature_id++;
//     }
// }