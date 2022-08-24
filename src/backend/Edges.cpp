#include "backend/Edges.h"
#include <sophus/se3.h>

namespace backend{
void EdgeReprojectionXYZ::ComputeResidual() {
    Vec3 Pw = verticies_[0]->get_parameters();
    VecX pose = verticies_[1]->get_parameters();
    Qd qwi(pose[6], pose[3], pose[4], pose[5]);
    Vec3 twi(pose.head(3));
    Vec3 Pi = qwi.inverse() * (Pw - twi);
    Vec3 Pc = qic_.inverse() * (Pi - tic_);
    residual_ = Vec2(Pc.x() / Pc.z(), Pc.y() / Pc.z()) - observation_;
}

void EdgeReprojectionXYZ::ComputeJacobians() {
    Vec3 Pw = verticies_[0]->get_parameters();
    VecX pose = verticies_[1]->get_parameters();
    Qd qwi(pose[6], pose[3], pose[4], pose[5]);
    Vec3 twi(pose.head(3));
    Vec3 Pi = qwi.inverse() * (Pw - twi);
    Vec3 Pc = qic_.inverse() * (Pi - tic_);
    Qd qcw = (qwi * qic_).inverse(); 
    Mat33 Ric = qic_.toRotationMatrix();
    Mat33 J_q = Ric.transpose() * Sophus::SO3::hat(Pi);  //关于左乘扰动和右乘扰动，只需要保证算雅克比时与更新状态时统一即可
    Mat33 J_p = qcw.toRotationMatrix();
    Mat33 J_t = -J_p;
    Mat36 J_pose;
    J_pose.leftCols(3) = J_q;
    J_pose.rightCols(3) = J_t;
    Mat23 J_k;
    J_k << 1./Pc.z(), 0, - Pc.x() / (Pc.z() * Pc.z()),
            0, 1./Pc.z(), - Pc.y() / (Pc.z() * Pc.z());
    jacobians_[0] = J_k * J_p;
    jacobians_[1] = J_k * J_pose;
}

void EdgeReprojectionXYZOnly::ComputeResidual() {
    Vec3 Pw = verticies_[0]->get_parameters();
    Vec3 Pc = qcw_ * Pw + tcw_;
    residual_ = Vec2(Pc.x() / Pc.z(), Pc.y() / Pc.z()) - observation_;
}

void EdgeReprojectionXYZOnly::ComputeJacobians() {
    Vec3 Pw = verticies_[0]->get_parameters();
    Vec3 Pc = qcw_ * Pw + tcw_;
    Mat33 J_p = qcw_.toRotationMatrix();
    Mat23 J_k;
    J_k << 1./Pc.z(), 0, - Pc.x() / (Pc.z() * Pc.z()),
            0, 1./Pc.z(), - Pc.y() / (Pc.z() * Pc.z());
    jacobians_[0] = J_k * J_p;
}
}