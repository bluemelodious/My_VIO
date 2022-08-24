#ifndef EIGEN_TYPES_H
#define EIGEN_TYPES_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <memory>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatXX;
typedef Eigen::Matrix<double, 2, 3> Mat23;
typedef Eigen::Matrix<double, 3, 6> Mat36;
typedef Eigen::Matrix3d Mat33;

typedef Eigen::VectorXd VecX;
typedef Eigen::Vector3d Vec3;
typedef Eigen::Vector2d Vec2;
typedef Eigen::Quaterniond Qd;

#endif