#pragma once
#include "Edge.h"

namespace backend{
class EdgeReprojectionXYZ : public Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeReprojectionXYZ() : Edge(2, 2) {}

    virtual std::string TypeInfo() const override {return "EdgeReprojectionXYZ";}

    virtual void ComputeResidual() override;

    virtual void ComputeJacobians() override;

    void SetTranslationImuFromCamera(const Qd& qic, const Vec3& tic) {
        qic_ = qic;
        tic_ = tic;
    }
private:
    Qd qic_;
    Vec3 tic_;
};

class EdgeReprojectionXYZOnly : public Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeReprojectionXYZOnly() : Edge(2, 1) {}

    virtual std::string TypeInfo() const override {return "EdgeReprojectionXYZOnly";}

    virtual void ComputeResidual() override;

    virtual void ComputeJacobians() override;

    void SetTranslationCameraFromWorld(const Qd& qcw, const Vec3& tcw) {
        qcw_ = qcw;
        tcw_ = tcw;
    }

private:
    Qd qcw_;
    Vec3 tcw_;
};
}