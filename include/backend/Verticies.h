#pragma once
#include "Vertex.h"

namespace backend{
class VertexPose : public Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexPose() : Vertex(7, 6) {}
    virtual std::string TypeInfo() const override {
        return "VertexPose";
    }
    virtual void plus(const VecX& params) override;
};

class VertexInverseDepth : public Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexInverseDepth() : Vertex(1) {}
    virtual std::string TypeInfo() const override {
        return "VertexInverseDepth";
    }
};

class VertexPointXYZ : public Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexPointXYZ() : Vertex(3) {}
    virtual std::string TypeInfo() const override {
        return "VertexPointXYZ";
    }
};
}