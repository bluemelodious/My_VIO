#ifndef EDGE_H
#define EDGE_H

#include "Vertex.h"
#include "loss_function.h"

namespace backend{
class Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Edge(int res_dimention, int num_verticies);

    unsigned long ID() const {return id_;}  

    void SetVertex(const std::vector<std::shared_ptr<Vertex>>& verticies) {
        verticies_ = verticies;
    }

    void addVertex(const std::shared_ptr<Vertex>& vertex) {
        verticies_.emplace_back(vertex);
    }

    std::vector<std::shared_ptr<Vertex>> get_verticies() {
        return verticies_;
    }

    std::shared_ptr<Vertex> get_vertex(int n) {
        return verticies_[n];
    }

    int num_verticies() {
        return verticies_.size();
    }

    void set_lossfunction(LossFunction* lossfunction) {lossfunction_ = lossfunction;}
    
    virtual std::string TypeInfo() const = 0;

    virtual void ComputeResidual() = 0;

    virtual void ComputeJacobians() = 0;

    double chi2();//计算平方误差，会乘以信息矩阵

    double robust_chi2();

    MatXX ComputeRobustInfo(double& drho);

    VecX get_res() const {return residual_;}

    std::vector<MatXX> get_jacobian() const {return jacobians_;}

    void set_info(const MatXX& info) {info_matrix_ = info;} 

    MatXX get_info() const {return info_matrix_;}

    void set_observation(const VecX& obs) {observation_ = obs;}

    VecX get_observation() const {return observation_;}

    // bool CheckValid(); 

    unsigned long get_orderingId() const {return ordering_id_;}

    void set_orderingId(unsigned long id) {ordering_id_ = id;}
protected:
    unsigned long id_;
    unsigned long ordering_id_;
    std::vector<std::shared_ptr<Vertex>> verticies_;
    VecX residual_;
    std::vector<MatXX> jacobians_;
    MatXX info_matrix_;
    VecX observation_;
    LossFunction* lossfunction_;
};

}
#endif 