#include "backend/Edge.h"
#include <iostream>

namespace backend{
unsigned long global_edge_id = 0;
Edge::Edge(int res_dimention, int num_verticies) {
    residual_.resize(res_dimention, 1);
    id_ = global_edge_id++;
    info_matrix_ = MatXX::Identity(res_dimention, res_dimention);
    observation_.resize(res_dimention, 1);
    jacobians_.resize(num_verticies);
    lossfunction_ = NULL;
}

double Edge::chi2() {
    return residual_.transpose() * info_matrix_ * residual_;
}

double Edge::robust_chi2() {
    double r_chi2;
    r_chi2 = chi2();
    if (lossfunction_) {
        Vec3 rho;
        lossfunction_->Compute(r_chi2, rho);
        r_chi2 = rho[0];
    }
    return r_chi2;
}

MatXX Edge::ComputeRobustInfo(double& drho) {
    if (lossfunction_) {
        Vec3 rho;
        double r_chi2;
        r_chi2 = chi2();
        lossfunction_->Compute(r_chi2, rho);
        MatXX robust_info(info_matrix_.rows(), info_matrix_.cols());
        drho = rho[1];
        // VecX info_res = info_matrix_ * residual_;
        // robust_info.noalias() = rho[1] * info_matrix_ + 2 * rho[2] * info_res * info_res.transpose();
        robust_info.noalias() = rho[1] * info_matrix_;
        return robust_info;
    } else {
        drho = 1;
        return info_matrix_;
    }
}
// bool Edge::CheckValid() {

// }

}