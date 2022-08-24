#include "backend/loss_function.h"
#include <iostream>

namespace backend{

void HuberLoss::Compute(double& chi2, Eigen::Vector3d& rho) const {
    double c_2 = c_ * c_;
    if (chi2 <= c_2) {
        rho[0] = chi2;
        rho[1] = 1;
        rho[2] = 0;
    }
    double sqrt_chi2 = sqrt(chi2);
    rho[0] = 2 * c_ * sqrt_chi2 - c_2;
    rho[1] = c_ / sqrt_chi2;
    rho[2] = -0.5 *rho[1] / chi2;
}

void CauchyLoss::Compute(double& chi2, Eigen::Vector3d& rho) const {
    double c_2 = c_ * c_;
    double tmp = 1.+chi2/(c_2);
    rho[0] = c_2 * log(tmp);
    rho[1] = 1./tmp;
    rho[2] = -1.*rho[1]*rho[1]/c_2;
}

}