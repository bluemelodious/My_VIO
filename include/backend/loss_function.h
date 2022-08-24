#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H
#include <Eigen/Core>

namespace backend {

// enum class Loss_Function_Type {
//     TrivalLoss,
//     HuberLoss,
//     CauchyLoss
// };
class LossFunction
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void Compute(double& chi2, Eigen::Vector3d& rho) const = 0;
};

class TrivalLoss : public LossFunction {
public:
    virtual void Compute(double& chi2, Eigen::Vector3d& rho) const {
        rho[0] = chi2;
        rho[1] = 1;
        rho[2] = 0;
    }
};

class HuberLoss : public LossFunction {
public:
    HuberLoss() {}
    HuberLoss(double c) : c_(c) {}
    virtual void Compute(double& chi2, Eigen::Vector3d& rho) const override;
private:
    double c_ = 1.345;
};

class CauchyLoss : public LossFunction {
public:
    CauchyLoss() {}
    CauchyLoss(double c) : c_(c) {}
    virtual void Compute(double& chi2, Eigen::Vector3d& rho) const override;
private:
    double c_ = 2.3849;
};




} // namespace backend

#endif