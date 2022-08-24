#ifndef PROBLEM_H
#define PROBLEM_H

#include "Edge.h"
#include "Vertex.h"
#include "Edges.h"
#include "Verticies.h"

namespace backend{
class problem{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    enum class ProblemType{
        Generic,
        Slam
    };
    enum class LM_lambda_option{
        Marquardt,
        Nielsen
    };
    problem(ProblemType problem_type) : problem_type_(problem_type) {};

    void setLM_lambda(LM_lambda_option LM_lambda_option) {LM_lambda_option_ = LM_lambda_option;}

    void addVertex(std::shared_ptr<Vertex> vertex);

    void addEdge(std::shared_ptr<Edge> edge);

    void solve(int iters);

    void set_ordering();

    void make_hessian();

    void make_hessian_normal();

    // void make_hessian_OpenMP();

    void set_initlambda();

    double compute_problem_chi2();

    void solve_linear();

    void update_state();

    bool isGoolStepInLM();

    void rollback_state();


private:
    ProblemType problem_type_;
    std::vector<std::shared_ptr<Vertex>> pose_verticies_;
    std::vector<std::shared_ptr<Vertex>> landmark_verticies_;
    std::vector<std::shared_ptr<Vertex>> verticies_;
    std::vector<std::shared_ptr<Edge>> edges_;
    MatXX H_;
    VecX b_;
    MatXX H_prior_;
    VecX b_prior_;
    VecX b_prior_back_;
    VecX error_prior_;
    VecX error_prior_back_;
    VecX delta_x_;
    MatXX Jt_prior_inverse_;
    double current_loss_;
    unsigned long ordering_poses_ = 0;
    unsigned long ordering_landmarks_ = 0;
    unsigned long ordering_generic_ = 0;
    int acc_option_ = 0;
    LM_lambda_option LM_lambda_option_;
    double current_lambda_ = -1.;
    double current_Nielsen_v_ = 2;
    double init_lambda_k = 1e-5;
    double stopThre_ = 0.;
    double stopThre_k_ = 1e-10;
};
}

#endif