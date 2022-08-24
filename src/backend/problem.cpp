#include "backend/problem.h"
#include <iostream>


namespace backend{
void problem::addVertex(std::shared_ptr<Vertex> vertex) {
    verticies_.emplace_back(vertex);
    if (vertex->TypeInfo() == "VertexPose") {
        pose_verticies_.emplace_back(vertex);
    } else if (vertex->TypeInfo() == "VertexInverseDepth" || vertex->TypeInfo() == "VertexPointXYZ") {
        landmark_verticies_.emplace_back(vertex);
    } 

}

void problem::addEdge(std::shared_ptr<Edge> edge) {
    edges_.emplace_back(edge);
}

void problem::solve(int iters) {
    if (verticies_.size() == 0 || edges_.size() == 0) {
        std::cout << "The problem haven't been setted completly" <<std::endl;
        return;
    }
    set_ordering();
    make_hessian();
    set_initlambda();
    int iter = 0;
    bool stop = false;
    double last_loss = 1e20;
    std::cout << stopThre_ << std::endl;
    while (!stop && iter < iters) {
        std::cout <<"----" << iter << "次迭代----"<<std::endl;
        std::cout <<"delta_x:"<<delta_x_<<std::endl;
        // std::cout << "lambda:" <<current_lambda_<<std::endl;
        int false_cnt = 0;
        bool step_success = false;
        while (!step_success && false_cnt < 10) {
            solve_linear();
            update_state();
            step_success = isGoolStepInLM();
            if (step_success) {
                make_hessian();
                false_cnt = 0;
            } else {
                rollback_state();
                false_cnt++;
            }
        }
        if (last_loss - current_loss_ < stopThre_) {  
            stop = true;
        }
        last_loss = current_loss_;
        iter++;
    }
}

void problem::set_ordering() {
    ordering_poses_ = 0;
    ordering_landmarks_ = 0;
    ordering_generic_ = 0;
    if (problem_type_ == ProblemType::Slam) {
        for (auto pose : pose_verticies_) {
            pose->set_orderingId(ordering_poses_);
            ordering_poses_ += pose->get_dimen();
        }
        for (auto landmark : landmark_verticies_) {
            landmark->set_orderingId(ordering_landmarks_+ordering_poses_);
            ordering_landmarks_ += landmark->get_dimen();
        }
        ordering_generic_ = ordering_poses_ + ordering_landmarks_;
    } else if (problem_type_ == ProblemType::Generic) {
        for (auto generic : verticies_) {
            generic->set_orderingId(ordering_generic_);
            ordering_generic_ += generic->get_dimen();
        }
    }
}

void problem::make_hessian() {
    int option = acc_option_;
    // switch (option) {
    //     case 0: make_hessian_normal();
    //             break;
    //     case 1: make_hessian_OpenMP();
    //             break;
    // }
    make_hessian_normal();
}

// void problem::make_hessian_OpenMP() {

// }

void problem::make_hessian_normal() {
    unsigned long size = ordering_generic_;
    MatXX H(MatXX::Zero(size, size));
    VecX b(VecX::Zero(size));
    for (auto edge : edges_) {
        edge->ComputeResidual();
        edge->ComputeJacobians();
        std::vector<MatXX> jacobians = edge->get_jacobian();
        VecX res = edge->get_res();
        auto edge_verticies = edge->get_verticies();
        double drho;
        MatXX robust_info = edge->ComputeRobustInfo(drho);
        // std::cout << "drho:" <<drho<<std::endl;
        for (unsigned long i = 0; i < edge_verticies.size(); i++) {
            if (edge_verticies[i]->if_fixed()) continue;
            unsigned long row_begin = edge_verticies[i]->get_orderingId();
            size_t rows = edge_verticies[i]->get_dimen();
            MatXX jacobian_i = jacobians[i];
            MatXX Jtw = jacobian_i.transpose() * robust_info;
            for (unsigned long j = i; j < edge_verticies.size(); j++) {
                if (edge_verticies[j]->if_fixed()) continue;
                MatXX jacobian_j = jacobians[j];
                MatXX H_i_j = Jtw * jacobian_j;
                unsigned long col_begin = edge_verticies[j]->get_orderingId();
                size_t cols = edge_verticies[j]->get_dimen();
                H.block(row_begin, col_begin, rows, cols).noalias() += H_i_j;
                if (j != i) {
                    H.block(col_begin, row_begin, cols, rows).noalias() += H_i_j.transpose();
                }
            }
            b.segment(row_begin, rows).noalias() -= drho * jacobian_i.transpose() * edge->get_info() * res;

        }
    }
    H_ = H;
    b_ = b;
    unsigned long prior_size = H_prior_.rows();
    if (prior_size > 0) {
        for (auto vertex : verticies_) {
            if (vertex->if_fixed()) {
                unsigned long row_begin = vertex->get_orderingId();
                size_t rows = vertex->get_dimen();
                H_prior_.block(row_begin, 0, rows, prior_size).setZero();
                H_prior_.block(0, row_begin, prior_size, rows).setZero();
                b_prior_.segment(row_begin, rows).setZero();
            }
        }
        H_.topLeftCorner(prior_size, prior_size) += H_prior_;
        b_.head(prior_size) += b_prior_;
    }
}

void problem::set_initlambda() {
    //这里不将停止阈值设置为定值是很好的
    current_loss_ = compute_problem_chi2();
    stopThre_ = current_loss_ * stopThre_k_;
    double max_diagonal = 0.;
    for (unsigned long i = 0; i < H_.rows(); i++) {
        max_diagonal = std::max(fabs(H_(i, i)), max_diagonal);
    }
    max_diagonal = std::min(max_diagonal, 5e10);
    current_lambda_ = max_diagonal * init_lambda_k;
}

double problem::compute_problem_chi2() {
    double current_chi2 = 0.0;
    for (auto edge :edges_) {
        edge->ComputeResidual();
        current_chi2 += edge->robust_chi2();
    }
    if (error_prior_.rows() > 0) {
        current_chi2 += error_prior_.norm();    //此处的处理有疑问
    }
    return current_chi2 * 0.5;
}

//solve Hx = b
void problem::solve_linear() {
    delta_x_ = VecX::Zero(ordering_generic_);
    if (problem_type_ == ProblemType::Generic) {
        MatXX H_lambda = H_;
        for (unsigned long i = 0; i < H_lambda.rows(); i++) {
            H_lambda(i, i) += current_lambda_;
        }
        // std::cout << "H_lambda" << H_lambda <<"lambda:"<<current_lambda_<<std::endl;
        delta_x_ = H_lambda.ldlt().solve(b_);
    } else if (problem_type_ == ProblemType::Slam) {
        unsigned long p_length = ordering_poses_;
        unsigned long l_length = ordering_landmarks_;
        MatXX Hpp = H_.block(0, 0, p_length, p_length);
        MatXX Hpl = H_.block(0, p_length, p_length, l_length);
        MatXX Hll = H_.block(p_length, p_length, l_length, l_length);
        MatXX Hll_inv(MatXX::Zero(l_length, l_length));
        VecX bp = b_.head(p_length);
        VecX bl = b_.tail(l_length);
        MatXX Hpl_HllInv(MatXX::Zero(p_length, l_length));
        for (auto vertex : landmark_verticies_) {
            unsigned long row_begin = vertex->get_orderingId()-ordering_poses_;
            unsigned long rows = vertex->get_dimen();
            Hll_inv.block(row_begin, row_begin, rows, rows) = Hll.block(row_begin, row_begin, rows, rows).inverse();
        }
        Hpl_HllInv.noalias() = Hpl * Hll_inv;
        MatXX Hshur(MatXX::Zero(p_length, p_length));
        Hshur.noalias() = Hpp - Hpl_HllInv * Hpl.transpose();
        for (unsigned long i = 0; i < Hshur.rows(); i++) {
            Hshur(i, i) += current_lambda_;     //这里并没有直接将H+lambda*I，应该是为了减小计算量
        }
        VecX bshur(VecX::Zero(p_length));
        bshur.noalias() = bp - Hpl_HllInv * bl;
        VecX delta_xp = Hshur.ldlt().solve(bshur);
        VecX delta_xl = Hll_inv * (bl - Hpl.transpose() * delta_xp);
        delta_x_.head(p_length) = delta_xp;
        delta_x_.tail(l_length) = delta_xl;
    }
}

void problem::update_state() {
    for (auto vertex : verticies_) {
        vertex->backup_parameter();
        VecX delta_x = delta_x_.segment(vertex->get_orderingId(), vertex->get_dimen());
        vertex->plus(delta_x);
    }
    //状态量更新之后，由于上一次边缘化所带来的约束（比如两个共视相机pose之间）无法重新计算J，只能通过以下方式（先验的J）来更新这些约束的误差和b
    if (error_prior_.rows() > 0) {
        b_prior_back_ = b_prior_;
        error_prior_back_ = error_prior_;
        b_prior_ += H_prior_ * delta_x_.head(ordering_poses_);//此处写的和源代码不一样，怀疑是源代码错误
        error_prior_ = - Jt_prior_inverse_ * b_prior_.head(ordering_poses_-15);//???为何要-15，为何不去除信息矩阵
    }
    
}

void problem::rollback_state() {
    for (auto vertex : verticies_) {
        vertex->rollback_parameter();
    }
    if (error_prior_.rows() > 0) {
        b_prior_ = b_prior_back_;
        error_prior_ = error_prior_back_;
    }
    
}

bool problem::isGoolStepInLM() {
    double tmp_loss = compute_problem_chi2();
    double fit_loss_gap = 0.5 * delta_x_.transpose() * (b_ + current_lambda_ * delta_x_);
    fit_loss_gap += 1e-10;   //避免分母是0
    double rho = (current_loss_ - tmp_loss) / fit_loss_gap;
    std::cout << "current_loss_" << current_loss_ << "tmp_loss" << tmp_loss <<"fit_loss_gap"<< fit_loss_gap;
    switch (LM_lambda_option_) {
        case LM_lambda_option::Marquardt :{
            if (rho < 0.25) current_lambda_ *= 2;
            else if (rho > 0.75) current_lambda_ /= 3.;
            break;
        }
        case LM_lambda_option::Nielsen :{
            if (rho > 0) {
                current_lambda_ = std::max(1./3., std::min(1.-std::pow(2*rho-1, 3), 2./3.));
                current_Nielsen_v_ = 2;
            } else {
                current_lambda_ *= current_Nielsen_v_;
                current_Nielsen_v_ *= 2;
            }
            break;
        }
    }
    // std::cout <<"rho:"<<rho<<"lambda_try:" <<current_lambda_<<std::endl;
    if (rho > 0) current_loss_ = tmp_loss;
    return rho > 0;
}


}