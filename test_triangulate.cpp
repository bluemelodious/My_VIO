#include <iostream>
#include <vector>
#include <random>  
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include "backend/problem.h"
#include "tictoc.h"
#include <memory>

struct Pose
{
    Pose(Eigen::Matrix3d R, Eigen::Vector3d t):Rwc(R),qwc(R),twc(t) {};
    Eigen::Matrix3d Rwc;
    Eigen::Quaterniond qwc;
    Eigen::Vector3d twc;

    Eigen::Vector2d uv;    // 这帧图像观测到的特征坐标
};
using namespace backend;

int main()
{

    int poseNums = 10;
    double radius = 8;
    double fx = 1.;
    double fy = 1.;
    // std::vector<std::shared_ptr<Vertex>> camera_poses;
    // for(int n = 0; n < poseNums; ++n ) {
    //     double theta = n * 2 * M_PI / ( poseNums * 4); // 1/4 圆弧
    //     // 绕 z轴 旋转
    //     Eigen::Matrix3d R;
    //     R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
    //     Qd q(R);
    //     Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
    //     std::shared_ptr<VertexPose> vertex_pose = std::make_shared<VertexPose>();
    //     VecX init_pose(VecX::Zero(7));
    //     init_pose.head(3) = t;
    //     init_pose(3) = q.x();init_pose(4) = q.y();init_pose(5) = q.z();init_pose(6) = q.w();

    //     vertex_pose->setInit(init_pose);
    //     camera_poses.push_back(vertex_pose);

    // }
    std::vector<Pose> camera_poses;
    for(int n = 0; n < poseNums; ++n ) {
        double theta = n * 2 * M_PI / ( poseNums * 4); // 1/4 圆弧
        // 绕 z轴 旋转
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
        Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        camera_poses.push_back(Pose(R,t));
    }

    // 随机数生成 1 个 三维特征点
    std::default_random_engine generator;
    std::uniform_real_distribution<double> xy_rand(-4, 4.0);
    std::uniform_real_distribution<double> z_rand(8., 10.);
    double tx = xy_rand(generator);
    double ty = xy_rand(generator);
    double tz = z_rand(generator);

    Eigen::Vector3d Pw(tx, ty, tz);
    double sigma = 0.05;    //pixel/facal
    std::normal_distribution<double> duv_rand(0, sigma);
    

    // 这个特征从第三帧相机开始被观测，i=3
    int start_frame_id = 3;
    int end_frame_id = poseNums;
    for (int i = start_frame_id; i < end_frame_id; ++i) {
        Eigen::Matrix3d Rcw = camera_poses[i].Rwc.transpose();
        Eigen::Vector3d Pc = Rcw * (Pw - camera_poses[i].twc);

        double x = Pc.x();
        double y = Pc.y();
        double z = Pc.z();

        double du = duv_rand(generator);
        double dv = duv_rand(generator);
        camera_poses[i].uv = Eigen::Vector2d(x/z + du, y/z + dv);
        // camera_poses[i].uv = Eigen::Vector2d(x/z, y/z);
    }
    std::shared_ptr<VertexPointXYZ> vertex_pt = std::make_shared<VertexPointXYZ>();
    std::shared_ptr<problem> solver(new problem(problem::ProblemType::Generic));
    solver->setLM_lambda(problem::LM_lambda_option::Nielsen);
    // auto kernel = new CauchyLoss(1.0);
    auto kernel = new TrivalLoss();
    solver->addVertex(vertex_pt);

    // 遍历所有的观测数据，并三角化
    Eigen::Vector3d P_est;           // 结果保存到这个变量
    P_est.setZero();
    /* your code begin */
    int size = end_frame_id-start_frame_id;
    Eigen::MatrixXd D(size*2, 4);
    int index = 0;
    for (int i = start_frame_id; i < end_frame_id; ++i) {
        Eigen::Matrix3d Rcw = camera_poses[i].Rwc.transpose();
        Eigen::Vector3d tcw = -Rcw * camera_poses[i].twc;
        Eigen::Vector2d uv = camera_poses[i].uv;
        Eigen::Matrix<double, 3, 4> P;
        P.block(0, 0, 3, 3) = Rcw;
        P.block(0, 3, 3, 1) = tcw;
        D.row(index*2) = P.row(2) * uv.x() -  P.row(0);
        D.row(index*2+1) = P.row(2) * uv.y() -  P.row(1);

        std::vector<std::shared_ptr<Vertex>> verticies;
        verticies.emplace_back(vertex_pt);
        std::shared_ptr<EdgeReprojectionXYZOnly> edge = std::make_shared<EdgeReprojectionXYZOnly>();
        edge->set_lossfunction(kernel);
        edge->SetVertex(verticies);
        edge->set_observation(uv);
        edge->SetTranslationCameraFromWorld(Qd(Rcw), tcw);
        solver->addEdge(edge);

        index++;
    }
    Eigen::Matrix<double, 4, 4> DTD = D.transpose() * D;
    std::cout << "DTD: " << DTD << std::endl;
    Eigen::VectorXd sin;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(D, Eigen::ComputeThinU | Eigen::ComputeThinV);
    sin = svd.singularValues();
    std::cout << "sing: " << sin << std::endl;
    Eigen::Matrix4d V = svd.matrixV();
    std::cout << "V: " << V << std::endl;
    P_est << V(0, 3)/V(3, 3), V(1, 3)/V(3, 3), V(2, 3)/V(3, 3);
   
    vertex_pt->setInit(P_est);
    TicToc m;
    solver->solve(30);
    std::cout <<"cost time"<< m.toc() <<std::endl;
    std::cout <<"ground truth: \n"<< Pw.transpose() <<std::endl;
    std::cout <<"init result: \n"<< P_est.transpose() <<std::endl;
    std::cout <<"solve result: \n"<< vertex_pt->get_parameters().transpose() <<std::endl;


}
