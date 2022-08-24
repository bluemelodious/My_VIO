#include <random>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include "backend/problem.h"

using namespace backend;

class CurveFittingVertex : public Vertex {
public:
    CurveFittingVertex() : Vertex(3) {}
    virtual void plus(const VecX& params) override {
        parameters_+=params;
    }

    virtual std::string TypeInfo() const override {
        return "CurveFittingVertex";
    }
};

class CurveFittingEdge : public Edge {
public:
    CurveFittingEdge(double x, double y) : Edge(1, 1), x_(x), y_(y) {}
    virtual std::string TypeInfo() const override {
        return "CurveFittingEdge";
    }

    virtual void ComputeResidual() override {
        VecX abc = verticies_[0]->get_parameters();
        residual_(0) = std::exp(abc(0) *x_*x_ + abc(1) *x_ + abc(2)) - y_;
    }

    virtual void ComputeJacobians() override {
        VecX abc = verticies_[0]->get_parameters();
        double pre = std::exp(abc(0) *x_*x_ + abc(1) *x_ + abc(2));
        MatXX jacobian(1, 3);
        jacobian << x_ * x_ * pre, x_ * pre, pre;
        jacobians_[0] = jacobian;
    }
private:
    double x_;
    double y_;
};

int main() {
    double a = 1., b = 2., c = 1.;
    double sigma = 1.;
    int N = 100;
    std::default_random_engine generator;
    std::normal_distribution<double> noise(0, sigma);
    std::shared_ptr<CurveFittingVertex> vertex(new CurveFittingVertex());
    vertex->setInit(Vec3 (0., 0., 0.));
    std::shared_ptr<problem> solver(new problem(problem::ProblemType::Generic));
    solver->setLM_lambda(problem::LM_lambda_option::Nielsen);
    // auto kernel = new CauchyLoss(1.0);
    // auto kernel = new TrivalLoss();
    auto kernel = new HuberLoss();
    solver->addVertex(vertex);

    for (int i = 0; i < N; i++) {
        double x = i/100.;
        double n = noise(generator);
        double y = exp(a*x*x + b*x + c) + n;
        std::vector<std::shared_ptr<Vertex>> edge_vertexs;
        std::shared_ptr<CurveFittingEdge> edge(new CurveFittingEdge(x, y));
        edge->set_lossfunction(kernel);
        edge_vertexs.emplace_back(vertex);
        edge->SetVertex(edge_vertexs);
        solver->addEdge(edge);
    }
    auto begin = std::chrono::system_clock::now();
    solver->solve(30);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> cost_time = end - begin;
    std::cout << "cost_time: " << cost_time.count() * 1000 << std::endl;
    std::cout << "solve result: " << vertex->get_parameters().transpose() << std::endl;
    std::cout << "true result: " << a << b << c << std::endl;
}
