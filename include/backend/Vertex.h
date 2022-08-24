#ifndef VERTEX_H
#define VERTEX_H

#include "eigen_types.h"
namespace backend {

class Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    Vertex(int dimention, int local_dimention = -1);
    void setInit(VecX init) {parameters_ = init;}

    int get_dimen() const {return local_dimention_;}

    unsigned long ID() const {return id_;}

    void backup_parameter() {last_parameters_ = parameters_;}

    void rollback_parameter() {parameters_ = last_parameters_;}

    VecX get_parameters() const {return parameters_;}

    VecX& get_parameters() {return parameters_;}

    virtual void plus(const VecX& params) {parameters_+=params;}

    virtual std::string TypeInfo() const = 0;// 定义纯虚类

    unsigned long get_orderingId() const {return ordering_id_;}

    void set_orderingId(unsigned long id) {ordering_id_ = id;}

    bool if_fixed() const {return fixed_;}

    void setfix() {fixed_ = true;}
    
protected:
    VecX parameters_;
    int local_dimention_;
    VecX last_parameters_;
    unsigned long id_;
    unsigned long ordering_id_;
    bool fixed_ = false;
};
}
#endif