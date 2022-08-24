#include "backend/Verticies.h"
#include <sophus/se3.h>

namespace backend{
void VertexPose::plus(const VecX& params) {
    parameters_.head(3) += params.head(3);
    Qd q(parameters_(3), parameters_(4), parameters_(5), parameters_(6));
    q = q * Sophus::SO3::exp(Vec3(params(0), params(1), params(2))).unit_quaternion();
    q.normalize();
    parameters_(3) = q.x();
    parameters_(4) = q.y();
    parameters_(5) = q.z();
    parameters_(6) = q.w();
}
}