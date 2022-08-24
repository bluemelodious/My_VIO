#include "backend/Vertex.h"

namespace backend{
    unsigned long global_vertex_id = 0;
    Vertex::Vertex(int dimention, int local_dimention) {
        if (local_dimention != -1) 
            local_dimention_ = local_dimention;
        else local_dimention_ = dimention;
        parameters_.resize(dimention, 1);
        id_ = global_vertex_id++;
    }
}