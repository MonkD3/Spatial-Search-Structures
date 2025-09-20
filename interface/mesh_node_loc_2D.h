#ifndef __INTERFACE_MNL_2D__
#define __INTERFACE_MNL_2D__

#include "point.h"
#include "bbox.h"
#include "quadtree.hpp"

using VecT = Vec2d;
using BBoxT = BBox2d;
using ScalarType = double;

constexpr int dim = 2;
constexpr int bucketsize = 4;
constexpr int depth = 12;

struct Tri {
    using Scalar = ScalarType;

    VecT a;  
    VecT b; 
    VecT c;
    
    Tri(const VecT& _a, const VecT& _b, const VecT& _c) noexcept;
    VecT  get_centroid() const;
    BBoxT get_bbox()     const;
};

struct ComputeMeshTree2DInternalData {
    using Scalar = Tri::Scalar;

    BBoxT operator()(const Tri& tri) const;
    BBoxT operator()(const BBoxT& e1, const BBoxT& e2) const;
};

using MeshTree2D = Quadtree<Tri, ComputeMeshTree2DInternalData, dim, bucketsize, depth>; 
using Node = MeshTree2D::Node;

#ifdef __cplusplus
extern "C" {
#endif

/* @brief Create the surface spatial search structure
 *
 * @param nodes : coordinates of the nodes stored in row-major format :
 *                x0 y0 x1 y1 ... 
 * @param edges : Indices of the nodes forming edges, stored in row major format :
 *                e00 e01 e10 e11 ...
 * @param nnodes : number of nodes
 * @param nedges : number of edges
 */
MeshTree2D* MeshTree2D_create(
    ScalarType const* const __restrict__ nodes,
    uint32_t const * const __restrict__ tri,
    uint32_t nnodes,
    uint32_t ntri
);

/* @brief Free the memory of the tree
 *
 * @param tree : the pointer to the memory
 */
void MeshTree2D_destroy(MeshTree2D* tree);

/* @brief Queries of 2D distance to surface
 *
 * @param t : the tree
 * @param q : the query coordinates stored in row-major format :
 *            x0 y0 x1 y1 ...
 * @param d : the resulting triangle each coordinates is located in.
 *            If the node is not in a triangle, return -1
 */
void MeshTree2D_query(
    MeshTree2D * const __restrict__ t,
    ScalarType const * const __restrict__ q,
    int32_t * const __restrict__ tri_id,
    uint32_t nqueries
);

#ifdef __cplusplus
}
#endif

#endif // __INTERFACE_MNL_2D__
