#ifndef __INTERFACE_SD_2D__
#define __INTERFACE_SD_2D__

#include "point.h"
#include "bbox.h"
#include "quadtree.hpp"

using VecT = Vec2d;
using BBoxT = BBox2d;
using ScalarType = double;

constexpr int dim = 2;
constexpr int bucketsize = 4;
constexpr int depth = 12;
constexpr ScalarType beta = 2.0;

struct Edge {
    using Scalar = ScalarType;

    VecT s;  // start
    VecT t;  // target
    
    Edge(const VecT& _s, const VecT& _t) noexcept;
    VecT  get_centroid() const;
    BBoxT get_bbox() const;
};

struct ComputeInternalData {
    using Scalar = Edge::Scalar;

    BBoxT operator()(const Edge& edge) const;
    BBoxT operator()(const BBoxT& e1, const BBoxT& e2) const;
};

using SurfaceTree2D = Quadtree<Edge, ComputeInternalData, dim, bucketsize, depth>; 
using Node = SurfaceTree2D::Node;

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
SurfaceTree2D* SurfaceTree2D_create(
    ScalarType const* const __restrict__ nodes,
    uint32_t const * const __restrict__ edges,
    uint32_t nnodes,
    uint32_t nedges
);

/* @brief Free the memory of the tree
 *
 * @param tree : the pointer to the memory
 */
void SurfaceTree2D_destroy(SurfaceTree2D* tree);

/* @brief Queries of 2D distance to surface
 *
 * @param t : the tree
 * @param q : the query coordinates stored in row-major format :
 *            x0 y0 x1 y1 ...
 * @param d : the resulting distance to surface
 */
void SurfaceTree2D_query(
    SurfaceTree2D * const __restrict__ t,
    ScalarType const * const __restrict__ q,
    ScalarType * const __restrict__ d,
    uint32_t nqueries
);

#ifdef __cplusplus
}
#endif

#endif // __INTERFACE_FWN_2D__
