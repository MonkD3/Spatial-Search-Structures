#ifndef __INTERFACE_FWN_2D__
#define __INTERFACE_FWN_2D__

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

struct OrientedEdge {
    using Scalar = ScalarType;

    VecT n;  // normal
    VecT c;  // centroid
    
    OrientedEdge(const VecT& _s, const VecT& _t);
    VecT  get_centroid() const;
    BBoxT get_bbox() const;
};


struct InternalData {
    using Scalar = OrientedEdge::Scalar;

    BBoxT bb;  // bounding box
    VecT  p;   // centroid
    VecT  wn;  // weighted normal
    Scalar r;   // Size of the subtree, computed as the half-diagonal of the bounding box
    Scalar wl;  // weighted length
};

struct ComputeInternalData {
    using Scalar = OrientedEdge::Scalar;

    InternalData operator()(const OrientedEdge& edge) const;
    InternalData operator()(const InternalData& e1, const InternalData& e2) const;
};

using WNQuadTree = Quadtree<OrientedEdge, ComputeInternalData, dim, bucketsize, depth>; 
using Node = WNQuadTree::Node;

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
WNQuadTree* WNQuadTree_create(
    ScalarType const* const __restrict__ nodes,
    uint32_t const * const __restrict__ edges,
    uint32_t nnodes,
    uint32_t nedges
);

/* @brief Free the memory of the tree
 *
 * @param tree : the pointer to the memory
 */
void WNQuadTree_destroy(WNQuadTree* tree);

/* @brief Queries of 2D winding number
 *
 * @param t : the tree
 * @param q : the query coordinates stored in row-major format :
 *            x0 y0 x1 y1 ...
 * @param wn : the resulting winding number
 */
void WNQuadTree_query(
    WNQuadTree * const __restrict__ t,
    ScalarType const * const __restrict__ q,
    ScalarType * const __restrict__ wn,
    uint32_t nqueries
);

#ifdef __cplusplus
}
#endif

#endif // __INTERFACE_FWN_2D__
