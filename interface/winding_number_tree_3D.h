#ifndef __INTERFACE_FWN_3D__
#define __INTERFACE_FWN_3D__

#include "point.h"
#include "bbox.h"
#include "quadtree.hpp"

using VecT = Vec3d;
using BBoxT = BBox3d;
using ScalarType = double;

constexpr int dim = 3;
constexpr int bucketsize = 1;
constexpr int depth = 21;
constexpr ScalarType beta = 2.0;

struct OrientedTri {
    using Scalar = ScalarType;

    // For fast winding number computation, we do not actually need 
    // the triangle. We only require the normal and centroid.
    VecT c; // centroid
    VecT n; // normal
    
    OrientedTri(const VecT& _s, const VecT& _t, const VecT& _u);
    VecT  get_centroid() const;
    BBoxT get_bbox() const;
};


struct WNOcTreeData {
    using Scalar = OrientedTri::Scalar;

    BBoxT bb;  // bounding box
    VecT  p;   // centroid
    VecT  wn;  // weighted normal
    Scalar r;   // Size of the subtree, computed as the half-diagonal of the bounding box
    Scalar wl;  // weighted length
};

struct ComputeWNOcTreeData {
    using Scalar = OrientedTri::Scalar;

    WNOcTreeData operator()(const OrientedTri& edge) const;
    WNOcTreeData operator()(const WNOcTreeData& t1, const WNOcTreeData& t2) const;
};

using WNOcTree = Quadtree<OrientedTri, ComputeWNOcTreeData, dim, bucketsize, depth>; 
using Node = WNOcTree::Node;

#ifdef __cplusplus
extern "C" {
#endif

/* @brief Create the surface spatial search structure
 *
 * @param nodes : coordinates of the nodes stored in row-major format :
 *                x0 y0 z0 x1 y1 z1 ... 
 * @param tri   : Indices of the nodes forming triangles, stored in row major format :
 *                t00 t01 t02 t10 t11 t12 ...
 * @param nnodes : number of nodes
 * @param ntri   : number of triangles
 */
WNOcTree* WNOcTree_create(
    ScalarType const* const __restrict__ nodes,
    uint32_t const * const __restrict__ tri,
    uint32_t nnodes,
    uint32_t ntri
);

/* @brief Free the memory of the tree
 *
 * @param tree : the pointer to the memory
 */
void WNOcTree_destroy(WNOcTree* tree);

/* @brief Queries of 3D winding number
 *
 * @param t : the tree
 * @param q : the query coordinates stored in row-major format :
 *            x0 y0 z0 x1 y1 z1 ...
 * @param wn : the resulting winding number
 */
void WNOcTree_query(
    WNOcTree * const __restrict__ t,
    ScalarType const * const __restrict__ q,
    ScalarType * const __restrict__ wn,
    uint32_t nqueries
);

#ifdef __cplusplus
}
#endif

#endif // __INTERFACE_FWN_3D__
