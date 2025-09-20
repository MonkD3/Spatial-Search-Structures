#ifndef __INTERFACE_IMNL_2D__
#define __INTERFACE_IMNL_2D__

#include "point.h"
#include "bbox.h"
#include "indexed_quadtree.hpp"

using VecT = Vec2d;
using BBoxT = BBox2d;
using ScalarType = double;

constexpr int dim = 2;
constexpr int bucketsize = 4;
constexpr int depth = 12;

struct Tri {
    using Scalar = ScalarType;

    uint32_t a;  
    uint32_t b; 
    uint32_t c;
    
    Tri(const uint32_t& _a, const uint32_t& _b, const uint32_t& _c) noexcept;
    VecT  get_centroid(const std::vector<VecT>& coords) const;
    BBoxT get_bbox(const std::vector<VecT>& coords)     const;
};

struct ComputeIndexedMeshTree2DInternalData {
    using Scalar = Tri::Scalar;
    const std::vector<VecT>& coords;

    BBoxT operator()(const Tri& tri) const;
    BBoxT operator()(const BBoxT& e1, const BBoxT& e2) const;
};

using IndexedMeshTree2D = IndexedQuadTree<Tri, ComputeIndexedMeshTree2DInternalData, dim, bucketsize, depth>; 
using Node = IndexedMeshTree2D::Node;

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
IndexedMeshTree2D* IndexedMeshTree2D_create(
    ScalarType const* const __restrict__ nodes,
    uint32_t const * const __restrict__ tri,
    uint32_t nnodes,
    uint32_t ntri
);

/* @brief Free the memory of the tree
 *
 * @param tree : the pointer to the memory
 */
void IndexedMeshTree2D_destroy(IndexedMeshTree2D* tree);

/* @brief Queries of 2D distance to surface
 *
 * @param t : the tree
 * @param q : the query coordinates stored in row-major format :
 *            x0 y0 x1 y1 ...
 * @param d : the resulting triangle each coordinates is located in.
 *            If the node is not in a triangle, return -1
 */
void IndexedMeshTree2D_query(
    IndexedMeshTree2D * const __restrict__ t,
    ScalarType const * const __restrict__ q,
    int32_t * const __restrict__ tri_id,
    uint32_t nqueries
);

#ifdef __cplusplus
}
#endif

#endif // __INTERFACE_IMNL_2D__
