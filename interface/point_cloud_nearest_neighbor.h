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

struct Point {
    using Scalar = ScalarType;

    VecT p; 
    
    Point(const VecT& _p) noexcept;
    VecT  get_centroid() const;
    BBoxT get_bbox() const;
};

struct NNTree2D_ComputeInternalData {
    using Scalar = Point::Scalar;

    BBoxT operator()(const Point& edge) const;
    BBoxT operator()(const BBoxT& e1, const BBoxT& e2) const;
};

using NNTree2D = Quadtree<Point, NNTree2D_ComputeInternalData, dim, bucketsize, depth>; 
using Node = NNTree2D::Node;

#ifdef __cplusplus
extern "C" {
#endif

/* @brief Create the spatial search structure
 *
 * @param nodes : [in] coordinates of the nodes stored in row-major format :
 *                x0 y0 x1 y1 ... 
 * @param nnodes : [in] number of nodes
 */
NNTree2D* NNTree2D_create(
    ScalarType const* const __restrict__ nodes,
    uint32_t nnodes
);

/* @brief Free the memory of the tree
 *
 * @param tree : the pointer to the memory
 */
void NNTree2D_destroy(NNTree2D* tree);

/* @brief Queries of nearest neighbor for each node
 *
 * @param t : [in] the tree
 * @param s : [in] the maximum search distance for each query point (size = nnodes)
 * @param d : [out] the index of the closest node, id of the node if no neighbor found (distance = 0)
 */
void NNTree2D_query(
    NNTree2D * const __restrict__ t,
    ScalarType const * const __restrict__ s,
    uint32_t * const __restrict__ d
);

#ifdef __cplusplus
}
#endif

#endif // __INTERFACE_FWN_2D__
