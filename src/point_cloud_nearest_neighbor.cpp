#include <limits>
#include <algorithm>
#include "point.h"
#include "bbox.h"
#include "point_cloud_nearest_neighbor.h"

Point::Point(const VecT& _p) noexcept : p(_p){ }

VecT Point::get_centroid() const { 
    return p;
}

BBoxT Point::get_bbox() const {
    BBoxT bb(p, p);
    return bb;
}

BBoxT NNTree2D_ComputeInternalData::operator()(const Point& point) const {
    return point.get_bbox();
};

BBoxT NNTree2D_ComputeInternalData::operator()(const BBoxT& e1, const BBoxT& e2) const {
    BBoxT ret = e1;
    ret.combineBox(e2);
    return ret;
}


ScalarType nearest_neighbor(ScalarType sqBestDist, const VecT& q, const uint32_t idx, uint32_t& cur_best, const Node& node, const NNTree2D& tree){
    if (node.isleaf()){
        for (size_t i = 0; i < node.obj.size(); i++){
            uint32_t id = node.obj[i].first;
            if (id == idx) continue; // do not process the query node

            const Point& point = tree.objects[id];
            VecT const p = point.p;

            ScalarType const d2 = (p - q).sqnorm();
            if (sqBestDist > d2) {
                sqBestDist = d2;
                cur_best = id;
            }
        }
    } else {
        for (size_t i = 0; i < NNTree2D::childPerNode; i++){
            size_t child_id = node.children[i];
            if (tree.nodes[child_id].n_obj) {
                const BBoxT& tmpbox = node.data;
                ScalarType sqDist = 0.0;
                for (int j = 0; j < 2; j++){
                    ScalarType const v = q[j];
                    ScalarType const dmax = std::max(v - tmpbox.max(j), static_cast<ScalarType>(0.0));
                    ScalarType const dmin = std::max(tmpbox.min(j) - v, static_cast<ScalarType>(0.0));
                    sqDist += dmax*dmax + dmin*dmin;
                }
                if (sqDist < sqBestDist) {
                    sqBestDist = std::min(sqBestDist, nearest_neighbor(sqBestDist, q, idx, cur_best, tree.nodes[child_id], tree));
                }
            }
        }
    }

    return sqBestDist;
}


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
NNTree2D* NNTree2D_create(
    ScalarType const* const __restrict__ nodes,
    uint32_t nnodes
) {
    BBoxT bb;
    for (uint32_t i = 0; i < nnodes; ++i) {
        bb.combinePoint(VecT(nodes[2*i], nodes[2*i+1]));
    }

    NNTree2D* ret = new NNTree2D(bb, nnodes);
    for (uint32_t i = 0; i < nnodes; ++i) {
        Point const point(VecT(nodes[2*i], nodes[2*i+1]));
        ret->insert(point);
    }
    ret->fit_internal_nodes();

    return ret;
}

/* @brief Free the memory of the SurfaceTree
 *
 * @param tree : the pointer to the memory
 */
void NNTree2D_destroy(NNTree2D* tree){
    delete tree;
}

/* @brief Queries the SurfaceTree for winding number
 *
 * @param t : the tree
 * @param s : the maximum search distance for each query point
 * @param d : the index of the closest node, -1 if not found
 */
void NNTree2D_query(
    NNTree2D * const __restrict__ t,
    ScalarType const * const __restrict__ s,
    uint32_t * const __restrict__ d
){
#pragma omp parallel for
    for (uint32_t i = 0; i < t->objects.size(); ++i){
        std::sqrt(nearest_neighbor(
                s[i]*s[i],
                t->objects[i].p,
                i,
                d[i],
                t->nodes[0], 
                *t
            ));
    }
}

#ifdef __cplusplus
}
#endif
