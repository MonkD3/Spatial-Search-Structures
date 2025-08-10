#include <limits>
#include <algorithm>
#include "point.h"
#include "bbox.h"
#include "surface_distance_2D.h"

Edge::Edge(const VecT& _s, const VecT& _t) noexcept : s(_s), t(_t) { }

VecT Edge::get_centroid() const { 
    return static_cast<ScalarType>(0.5)*(s+t);
}

BBoxT Edge::get_bbox() const {
    BBoxT bb;
    bb.min(0) = std::min(s[0], t[0]);
    bb.min(1) = std::min(s[1], t[1]);
    bb.max(0) = std::max(s[0], t[0]);
    bb.max(1) = std::max(s[1], t[1]);
    return bb;
}

BBoxT ComputeInternalData::operator()(const Edge& edge) const {
    return edge.get_bbox();
};

BBoxT ComputeInternalData::operator()(const BBoxT& e1, const BBoxT& e2) const {
    BBoxT ret = e1;
    ret.combineBox(e2);
    return ret;
}


ScalarType distance_to_surface(ScalarType sqBestDist, const VecT& q, const Node& node, const SurfaceTree2D& tree){
    if (node.isleaf()){
        for (size_t i = 0; i < node.obj.size(); i++){
            const Edge& edge = tree.objects[node.obj[i].first];

            VecT const a = edge.s;
            VecT const b = edge.t;
            VecT const ab = b - a;
            VecT const ac = q - a;

            // Find parametric position of qp projected on ab
            ScalarType const abac = ab[0]*ac[0] + ab[1]*ac[1];
            ScalarType const abab = ab[0]*ab[0] + ab[1]*ab[1];
            ScalarType param = (abac)/(abab);
            param = std::clamp(param, static_cast<ScalarType>(0.0), static_cast<ScalarType>(1.0));
            VecT const proj = a + param*ab;
            VecT const qp_proj = q - proj;
            ScalarType const d2 = qp_proj[0]*qp_proj[0] + qp_proj[1]*qp_proj[1];
            sqBestDist = std::min(sqBestDist, d2);
        }
    } else {
        for (size_t i = 0; i < SurfaceTree2D::childPerNode; i++){
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
                    sqBestDist = std::min(sqBestDist, distance_to_surface(sqBestDist, q, tree.nodes[child_id], tree));
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
SurfaceTree2D* SurfaceTree2D_create(
    ScalarType const* const __restrict__ nodes,
    uint32_t const * const __restrict__ edges,
    uint32_t nnodes,
    uint32_t nedges
) {
    BBoxT bb;
    for (uint32_t i = 0; i < nnodes; ++i) {
        bb.combinePoint(VecT(nodes[2*i], nodes[2*i+1]));
    }

    SurfaceTree2D* ret = new SurfaceTree2D(bb, nnodes);
    for (uint32_t i = 0; i < nedges; ++i) {
        Edge const edge(
            VecT(nodes[2*edges[2*i]], nodes[2*edges[2*i]+1]),
            VecT(nodes[2*edges[2*i+1]], nodes[2*edges[2*i+1]+1])
        );
        ret->insert(edge);
    }
    ret->fit_internal_nodes();

    return ret;
}

/* @brief Free the memory of the SurfaceTree
 *
 * @param tree : the pointer to the memory
 */
void SurfaceTree2D_destroy(SurfaceTree2D* tree){
    delete tree;
}

/* @brief Queries the SurfaceTree for winding number
 *
 * @param t : the tree
 * @param q : the query coordinates stored in row-major format :
 *            x0 y0 x1 y1 ...
 * @param d : the resulting distance
 */
void SurfaceTree2D_query(
    SurfaceTree2D * const __restrict__ t,
    ScalarType const * const __restrict__ q,
    ScalarType * const __restrict__ d,
    uint32_t nqueries
){
#pragma omp parallel for
    for (uint32_t i = 0; i < nqueries; ++i){
        d[i] = std::sqrt(distance_to_surface(
                std::numeric_limits<ScalarType>::max(),
                VecT(q[2*i], q[2*i+1]), 
                t->nodes[0], 
                *t
            ));
    }
}

#ifdef __cplusplus
}
#endif
