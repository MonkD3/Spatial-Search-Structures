#include "point.h"
#include "bbox.h"
#include "winding_number_tree_2D.h"

OrientedEdge::OrientedEdge(const VecT& _s, const VecT& _t) {
    VecT const dir = _t - _s;
    n = VecT(dir[1], -dir[0]);
    c = static_cast<ScalarType>(0.5)*(_s + _t);
}

VecT OrientedEdge::get_centroid() const { 
    return c;
}

BBoxT OrientedEdge::get_bbox() const {
    return c.get_bbox();
}

InternalData ComputeInternalData::operator()(const OrientedEdge& edge) const {
    InternalData ret;
    ret.bb = edge.get_bbox();
    ret.p = edge.get_centroid();
    ret.r = static_cast<Scalar>(0);
    
    ret.wl = edge.n.norm();
    ret.wn = edge.n;
    return ret;
};

InternalData ComputeInternalData::operator()(const InternalData& e1, const InternalData& e2) const {
    InternalData ret;
    ret.bb = e1.bb;
    ret.bb.combineBox(e2.bb);
    ret.r = static_cast<Scalar>(0.5) * (ret.bb.pmax - ret.bb.pmin).norm();
    ret.wl = (e1.wl + e2.wl);
    ret.p = (e1.wl*e1.p + e2.wl*e2.p) / ret.wl;
    ret.wn = (e1.wn + e2.wn);

    return ret;
}


ScalarType fast_wn(const VecT& q, const Node& node, const WNQuadTree& tree){

    ScalarType val = 0.0;
    if (node.isleaf()) {
        for (size_t i = 0; i < node.obj.size(); i++) {
            const OrientedEdge& edge = tree.objects[std::get<0>(node.obj[i])];

            // First order contribution
            VecT const n = edge.n;

            VecT const pi = edge.get_centroid();
            VecT const d = pi - q;
            ScalarType const sqdist = d.sqnorm();

            val += d.dot(n)/(2.0*M_PI*sqdist);
        }
    } else {
        VecT const vec = node.data.p - q;
        ScalarType const sqnorm = vec.sqnorm();
        ScalarType const far_field = beta*node.data.r;
        if (sqnorm > far_field*far_field) {
            // Compute wn with the approx
            return (vec.dot(node.data.wn))/(2.0*M_PI*sqnorm);
        } else {
            for (size_t i = 0; i < WNQuadTree::childPerNode; i++){
                if (tree.nodes[node.children[i]].n_obj)
                    val += fast_wn(q, tree.nodes[node.children[i]], tree);
            }
        }
    }

    return val;
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
WNQuadTree* WNQuadTree_create(
    ScalarType const* const __restrict__ nodes,
    uint32_t const * const __restrict__ edges,
    uint32_t nnodes,
    uint32_t nedges
) {
    BBoxT bb;
    for (uint32_t i = 0; i < nnodes; ++i) {
        bb.combinePoint(VecT(nodes[2*i], nodes[2*i+1]));
    }

    WNQuadTree* ret = new WNQuadTree(bb, nnodes);
    for (uint32_t i = 0; i < nedges; ++i) {
        OrientedEdge const edge(
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
void WNQuadTree_destroy(WNQuadTree* tree){
    delete tree;
}

/* @brief Queries the SurfaceTree for winding number
 *
 * @param t : the tree
 * @param q : the query coordinates stored in row-major format :
 *            x0 y0 x1 y1 ...
 * @param d : the resulting distance
 * @param cp : the closest point on the surface
 * @param eid : the id of the closest edge
 */
void WNQuadTree_query(
    WNQuadTree * const __restrict__ t,
    ScalarType const * const __restrict__ q,
    ScalarType * const __restrict__ wn,
    uint32_t nqueries
){
#pragma omp parallel for
    for (uint32_t i = 0; i < nqueries; ++i){
        wn[i] = fast_wn(
                VecT(q[2*i], q[2*i+1]), 
                t->nodes[0], 
                *t
            );
    }
}

#ifdef __cplusplus
}
#endif
