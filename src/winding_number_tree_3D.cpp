#include "point.h"
#include "bbox.h"
#include "winding_number_tree_3D.h"

OrientedTri::OrientedTri(const VecT& _s, const VecT& _t, const VecT& _u) {
    VecT const a = _t - _s;
    VecT const b = _u - _s;
    // First order contribution
    n = VecT(
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    );
    c = static_cast<Scalar>(0.333333333333333333333333)*(_t + _s + _u); 
}

VecT OrientedTri::get_centroid() const { 
    return c; 
}

BBoxT OrientedTri::get_bbox() const {
    return c.get_bbox();
}

WNOcTreeData ComputeWNOcTreeData::operator()(const OrientedTri& tri) const {
    WNOcTreeData ret;
    ret.bb = tri.get_bbox();
    ret.p = tri.get_centroid();
    ret.r = static_cast<Scalar>(0);
    ret.wn = tri.n;
    ret.wl = static_cast<ScalarType>(0.5) * ret.wn.norm();
    return ret;
};

WNOcTreeData ComputeWNOcTreeData::operator()(const WNOcTreeData& e1, const WNOcTreeData& e2) const {
    WNOcTreeData ret;
    ret.bb = e1.bb;
    ret.bb.combineBox(e2.bb);
    ret.r = static_cast<Scalar>(0.5) * (ret.bb.pmax - ret.bb.pmin).norm();
    ret.wl = (e1.wl + e2.wl);
    ret.p = (e1.wl*e1.p + e2.wl*e2.p) / ret.wl;
    ret.wn = (e1.wn + e2.wn);

    return ret;
}


ScalarType fast_wn(const VecT& q, const Node& node, const WNOcTree& tree){

    ScalarType val = 0.0;

    if (node.isleaf()) {
        for (size_t i = 0; i < node.obj.size(); i++) {
            const OrientedTri& tri = tree.objects[node.obj[i].first];

            VecT const n = tri.n;
            VecT const pi = tri.get_centroid();
            VecT const d = pi - q;
            ScalarType const sqdist = d.sqnorm();

            val += d.dot(n)/(4.0*M_PI*sqdist);
        }
    } else {
        VecT const vec = node.data.p - q;
        ScalarType const norm = vec.norm();
        ScalarType const far_field = beta*node.data.r;
        if (norm > far_field) {
            // Compute wn with the approx
            return (vec.dot(node.data.wn))/(4.0*M_PI*norm*norm);
        } else {
            for (size_t i = 0; i < WNOcTree::childPerNode; i++){
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
WNOcTree* WNOcTree_create(
    ScalarType const* const __restrict__ nodes,
    uint32_t const * const __restrict__ tri,
    uint32_t nnodes,
    uint32_t ntri
) {
    BBoxT bb;
    for (uint32_t i = 0; i < nnodes; ++i) {
        bb.combinePoint(VecT(nodes[dim*i], nodes[dim*i+1], nodes[dim*i+2]));
    }

    WNOcTree* ret = new WNOcTree(bb, nnodes);
    for (uint32_t i = 0; i < ntri; ++i) {
        OrientedTri const t(
            VecT(nodes[dim*tri[dim*i  ]], nodes[dim*tri[dim*i  ]+1], nodes[dim*tri[dim*i  ]+2]),
            VecT(nodes[dim*tri[dim*i+1]], nodes[dim*tri[dim*i+1]+1], nodes[dim*tri[dim*i+1]+2]),
            VecT(nodes[dim*tri[dim*i+2]], nodes[dim*tri[dim*i+2]+1], nodes[dim*tri[dim*i+2]+2])
        );
        ret->insert(t);
    }
    ret->fit_internal_nodes();

    printf(
        "Weighted area : %.5f\n"
        "Weighted normal : (%.5f, %.5f)\n"
        "Weighted centroid : (%.5f, %.5f)\n"
        "Total bb : (%.5f, %.5f) -- (%.5f, %.5f)\n"
        "Total r : %.5f\n", 
        ret->nodes[0].data.wl,
        ret->nodes[0].data.wn[0], ret->nodes[0].data.wn[1],
        ret->nodes[0].data.p[0], ret->nodes[0].data.p[1],
        ret->nodes[0].data.bb.min(0), ret->nodes[0].data.bb.min(1),
        ret->nodes[0].data.bb.max(0), ret->nodes[0].data.bb.max(1),
        ret->nodes[0].data.r
    );

    return ret;
}

/* @brief Free the memory of the SurfaceTree
 *
 * @param tree : the pointer to the memory
 */
void WNOcTree_destroy(WNOcTree* tree){
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
void WNOcTree_query(
    WNOcTree * const __restrict__ t,
    ScalarType const * const __restrict__ q,
    ScalarType * const __restrict__ wn,
    uint32_t nqueries
){
#pragma omp parallel for
    for (uint32_t i = 0; i < nqueries; ++i){
        wn[i] = fast_wn(
                VecT(q[dim*i], q[dim*i+1], q[dim*i+2]), 
                t->nodes[0], 
                *t
            );
    }
}

#ifdef __cplusplus
}
#endif
