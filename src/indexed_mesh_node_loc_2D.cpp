#include <cstdint>
#include <limits>
#include <algorithm>
#include "point.h"
#include "bbox.h"
#include "indexed_mesh_node_loc_2D.h"
#include "predicates.hpp"

Tri::Tri(const uint32_t& _a, const uint32_t& _b, const uint32_t& _c) noexcept 
    : a(_a)
    , b(_b)
    , c(_c) { }

VecT Tri::get_centroid(const std::vector<VecT>& coords) const { 
    return static_cast<ScalarType>(0.33333333333333333333)*(
            coords[a] + coords[b] + coords[c]
        );
}

BBoxT Tri::get_bbox(const std::vector<VecT>& coords) const {
    BBoxT bb;
    const VecT& pa = coords[a];
    const VecT& pb = coords[b];
    const VecT& pc = coords[c];
    bb.min(0) = std::min(std::min(pa[0], pb[0]), pc[0]);
    bb.min(1) = std::min(std::min(pa[1], pb[1]), pc[1]);
    bb.max(0) = std::max(std::max(pa[0], pb[0]), pc[0]);
    bb.max(1) = std::max(std::max(pa[1], pb[1]), pc[1]);
    return bb;
}

BBoxT ComputeIndexedMeshTree2DInternalData::operator()(const Tri& tri) const {
    return tri.get_bbox(coords);
};

BBoxT ComputeIndexedMeshTree2DInternalData::operator()(const BBoxT& e1, const BBoxT& e2) const {
    BBoxT ret = e1;
    ret.combineBox(e2);
    return ret;
}


int32_t tri_node_collision(const VecT& q, const Node& node, const IndexedMeshTree2D& tree){
    if (node.isleaf()){
        for (size_t i = 0; i < node.obj.size(); i++){
            const Tri& tri = tree.objects[node.obj[i].first];

            const VecT& pa = tree.coords[tri.a];
            const VecT& pb = tree.coords[tri.b];
            const VecT& pc = tree.coords[tri.c];

            ScalarType left_of_ab = predicates::adaptive::orient2d(pa.coord.data(), pb.coord.data(), q.coord.data());
            ScalarType left_of_bc = predicates::adaptive::orient2d(pb.coord.data(), pc.coord.data(), q.coord.data());
            ScalarType left_of_ca = predicates::adaptive::orient2d(pc.coord.data(), pa.coord.data(), q.coord.data());

            if ((left_of_ab >= 0.0 && left_of_bc >= 0.0 && left_of_ca >= 0.0)){
                return static_cast<int32_t>(node.obj[i].first);
            }
        }
    } else {
        for (size_t i = 0; i < IndexedMeshTree2D::childPerNode; i++){
            size_t child_id = node.children[i];
            if (tree.nodes[child_id].n_obj) {
                const BBoxT& tmpbox = node.data;
                bool in_bbox = true;
                for (int j = 0; j < 2; j++){
                    in_bbox &= q[j] <= tmpbox.max(j);
                    in_bbox &= q[j] >= tmpbox.min(j);
                }
                if (in_bbox) {
                    int32_t tri_id = tri_node_collision(q, tree.nodes[child_id], tree);
                    if (tri_id >= 0) return tri_id;
                }
            }
        }
    }

    return -1;
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
IndexedMeshTree2D* IndexedMeshTree2D_create(
    ScalarType const* const __restrict__ nodes,
    uint32_t const * const __restrict__ tri,
    uint32_t nnodes,
    uint32_t ntri
) {
    BBoxT bb;
    for (uint32_t i = 0; i < nnodes; ++i) {
        bb.combinePoint(VecT(nodes[2*i], nodes[2*i+1]));
    }

    IndexedMeshTree2D* ret = new IndexedMeshTree2D(bb, ntri, nnodes);

    for (uint32_t i = 0; i < nnodes; ++i) {
        ret->insert_coord(VecT(nodes[2*i], nodes[2*i+1]));
    }

    for (uint32_t i = 0; i < ntri; ++i) {
        Tri const t(tri[3*i], tri[3*i+1], tri[3*i+2]);
        ret->insert_object(t);
    }
    ret->fit_internal_nodes();

    return ret;
}

/* @brief Free the memory of the SurfaceTree
 *
 * @param tree : the pointer to the memory
 */
void IndexedMeshTree2D_destroy(IndexedMeshTree2D* tree){
    delete tree;
}

/* @brief Queries the SurfaceTree for winding number
 *
 * @param t : the tree
 * @param q : the query coordinates stored in row-major format :
 *            x0 y0 x1 y1 ...
 * @param d : the resulting distance
 */
void IndexedMeshTree2D_query(
    IndexedMeshTree2D * const __restrict__ t,
    ScalarType const * const __restrict__ q,
    int32_t * const __restrict__ tri_id,
    uint32_t nqueries
){
#pragma omp parallel for
    for (uint32_t i = 0; i < nqueries; ++i){
        tri_id[i] = tri_node_collision(
                VecT(q[2*i], q[2*i+1]), 
                t->nodes[0], 
                *t
            );
    }
}

#ifdef __cplusplus
}
#endif
