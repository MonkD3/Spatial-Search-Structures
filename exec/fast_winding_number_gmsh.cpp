#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cwchar>
#include <stdexcept>
#include "quadtree.hpp"
#include "gmsh.h"
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using Scalar = double;
using BBox2 = BBox<Scalar, 2>;
using Vec2 = Vec<Scalar, 2>;

struct MyHash {
    std::size_t operator()(const std::pair<size_t, size_t>& s) const noexcept {
        std::size_t h1 = std::hash<size_t>{}(s.first);
        std::size_t h2 = std::hash<size_t>{}(s.second);
        return h1 ^ (h2 << 1); // or use boost::hash_combine
    }
};

constexpr int bucketsize = 1;
constexpr int depth = 31;
constexpr Scalar beta = 2.0f;
constexpr size_t w = 100; // number of equispaced query points on the x axis
constexpr size_t h = 100; // number of equispaced query points on the y axis

struct OrientedEdge {
    using Scalar = double;

    Vec2 s;  // start
    Vec2 t;  // target
    
    OrientedEdge(const Vec2& _s, const Vec2& _t) : s(_s), t(_t) {}

    Vec2 get_centroid() const { return static_cast<Scalar>(0.5)*(t + s); }

    BBox2 get_bbox() const {
        BBox2 ret;
        ret.min(0) = std::fmin(s[0], t[0]);
        ret.min(1) = std::fmin(s[1], t[1]);
        ret.max(0) = std::fmax(s[0], t[0]);
        ret.max(1) = std::fmax(s[1], t[1]);
        return ret;
    }
};

struct InternalData {
    using Scalar = double;

    BBox2  bb;  // bounding box
    Vec2 p;   // centroid
    Vec2 wn;  // weighted normal
    Scalar r;            // Size of the subtree, computed as the half-diagonal of the bounding box
    Scalar wl;           // weighted length
};

struct ComputeInternalData {
    using Scalar = OrientedEdge::Scalar;

    InternalData operator()(const OrientedEdge& edge) {
        InternalData ret;
        ret.bb = edge.get_bbox();
        ret.r = 0.0f;
        ret.p = edge.get_centroid();
        
        Vec2 dir = edge.t - edge.s;
        ret.wl = dir.norm();

        // Do not make the normalize : n_tilde = l * n
        ret.wn[0] = dir[1];
        ret.wn[1] = -dir[0];
        return ret;
    };

    InternalData operator()(const InternalData& e1, const InternalData& e2) {
        InternalData ret;
        ret.bb = e1.bb;
        ret.bb.combineBox(e2.bb);
        ret.r = 0.5f* std::hypot(
            ret.bb.max(0) - ret.bb.min(0), 
            ret.bb.max(1) - ret.bb.min(1)
        );

        ret.wl = (e1.wl + e2.wl);
        ret.p = (e1.wl*e1.p + e2.wl*e2.p) * (1.0/ret.wl);
        ret.wn = (e1.wn + e2.wn);

        return ret;
    }
};

using Tree = Quadtree<OrientedEdge, ComputeInternalData, bucketsize, depth>; 
using Node = Tree::Node;
Scalar fast_wn(const Vec2& q, const Node& node, const Tree& tree){

    Vec2 vec = node.data.p - q;
    Scalar norm = vec.norm();
    if (norm > beta*node.data.r) {
        // Compute wn with the approx
        return (vec[0]*node.data.wn[0] + vec[1]*node.data.wn[1])/(2.0*M_PI*norm*norm);
    } else {
        if (node.isleaf() && !node.obj.size()) return 0.0;

        Scalar val = 0.0f;
        if (node.isleaf()){
            for (size_t i = 0; i < node.obj.size(); i++) {
                const OrientedEdge& edge = tree.get_leaf_obj(node, i);

                // First order contribution
                Vec2 dir = edge.t - edge.s;
                Vec2 n(dir[1], -dir[0]);

                Vec2 pi = edge.get_centroid();
                Vec2 d = pi - q;
                Scalar dist = d.norm();

                val += (d[0]*n[0] + d[1]*n[1])/(2.0*M_PI*dist*dist);
            }
        } else {
            for (size_t i = 0; i < node.children.size(); i++){
                val += fast_wn(q, tree.nodes[node.children[i]], tree);
            }
        }
        return val;
    }

    return 0.0;
}


int main(int argc, char** argv){
    gmsh::initialize(argc, argv);
    gmsh::option::setNumber("General.Verbosity", 0);

    if (argc < 2) {
        std::runtime_error("Usage :\n\tgmsh_wn <.geo input file> [gmsh options]");
    }

    struct timespec t0, t1;

    // ========================== Load and generate 1D mesh ===================
    timespec_get(&t0, TIME_UTC);
    gmsh::open(argv[1]);
    // gmsh::model::mesh::setOutwardOrientation(1);
    gmsh::model::mesh::generate(2); // only generate edges
    timespec_get(&t1, TIME_UTC);
    printf("Time to load file and generate mesh : %.5f ms\n",
            (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)*1e-6);
    
    // ========================== Extract mesh edges ==========================

    timespec_get(&t0, TIME_UTC);
    std::vector<size_t> nodeTags;
    std::vector<double> nodeCoords;
    std::vector<double> parametricCoord;
    gmsh::model::mesh::getNodes(nodeTags, nodeCoords, parametricCoord, 2, -1, true, false);

    printf("nodeTags.size() : %zu\n", nodeTags.size());
    printf("nodeCoords.size() : %zu\n", nodeCoords.size());

    std::unordered_map<size_t, size_t> tag2idx;
    for (size_t i = 0; i < nodeTags.size(); i++){
        tag2idx.insert({nodeTags[i], i});
    }
    std::vector<int> elemTypes;
    std::vector<std::vector<size_t>> allElemTags;
    std::vector<std::vector<size_t>> allElemNodeTags;
    gmsh::model::mesh::getElements(elemTypes, allElemTags, allElemNodeTags, 2, -1);
    std::vector<size_t>& elemTags = allElemTags[0];
    std::vector<size_t>& elemNodeTags = allElemNodeTags[0];
    printf("elemTags.size() : %zu\n", elemTags.size());
    printf("elemNodeTags.size() : %zu\n", elemNodeTags.size());

    // Compute boundaries
    std::unordered_set<std::pair<size_t, size_t>, MyHash> bnd;
    for (size_t iElem = 0; iElem < elemTags.size(); iElem++){
        size_t s = tag2idx[elemNodeTags[3*iElem]];
        size_t t = tag2idx[elemNodeTags[3*iElem+1]];
        size_t u = tag2idx[elemNodeTags[3*iElem+2]];

        // s-t edge
        if (bnd.contains({t, s})) bnd.erase({t, s}); // check if t-s exists
        else bnd.insert({s, t}); // insert s-t

        if (bnd.contains({u, t})) bnd.erase({u, t}); 
        else bnd.insert({t, u});

        if (bnd.contains({s, u})) bnd.erase({s, u}); 
        else bnd.insert({u, s});
    }

    std::vector<size_t> bnd_edges;
    for (auto edge = bnd.begin(); edge != bnd.end(); ++edge) {
        bnd_edges.push_back(edge->first);
        bnd_edges.push_back(edge->second);
    }

    timespec_get(&t1, TIME_UTC);
    printf("Time to extract edges : %.5f ms\n",
            (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)*1e-6);

    // ========================= Compute bounding box =========================

    timespec_get(&t0, TIME_UTC);
    BBox2 bb;
    for (size_t i = 0; i < bnd_edges.size(); i++){
        size_t iEdge = bnd_edges[i];
        double const x = nodeCoords[3*iEdge];
        double const y = nodeCoords[3*iEdge+1];

        bb.min(0) = std::fmin(bb.min(0), x);
        bb.max(0) = std::fmax(bb.max(0), x);

        bb.min(1) = std::fmin(bb.min(1), y);
        bb.max(1) = std::fmax(bb.max(1), y);
    }
    timespec_get(&t1, TIME_UTC);
    printf("Time to compute bounding box: %.5f ms\n",
            (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)*1e-6);


    // ======================== Output to file ================================
    FILE* fdedges = fopen("boundary.csv", "w+");
    for (size_t i = 0; i < bnd_edges.size()/2; i++){
        size_t tag_s = bnd_edges[2*i];
        size_t tag_t = bnd_edges[2*i+1];
        fprintf(fdedges, 
            "%.5f,%.5f,%.5f,%.5f\n",
            nodeCoords[3*tag_s], nodeCoords[3*tag_s+1],
            nodeCoords[3*tag_t], nodeCoords[3*tag_t+1]
        );
    }
    fclose(fdedges);

    // ========================= Insert all edges into the tree ===============
    timespec_get(&t0, TIME_UTC);

    Tree tree(bb, (int) bnd_edges.size()/2);
    for (size_t i = 0; i < bnd_edges.size()/2; i++){
        size_t tag_s = bnd_edges[2*i];
        size_t tag_t = bnd_edges[2*i+1];
        Vec2 s( nodeCoords[3*tag_s], nodeCoords[3*tag_s+1]);
        Vec2 t( nodeCoords[3*tag_t], nodeCoords[3*tag_t+1]);
        OrientedEdge edge(s, t);
        tree.insert(edge);
    }

    timespec_get(&t1, TIME_UTC);
    printf("Time taken to construct the quadtree : %f ms\n",
            (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)*1e-6);


    // ======================== Compute internal node data ====================
    timespec_get(&t0, TIME_UTC);
    tree.fit_internal_nodes();
    timespec_get(&t1, TIME_UTC);
    printf("Time taken to fit the internal nodes : %f ms\n",
            (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)*1e-6);

    printf(
        "Weighted length : %.5f\n"
        "Weighted normal : (%.5f, %.5f)\n"
        "Weighted centroid : (%.5f, %.5f)\n"
        "Total bb : (%.5f, %.5f) -- (%.5f, %.5f)\n"
        "Total r : %.5f\n", 
        tree.nodes[0].data.wl,
        tree.nodes[0].data.wn[0], tree.nodes[0].data.wn[1],
        tree.nodes[0].data.p[0], tree.nodes[0].data.p[1],
        tree.nodes[0].data.bb.min(0), tree.nodes[0].data.bb.min(1),
        tree.nodes[0].data.bb.max(0), tree.nodes[0].data.bb.max(1),
        tree.nodes[0].data.r
    );


    // ========================= in-out query =================================

    FILE* fdout = fopen("query_coords.csv", "w+");
    FILE* fdwn = fopen("winding_number.csv", "w+");
    std::vector<Vec2> ptest(w*h);
    std::vector<Scalar> wn(w*h);
    for (size_t id = 0; id < h*w; id++){
        size_t i = id % w;
        size_t j = id / w;

        Scalar x = bb.min(0) + ((bb.max(0) - bb.min(0))*i) / h;
        Scalar y = bb.min(1) + ((bb.max(1) - bb.min(1))*j) / w;

        ptest[id] = Vec2(x, y);
    }

    timespec_get(&t0, TIME_UTC);

    #pragma omp parallel for
    for (size_t id = 0; id < h*w; id++){
        wn[id] = fast_wn(ptest[id], tree.nodes[0], tree);
    }
    timespec_get(&t1, TIME_UTC);
    printf("Time taken to compute the winding number of %zu points: %f ms\n", 
            w*h,
            (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)*1e-6);

    for (size_t id = 0; id < h*w; id++){
        fprintf(fdout, "%.5f, %.5f\n", ptest[id][0], ptest[id][1]);
        fprintf(fdwn, "%.5f\n", wn[id]);
    }
    fclose(fdout);
    fclose(fdwn);

    gmsh::finalize();
}
