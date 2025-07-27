#include <cstdlib>
#include <ctime>
#include "quadtree.hpp"

using Scalar = float;
using BBox2 = BBox<Scalar, 2>;
using Vec2 = Vec<Scalar, 2>;

constexpr int bucketsize = 1;
constexpr int depth = 31;
constexpr float beta = 2.0f;

struct OrientedEdge {
    using Scalar = float;

    Vec2 s;  // start
    Vec2 t;  // target
    
    OrientedEdge(const Vec2& _s, const Vec2& _t) : s(_s), t(_t) {}

    Vec2 get_centroid() const { return static_cast<Scalar>(0.5)*(t + s); }

    BBox2 get_bbox() const {
        BBox2 ret;
        for (int i = 0; i < 2; i++){
            ret.min(i) = std::fmin(s[i], t[i]);
        }
        for (int i = 0; i < 2; i++){
            ret.max(i) = std::fmax(s[i], t[i]);
        }
        return ret;
    }
};

struct InternalData {
    using Scalar = float;

    BBox2 bb;  // bounding box
    Vec2  p;   // centroid
    Vec2  wn;  // weighted normal
    float r;   // Size of the subtree, computed as the half-diagonal of the bounding box
    float wl;  // weighted length
};

struct ComputeInternalData {
    using Scalar = OrientedEdge::Scalar;

    InternalData operator()(const OrientedEdge& edge) const {
        InternalData ret;
        ret.bb = edge.get_bbox();
        ret.p = edge.get_centroid();
        ret.r = 0.0f;
        
        Vec2 const dir = edge.t - edge.s;
        ret.wl = dir.norm();

        // Do not make the normalization : n_tilde = l * n
        ret.wn[0] = dir[1];
        ret.wn[1] = -dir[0];
        return ret;
    };

    InternalData operator()(const InternalData& e1, const InternalData& e2) const {
        InternalData ret;
        ret.bb = e1.bb;
        ret.bb.combineBox(e2.bb);
        ret.r = 0.5f * std::hypot(
            ret.bb.max(0) - ret.bb.min(0), 
            ret.bb.max(1) - ret.bb.min(1)
        );

        ret.wl = (e1.wl + e2.wl);
        ret.p = (e1.wl*e1.p + e2.wl*e2.p) / ret.wl;
        ret.wn = (e1.wn + e2.wn);

        return ret;
    }
};

using Tree = Quadtree<OrientedEdge, ComputeInternalData, 2, bucketsize, depth>; 
using Node = Tree::Node;
float fast_wn(const Vec2& q, const Node& node, const Tree& tree){

    Vec2 const vec = node.data.p - q;
    float const sqnorm = vec.sqnorm();
    float const far_field = beta*node.data.r;
    if (sqnorm > far_field*far_field) {
        // Compute wn with the approx
        return (vec.dot(node.data.wn))/(2.0f*M_PIf*sqnorm);
    } else {
        if (!node.n_obj) return 0.0f;

        float val = 0.0f;
        if (node.isleaf()){
            for (size_t i = 0; i < node.obj.size(); i++) {
                const OrientedEdge& edge = tree.objects[std::get<0>(node.obj[i])];

                // First order contribution
                Vec2 const dir = edge.t - edge.s;
                Vec2 const n(dir[1], -dir[0]);

                Vec2 const pi = edge.get_centroid();
                Vec2 const d = pi - q;
                float const sqdist = d.sqnorm();

                val += d.dot(n)/(2.0f*M_PIf*sqdist);
            }
        } else {
            for (size_t i = 0; i < node.children.size(); i++){
                val += fast_wn(q, tree.nodes[node.children[i]], tree);
            }
        }
        return val;
    }

    return 0.0f;
}


int main(int argc, char** argv){
    int n_circle = 100;
    size_t h = 200;
    size_t w = 200;

    if (argc > 1) n_circle = strtol(argv[1], NULL, 10);
    if (argc > 2) h = strtoull(argv[2], NULL, 10);
    if (argc > 3) w = strtoull(argv[3], NULL, 10);

    struct timespec t0, t1;

    srand(42);

    BBox2 bb;
    bb.min(0) = -1.f;
    bb.max(0) = 1.f;
    bb.min(1) = -1.f;
    bb.max(1) = 1.f;

    Tree tree(bb, n_circle);

    // FILE* fdcircle = fopen("circle.csv", "w+");
    std::vector<Vec2> coords;
    coords.reserve(n_circle);
    for (int i = 0; i < n_circle; i++){
        float s = i*2.0f*M_PIf/n_circle;
        Vec2 es(
            std::cos(s), 
            std::sin(s)
        );
        coords.push_back(es);
        // fprintf(fdcircle, "%.5f,%.5f\n", es[0], es[1]);
    }
    // fclose(fdcircle);

    timespec_get(&t0, TIME_UTC);
    for (int i = 0; i < n_circle; i++){
        OrientedEdge edge(coords[i], coords[(i+1)%n_circle]);
        tree.insert(edge);
    }
    timespec_get(&t1, TIME_UTC);
    printf("Time taken to construct a quadtree of %d objects : %f ms\n",
            n_circle,
            (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)*1e-6);

    timespec_get(&t0, TIME_UTC);
    tree.fit_internal_nodes();
    timespec_get(&t1, TIME_UTC);
    printf("Time taken to fit the internal nodes : %f ms\n",
            (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)*1e-6);


    // Sanity check :
    //     length = 2*pi*r = 2*pi 
    //     normal = 0.0, 0.0    (= sum of normals of edges of the circle)
    //     p = 0.0, 0.0         (= sum of positions of edges of the circle)
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

    // FILE* fdout = fopen("query_coords.csv", "w+");
    // FILE* fdwn = fopen("winding_number.csv", "w+");
    // for (size_t id = 0; id < h*w; id++){
    //     fprintf(fdout, "%.5f, %.5f\n", ptest[id][0], ptest[id][1]);
    //     fprintf(fdwn, "%.5f\n", wn[id]);
    // }
    // fclose(fdout);
    // fclose(fdwn);
}
