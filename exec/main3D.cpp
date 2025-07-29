#include <cstdlib>
#include <ctime>
#include "quadtree.hpp"

using Scalar = float;
using BBox3 = BBox<Scalar, 3>;
using Vec3 = Vec<Scalar, 3>;

constexpr int bucketsize = 4;
constexpr int depth = 16;

template<typename Scalar>
struct ObjData {

    BBox3 operator()(const Vec3& obj) const {
        return obj.get_bbox();
    };

    BBox3 operator()(const BBox3& a, const BBox3& b) const {
        BBox3 ret = a;
        ret.combineBox(b);
        return ret;
    }
};

using Tree = Quadtree<Vec3, ObjData<Scalar>, 3, bucketsize, depth>;
using Node = Tree::Node;

void compute_stats(const Tree& tree, size_t node_id, size_t& ninternal, size_t& nleaf, size_t& nemptyleaf) {
    const Node& node = tree.nodes[node_id];
    if (node.isleaf()) {
        nleaf++;
        if (node.obj.size() == 0) nemptyleaf++;
    } else {
        ninternal++;
        for (size_t i = 0; i < Tree::childPerNode; i++){
            compute_stats(tree, node.children[i], ninternal, nleaf, nemptyleaf);
        }
    }
}

int main(int argc, char** argv){
    int n = 100;

    if (argc > 1) n = strtol(argv[1], NULL, 10);

    struct timespec t0, t1;

    srand(42);

    BBox3 bb;
    bb.min(0) = 0.0f;
    bb.min(1) = 0.0f;
    bb.min(2) = 0.0f;
    bb.max(0) = 1.0f;
    bb.max(1) = 1.0f;
    bb.max(2) = 1.0f;

    Tree tree(bb, n);

    timespec_get(&t0, TIME_UTC);
    for (int i = 0; i < n; i++){
        Vec3 p;
        p[0] = static_cast<float>(rand()) / RAND_MAX;
        p[1] = static_cast<float>(rand()) / RAND_MAX;
        p[2] = static_cast<float>(rand()) / RAND_MAX;

        tree.insert(p);
    }
    timespec_get(&t1, TIME_UTC);
    printf("Time taken to construct a quadtree of %d objects : %f ms\n",
            n,
            (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)*1e-6);

    timespec_get(&t0, TIME_UTC);
    tree.fit_internal_nodes();
    timespec_get(&t1, TIME_UTC);
    printf("Time taken to fit the internal nodes : %f ms\n",
            (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)*1e-6);

    printf("Final bbox : [%.5f, %.5f, %.5f] -- [%.5f, %.5f, %.5f]\n", 
        tree.nodes[0].data.min(0),
        tree.nodes[0].data.min(1),
        tree.nodes[0].data.min(2),
        tree.nodes[0].data.max(0),
        tree.nodes[0].data.max(1),
        tree.nodes[0].data.max(2)
    );

    size_t ninternal = 0;
    size_t nleaf = 0;
    size_t nemptyleaf = 0;
    compute_stats(tree, 0, ninternal, nleaf, nemptyleaf);

    printf(
        "tree.nodes.size() : %zu\n"
        "tree.max_depth : %d\n"
        "Number of internal nodes : %zu\n"
        "Number of leaf nodes : %zu\n"
        "Number of empty leaf nodes : %zu\n",
        tree.nodes.size(), tree.max_depth, ninternal, nleaf, nemptyleaf
    );
}
