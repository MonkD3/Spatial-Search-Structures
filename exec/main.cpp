#include <cstdlib>
#include <ctime>
#include "quadtree.hpp"

constexpr int bucketsize = 1;
constexpr int depth = 31;

using Scalar = float;
using BBox2 = BBox<Scalar, 2>;
using Vec2 = Vec<Scalar, 2>;


template<typename Scalar>
struct ObjData {

    BBox2 operator()(const Vec2& obj) const {
        return obj.get_bbox();
    };

    BBox2 operator()(const BBox2& a, const BBox2& b) const {
        BBox2 ret = a;
        ret.combineBox(b);
        return ret;
    }
};

using Tree = Quadtree<Vec2, ObjData<Scalar>, 2, bucketsize, depth>;
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

    BBox2 bb;
    bb.min(0) = -1.0f;
    bb.max(0) =  1.0f;
    bb.min(1) = -1.0f;
    bb.max(1) =  1.0f;

    Tree tree(bb, n);

    std::vector<Vec2> points(n);
    for (int i = 0; i < n; i++){
        float s = i*2.0f*M_PIf/n;
        Vec2 es(
            std::cos(s), 
            std::sin(s)
        );
        points[i] = es;
    }

    timespec_get(&t0, TIME_UTC);
    for (int i = 0; i < n; i++){
        tree.insert(points[i]);
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

    size_t ninternal = 0;
    size_t nleaf = 0;
    size_t nemptyleaf = 0;
    compute_stats(tree, 0, ninternal, nleaf, nemptyleaf);

    printf(
        "tree.nodes.size() : %zu\n"
        "Number of internal nodes : %zu\n"
        "Number of leaf nodes : %zu\n"
        "Number of empty leaf nodes : %zu\n",
        tree.nodes.size(), ninternal, nleaf, nemptyleaf
    );

}
