#include <cstdlib>
#include <ctime>
#include "quadtree.hpp"

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

int main(int argc, char** argv){
    int n = 100;
    constexpr int bucketsize = 1;
    constexpr int depth = 31;

    if (argc > 1) n = strtol(argv[1], NULL, 10);

    struct timespec t0, t1;

    srand(42);

    BBox2 bb;
    bb.min(0) = 0.0f;
    bb.max(0) = 1.0f;
    bb.min(1) = 0.0f;
    bb.max(1) = 1.0f;

    Quadtree<Vec2, ObjData<Scalar>, bucketsize, depth> tree(bb, n);

    timespec_get(&t0, TIME_UTC);
    for (int i = 0; i < n; i++){
        Vec2 p;
        p[0] = static_cast<float>(rand()) / RAND_MAX;
        p[1] = static_cast<float>(rand()) / RAND_MAX;

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

}
