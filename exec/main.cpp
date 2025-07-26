#include <cstdlib>
#include <ctime>
#include "quadtree.hpp"

template<typename Scalar>
struct ObjData {

    BBox2<Scalar> operator()(const Point2<Scalar>& obj) const {
        return obj.get_bbox();
    };

    BBox2<Scalar> operator()(const BBox2<Scalar>& a, const BBox2<Scalar>& b) const {
        BBox2<Scalar> ret = a;
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

    BBox2<float> bb;
    bb.min(0) = 0.0f;
    bb.max(0) = 1.0f;
    bb.min(1) = 0.0f;
    bb.max(1) = 1.0f;

    Quadtree<Point2<float>, ObjData<float>, bucketsize, depth> tree(bb, n);

    timespec_get(&t0, TIME_UTC);
    for (int i = 0; i < n; i++){
        Point2<float> p;
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
