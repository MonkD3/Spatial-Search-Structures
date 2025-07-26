#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <queue>
#include <stack>
#include <vector>
#include "quadtree.hpp"

struct Planet {
    using Scalar = float;

    Point2<Scalar> center = {}; 
    Point2<Scalar> v = {};
    Scalar mass = 1.0;

    Point2<Scalar> get_centroid() const {
        return center;
    }
    BBox2<Scalar> get_bbox() const {
        BBox2<Scalar> bb;
        bb.min(0) = center[0];
        bb.max(0) = center[0];
        bb.min(1) = center[1];
        bb.max(1) = center[1];
        return bb;
    }
};

struct Cluster {
    using Scalar = Planet::Scalar;
    Point2<Scalar> centroid;
    BBox2<Scalar> bb;
    Scalar total_mass;
};

struct ComputeClusters {
    using Scalar = Planet::Scalar;

    Cluster operator()(const Planet& p) const {
        return {
            .centroid = p.center,
            .bb = p.get_bbox(),
            .total_mass = p.mass
        };
    };

    Cluster operator()(const Cluster& c1, const Cluster& c2) const {
        Cluster ret;
        ret.total_mass = c1.total_mass + c2.total_mass;
        ret.centroid = (1.0/ret.total_mass)*(c1.centroid*c1.total_mass + c2.centroid*c2.total_mass);
        ret.bb = c1.bb;
        ret.bb.combineBox(c2.bb);
        return ret;
    };
};

int main(int argc, char** argv){
    int n = 100;
    constexpr float dt = 0.1f;
    constexpr float G = 6.674e-8; // Should be 6.674e-11
    constexpr float far_field = 0.5;
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

    Quadtree<Planet, ComputeClusters, bucketsize, depth> tree(bb, n);
    std::vector<Planet> planets(n);

    timespec_get(&t0, TIME_UTC);
    Point2<float> zero = {};
    for (int i = 0; i < n; i++){
        Point2<float> p;
        p[0] = static_cast<float>(rand()) / RAND_MAX;
        p[1] = static_cast<float>(rand()) / RAND_MAX;
        float mass = 1.0f; // (static_cast<float>(rand()) / RAND_MAX) * 1e5;
        Planet planet(p, zero, mass);
        planets[i] = planet;
        tree.insert(planet);
    }
    timespec_get(&t1, TIME_UTC);
    printf("Time taken to construct a quadtree of %d planets : %f ms\n",
            n,
            (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)*1e-6);

    timespec_get(&t0, TIME_UTC);
    tree.fit_internal_nodes();
    timespec_get(&t1, TIME_UTC);
    printf("Time taken to fit the internal nodes : %f ms\n",
            (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)*1e-6);

    printf("Total mass : %.10e\nCentroid : (%.5f, %.5f)\n", 
            tree.nodes[0].data.total_mass, 
            tree.nodes[0].data.centroid[0],
            tree.nodes[0].data.centroid[1]
          );

    FILE* fdout = fopen("positions.csv", "w+");

    for (int step = 0; step < 1000; step++){
        std::stack<size_t> stack;
        for (size_t i = 0; i < (size_t) n; i++){
            Planet& p = planets[i];
            stack.push(0);

            while (!stack.empty()){
                size_t node_id = stack.top();
                stack.pop();

                auto node = tree.nodes[node_id];
                bool const is_far = p.center[0] < node.data.bb.min(0) - far_field
                                  || p.center[0] > node.data.bb.max(0) + far_field
                                  || p.center[1] < node.data.bb.min(1) - far_field
                                  || p.center[1] > node.data.bb.max(1) + far_field;

                if (!node.isleaf() && is_far) { // Use the planet cluster as an approximation 
                    Point2<Planet::Scalar> dir(
                        node.data.centroid[0] - p.center[0],
                        node.data.centroid[1] - p.center[1]
                    );
                    Planet::Scalar dist = dir.norm();
                    Point2<Planet::Scalar> accel =  G * node.data.total_mass * dir * (1.0/(dist*dist*dist));
                    p.v += dt*accel;
                }
                else if (node.isleaf()){ // Make the exact sum over all planets
                    for (size_t l = 0; l < node.obj.size(); l++){
                        size_t planet_id = std::get<0>(node.obj[l]);
                        if (i == planet_id) continue;
                        Planet& other = tree.objects[planet_id];

                        Point2<Planet::Scalar> dir(
                            node.data.centroid[0] - p.center[0],
                            node.data.centroid[1] - p.center[1]
                        );
                        Planet::Scalar dist = dir.norm();

                        Point2<Planet::Scalar> accel =  G * other.mass * dir * (1.0/(dist*dist*dist));
                        p.v += dt*accel;
                    }
                }
                else { // Recurse
                    for (size_t j = 0; j < node.children.size(); j++){
                        stack.push(node.children[j]);
                    }
                }
            }
            p.center += dt*p.v;
            fprintf(fdout, "%.10e,%.10e", p.center[0], p.center[1]);
            if (i < (size_t)n-1) fputc(',', fdout);
        }
        fputc('\n', fdout);

        BBox2<Planet::Scalar> new_bbox;
        for (int j = 0; j < n; j++){
            new_bbox.combineBox(planets[j].get_bbox());
        }
        Quadtree<Planet, ComputeClusters, bucketsize, depth> new_tree(new_bbox, n);

        for (int j = 0; j < n; j++){
            new_tree.insert(planets[j]);
        }
        new_tree.fit_internal_nodes();
        tree = std::move(new_tree);
    }

    fclose(fdout);
}
