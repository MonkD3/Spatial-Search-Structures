#include <array>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <immintrin.h>

#include "point.hpp"
#include "bbox.hpp"

/**
 *  class QuadtreeNode<ObjT, DataT>
 *  @brief : Represent an internal quadtree node
 *
 *  @attribute children : an std::array of size 8 containing pointers to 
 *                        the children of the node
 *  @attribute obj : the object stored at this node
 *     e.g. : points, triangles, spheres
 *  @attribute data : additional data about the object.
 *                    It can also store internal node data.
 *     e.g. : bounding box, depth, barnes-hut approximations data
 */
template <typename ObjT, typename DataT>
struct QuadtreeNode {
    std::array<size_t, 4> children = {};   // Contains indices of the nodes in the global node vector
    std::vector<std::pair<size_t, uint64_t>> obj = {}; // contains pairs (indices, morton) of the objects in the global object vector
    DataT data = {};
    size_t n_obj = 0; // number of objects in the subtree

    bool isleaf() const { return !children[0]; };
};

/**
 *  class Quadtree<ObjT, DataT>
 *  @brief : a quadtree
 *
 *  @template ObjT : the type of object to be indexed. It should implement the 
 *      following functions :
 *          - get_centroid() : return a point corresponding to the centroid of the
 *                             object
 *          - get_bbox() : return the bounding box of the object
 *      and have the following typename :
 *          - Scalar
 *
 *  @template ObjDataF : A functor to compute the data (of user-defined type DataT)
 *                       associated to the objects 
 *                       and internal nodes. It should overload operator() twice :
 *          - DataT operator()(const ObjT& obj) : 
 *                Return the data associated to the object.
 *          - DataT operator()(const DataT& d1, const DataT& d2) : 
 *                Return the data associated to the combination of multiple nodes.
 *                Allow to compute data of the internal nodes.
 *  @template BUCKETSIZE : The number of objects to store on leaf nodes
 *  @template MAX_DEPTH : The maximum depth of the tree
 *
 */
template <typename ObjT, typename ObjDataF, int BUCKETSIZE=1, int MAX_DEPTH=31>
struct Quadtree {
    using Tree = Quadtree<ObjT, ObjDataF, BUCKETSIZE, MAX_DEPTH>;
    using Scalar = typename ObjT::Scalar;
    using DataT = std::invoke_result<ObjDataF, ObjT>::type;
    using Node = QuadtreeNode<ObjT, DataT>;

    BBox<Scalar, 2> bb; 
    std::vector<Node> nodes;
    std::vector<ObjT> objects;
    ObjDataF functor;

    Vec<Scalar, 2> mortonFactor; // precomputed factor for correct Morton code computation

    // Initialize an empty quadtree 
    Quadtree(const BBox<Scalar, 2>& _bb) : bb(_bb) {
        nodes.resize(1); // Make space for the root node

        // set morton code factor
        constexpr uint64_t imax = 1ULL << 32;
        Scalar const xw = (bb.max(0) - bb.min(0));
        Scalar const yw = (bb.max(1) - bb.min(1));
        Scalar fx = imax / xw;
        Scalar fy = imax / yw;
        if (xw*fx >= static_cast<Scalar>(imax)) mortonFactor[0] = std::nextafter(fx, static_cast<Scalar>(0));
        if (yw*fy >= static_cast<Scalar>(imax)) mortonFactor[1] = std::nextafter(fy, static_cast<Scalar>(0));
    };

    // Initialize an empty quadtree but preallocate memory for n objects
    Quadtree(const BBox<Scalar, 2>& _bb, int n) : Quadtree<ObjT, ObjDataF, BUCKETSIZE, MAX_DEPTH>(_bb) {
        objects.reserve(n);
        nodes.reserve(n/BUCKETSIZE);
    };

    Node* head() const {
        return &nodes[0];
    };
    const ObjT& get_obj(size_t idx) const {
        return objects[idx];
    }
    const ObjT& get_leaf_obj(const Node& leaf, size_t idx) const {
        return objects[leaf.obj[idx].first];
    }

    void split(size_t const node_id, int const depth){

        // Allocate 4 new nodes in the tree and set a pointer from the children
        // to these node
        nodes.resize(nodes.size() + 4);
        Node& node = nodes[node_id];
        for (size_t i = 0; i < 4; i++){
            size_t id = nodes.size() - 4 + i;
            node.children[i] = id;
        }

        // Get the current morton code of the objects in the node and 
        // look at where we need to assign them
        uint32_t const bitID = 62 - 2*depth;
        for (size_t i = 0; i < node.obj.size(); i++){
            const std::pair<size_t, uint64_t>& cur = node.obj[i];
            uint64_t const morton = cur.second;
            uint32_t const child_id = (morton >> bitID) & 3;
            Node& child = nodes[node.children[child_id]];
            child.obj.push_back(cur);
            child.n_obj++;
        }

        node.obj.clear(); 
    }

    void add(size_t const node_id, int depth, std::pair<size_t, uint64_t>& obj) {
        Node& node = nodes[node_id];
        if (node.isleaf()){
            // Add the object in this node 
            // WARNING : if BUCKETSIZE objects share the same morton code 
            // then the tree will automatically go down to MAX_DEPTH.
            // When BUCKETSIZE is large this will not be a problem
            if (node.obj.size() < BUCKETSIZE || depth >= MAX_DEPTH) {
                node.n_obj++;
                node.obj.push_back(obj);
            } else {
                split(node_id, depth);           // Split the current node into 4
                add(node_id, depth, obj); // Retry adding on this node
            }
        } else {
            // Take only the two first bits of the morton code
            uint64_t morton = obj.second;
            uint32_t const child_id = (morton >> (62 - 2*depth)) & 3;

            // Recurse on next child
            add(node.children[child_id], depth+1, obj);
        }
    }

    void insert(const ObjT& obj) {
        // Start with the global bbox
        Vec<Scalar, 2> const centroid = obj.get_centroid();

        if (centroid[0] < bb.min(0) || centroid[0] > bb.max(0) || 
                centroid[1] < bb.min(1) || centroid[1] > bb.max(1)) {
            throw std::runtime_error("Centroid of inserted element is outside of the tree's bounding box");
        }

        objects.push_back(obj);
        size_t const obj_id = objects.size()-1;

        uint32_t const x = (centroid[0] - bb.min(0)) * mortonFactor[0];
        uint32_t const y = (centroid[1] - bb.min(1)) * mortonFactor[1];
        uint64_t const morton = _pdep_u64(x, UINT64_C(0xAAAAAAAAAAAAAAAA)) |  _pdep_u64(y, UINT64_C(0x5555555555555555));   

        std::pair<size_t, uint64_t> obj_pair(obj_id, morton);
        add(0, 0, obj_pair);
    }

    // Compute the data associated to each internal nodes
    // It does so recursively in the following manner :
    //   - First compute the data associated to each child of the node
    //     -> if the child is internal, recurse 
    //     -> if the child is a leaf : compute 
    void fit_internal_nodes() { 
        fit_internal_nodes_helper(0); // start on the root
    }
    void fit_internal_nodes_helper(size_t const node_id){ 
        Node& node = nodes[node_id];
        if (node.isleaf() && node.obj.size()) {
            // Combine the data of all objects on this leaf
            std::pair<size_t, uint64_t> o = node.obj[0];
            DataT data = functor(objects[o.first]);

            for (size_t i = 1; i < node.obj.size(); i++){
                o = node.obj[i];
                DataT data_other = functor(objects[o.first]);
                data = functor(data, data_other);
            }

            node.data = data;
        } else {
            bool data_is_init = false; 
            for (size_t i = 0; i < 4; i++){
                if (nodes[node.children[i]].n_obj) {
                    fit_internal_nodes_helper(node.children[i]);
                    if (!data_is_init) {
                        node.data = nodes[node.children[i]].data;
                        data_is_init = true;
                    } else {
                        node.data = functor(node.data, nodes[node.children[i]].data);
                    }
                }
            }
        }
    }
};
