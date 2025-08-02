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
template <typename ObjT, typename DataT, int dim>
struct QuadtreeNode {
    static constexpr int childPerNode = 1 << dim;
    std::array<size_t, childPerNode> children = {};   // Contains indices of the nodes in the global node vector
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
template <typename ObjT, typename ObjDataF, int dim, int BUCKETSIZE=1, int MAX_DEPTH=31>
struct Quadtree {
    static_assert(dim == 2 || dim == 3, "Tree dimension must be 2 or 3");

    using Scalar = typename ObjT::Scalar; // Type of scalars
    using DataT  = std::invoke_result<ObjDataF, ObjT>::type;  // Type of the data stored in the inner nodes

    // Shorthands
    using Tree   = Quadtree<ObjT, ObjDataF, BUCKETSIZE, MAX_DEPTH>;
    using Node   = QuadtreeNode<ObjT, DataT, dim>;
    using VecT   = Vec<Scalar, dim>;
    using BBoxT  = BBox<Scalar, dim>;

    static constexpr int      childPerNode = 1 << dim;
    static constexpr uint64_t childIDMask  = 2*dim - 1;

    BBoxT bb; 
    std::vector<Node> nodes;
    std::vector<ObjT> objects;
    VecT mortonFactor; // precomputed factor for correct Morton code computation
    int max_depth = 0;

    // Initialize an empty quadtree 
    Quadtree(const BBoxT& _bb) : bb(_bb) {
        nodes.resize(1); // Make space for the root node

        // set morton code factor
        constexpr uint64_t imax = 1ULL << (64 / dim);
        VecT const range = bb.pmax - bb.pmin;
        for (int i = 0; i < dim; ++i){
            Scalar f = imax / range[i];
            if (range[i]*f >= static_cast<Scalar>(imax)) f = std::nextafter(f, static_cast<Scalar>(0));
            mortonFactor[i] = f;
        }
    };

    // Initialize an empty quadtree but preallocate memory for n objects
    Quadtree(const BBoxT& _bb, int n) : Quadtree<ObjT, ObjDataF, dim, BUCKETSIZE, MAX_DEPTH>(_bb) {
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
        nodes.resize(nodes.size() + childPerNode);
        Node& node = nodes[node_id];
        for (size_t i = 0; i < childPerNode; i++){
            size_t id = nodes.size() - childPerNode + i;
            node.children[i] = id;
        }

        // Get the current morton code of the objects in the node and 
        // look at where we need to assign them
        uint32_t const bitID = (64 - dim) - dim*depth;
        for (size_t i = 0; i < node.obj.size(); i++){
            const std::pair<size_t, uint64_t>& cur = node.obj[i];
            uint64_t const morton = cur.second;
            uint32_t const child_id = (morton >> bitID) & childIDMask;
            Node& child = nodes[node.children[child_id]];
            child.obj.push_back(cur);
            child.n_obj++;
        }

        node.obj.clear(); 
    }

    void add(size_t const node_id, int depth, std::pair<size_t, uint64_t>& obj) {
        Node& node = nodes[node_id];
        max_depth = std::max(depth, max_depth);
        if (node.isleaf()){
            // Add the object in this node 
            // WARNING : if BUCKETSIZE objects share the same morton code 
            // then the tree will automatically go down to MAX_DEPTH.
            if (node.obj.size() < BUCKETSIZE || depth >= MAX_DEPTH) {
                node.n_obj++;
                node.obj.push_back(obj);
                return;
            } else {
                split(node_id, depth); // Split the current node
            }
        }  
        
        // Take only the two first bits of the morton code
        uint64_t morton = obj.second;
        uint32_t const child_id = (morton >> ((64 - dim) - dim*depth)) & childIDMask;

        // Recurse on next child
        add(nodes[node_id].children[child_id], depth+1, obj);
    }

    void insert(const ObjT& obj) {
        // Start with the global bbox
        VecT const centroid = obj.get_centroid();

        // Check if the object is inside the bounding box of the tree
        bool oobb = false;
        for (int i = 0; i < dim; ++i){
            oobb |= centroid[i] < bb.min(i) || centroid[i] > bb.max(i);
        }
        if (oobb) {
            throw std::runtime_error("Centroid of inserted element is outside of the tree's bounding box");
        }

        objects.push_back(obj);
        size_t const obj_id = objects.size()-1;

        VecT const coord_float = cMult(centroid - bb.pmin, mortonFactor);

        uint64_t morton = 0;
        if constexpr (dim == 2) {
            uint32_t const x = coord_float[0];
            uint32_t const y = coord_float[1];
            morton = _pdep_u64(x, UINT64_C(0b1010101010101010101010101010101010101010101010101010101010101010)) 
                   | _pdep_u64(y, UINT64_C(0b0101010101010101010101010101010101010101010101010101010101010101));  
        } 
        if constexpr (dim == 3) {
            uint32_t const x = coord_float[0];
            uint32_t const y = coord_float[1];
            uint32_t const z = coord_float[2];
            morton = _pdep_u64(x, UINT64_C(0b1001001001001001001001001001001001001001001001001001001001001001)) 
                   | _pdep_u64(y, UINT64_C(0b0100100100100100100100100100100100100100100100100100100100100100))  
                   | _pdep_u64(z, UINT64_C(0b0010010010010010010010010010010010010010010010010010010010010010));  
        }

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
        ObjDataF compute_functor;
        
        if (node.isleaf() && node.obj.size()) {
            // Combine the data of all objects on this leaf
            std::pair<size_t, uint64_t> o = node.obj[0];
            DataT data = compute_functor(objects[o.first]);

            for (size_t i = 1; i < node.obj.size(); i++){
                o = node.obj[i];
                DataT data_other = compute_functor(objects[o.first]);
                data = compute_functor(data, data_other);
            }

            node.data = data;
        } else {
            bool data_is_init = false; 
            for (size_t i = 0; i < childPerNode; i++){
                if (nodes[node.children[i]].n_obj) {
                    fit_internal_nodes_helper(node.children[i]);
                    if (!data_is_init) {
                        node.data = nodes[node.children[i]].data;
                        data_is_init = true;
                    } else {
                        node.data = compute_functor(node.data, nodes[node.children[i]].data);
                    }
                }
            }
        }
    }
};
