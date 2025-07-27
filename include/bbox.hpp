#ifndef __SPATIAL_SEARCH_STRUCTURE_HPP__
#define __SPATIAL_SEARCH_STRUCTURE_HPP__

#include "point.hpp"

template<typename Scalar, int dim>
struct BBox {
    using VecT = Vec<Scalar, dim>;
    using BBoxT = BBox<Scalar, dim>;

    VecT pmin;
    VecT pmax;

    // Init an inverted bounding box
    BBox() : pmin(VecT::maxPoint()), pmax(VecT::minPoint()) {};

    // Init a bounding box using the bottom-left and top-right corners
    BBox(const VecT& _pmin, const VecT& _pmax) : pmin(_pmin), pmax(_pmax) {};

    // ============================ Accessors =================================
    // Read-Write
    inline Scalar& min(int i) {return pmin[i];};
    inline Scalar& max(int i) {return pmax[i];};

    // Read-Only
    inline Scalar min(int i) const {return pmin[i];};
    inline Scalar max(int i) const {return pmax[i];};


    // ========================= Enlarge bounding box =========================
    // Combine the bounding box with another
    inline void combineBox(const BBoxT& other) {
        for (int i = 0; i < dim; i++)
            pmin[i] = std::min(pmin[i], other.pmin[i]);

        for (int i = 0; i < dim; i++)
            pmax[i] = std::max(pmax[i], other.pmax[i]);
    }

    // Add a point to the bounding box
    inline void combinePoint(const VecT& other) {
        for (int i = 0; i < dim; i++)
            pmin[i] = std::min(pmin[i], other[i]);

        for (int i = 0; i < dim; i++)
            pmax[i] = std::max(pmax[i], other[i]);
    }

    // =================== Requirements for Quadtree indexing =================
    inline VecT get_centroid() const {
        return static_cast<Scalar>(0.5) * (pmin + pmax);
    }

    inline BBoxT get_bbox() const { return *this; }
};

#endif // __SPATIAL_SEARCH_STRUCTURE_HPP__
