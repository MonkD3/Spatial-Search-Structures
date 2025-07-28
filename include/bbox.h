#ifndef __SPATIAL_SEARCH_STRUCTURE_H__
#define __SPATIAL_SEARCH_STRUCTURE_H__

#include "point.h"

template<typename Scalar, int dim>
struct BBox {
    using VecT = Vec<Scalar, dim>;
    using BBoxT = BBox<Scalar, dim>;

    VecT pmin;
    VecT pmax;

    // Init an inverted bounding box
    BBox() noexcept;

    // Init a bounding box using the bottom-left and top-right corners
    BBox(const VecT& _pmin, const VecT& _pmax) noexcept;

    // ============================ Accessors =================================
    // Read-Write
    Scalar& min(int i) noexcept;
    Scalar& max(int i) noexcept;

    // Read-Only
    Scalar min(int i) const noexcept;
    Scalar max(int i) const noexcept;


    // ========================= Enlarge bounding box =========================
    // Combine the bounding box with another
    void combineBox(const BBoxT& other) noexcept;
    // Add a point to the bounding box
    void combinePoint(const VecT& other) noexcept;

    // =================== Requirements for Quadtree indexing =================
    VecT get_centroid() const noexcept;
    BBoxT get_bbox() const noexcept;
};

#ifdef __cplusplus
extern "C" {
#endif

typedef BBox<float, 2>  BBox2f;
typedef BBox<float, 3>  BBox3f;
typedef BBox<double, 2> BBox2d;
typedef BBox<double, 3> BBox3d;

#ifdef __cplusplus
}
#endif

#endif // __SPATIAL_SEARCH_STRUCTURE_HPP__
