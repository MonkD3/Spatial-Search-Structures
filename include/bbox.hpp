#ifndef __SPATIAL_SEARCH_STRUCTURE_HPP__
#define __SPATIAL_SEARCH_STRUCTURE_HPP__

#include "point.h"
#include "bbox.h"

template<typename Scalar, int dim>
BBox<Scalar, dim>::BBox() noexcept : pmin(VecT::maxPoint()), pmax(VecT::minPoint()) {};

template<typename Scalar, int dim>
BBox<Scalar, dim>::BBox(const VecT& _pmin, const VecT& _pmax) noexcept : pmin(_pmin), pmax(_pmax) {};

// ============================ Accessors =================================
// Read-Write
template<typename Scalar, int dim>
Scalar& BBox<Scalar, dim>::min(int i) noexcept {return pmin[i];};

template<typename Scalar, int dim>
Scalar& BBox<Scalar, dim>::max(int i) noexcept {return pmax[i];};

// Read-Only
template<typename Scalar, int dim>
Scalar BBox<Scalar, dim>::min(int i) const noexcept {return pmin[i];};

template<typename Scalar, int dim>
Scalar BBox<Scalar, dim>::max(int i) const noexcept {return pmax[i];};


// ========================= Enlarge bounding box =========================
template<typename Scalar, int dim>
void BBox<Scalar, dim>::combineBox(const BBoxT& other) noexcept {
    for (int i = 0; i < dim; i++)
        pmin[i] = std::min(pmin[i], other.pmin[i]);

    for (int i = 0; i < dim; i++)
        pmax[i] = std::max(pmax[i], other.pmax[i]);
}

template<typename Scalar, int dim>
void BBox<Scalar, dim>::combinePoint(const VecT& other) noexcept {
    for (int i = 0; i < dim; i++)
        pmin[i] = std::min(pmin[i], other[i]);

    for (int i = 0; i < dim; i++)
        pmax[i] = std::max(pmax[i], other[i]);
}

template<typename Scalar, int dim>
bool BBox<Scalar, dim>::contains(const VecT& point) const noexcept{
    bool in_bb = true;
    for (int i = 0; i < dim; ++i) in_bb &= point[i] <= pmax[i];
    if (!in_bb) return false; // early return if point outside
    for (int i = 0; i < dim; ++i) in_bb &= point[i] >= pmin[i];

    return in_bb;
}

// =================== Requirements for Quadtree indexing =================
template<typename Scalar, int dim>
Vec<Scalar, dim> BBox<Scalar, dim>::get_centroid() const noexcept {
    return static_cast<Scalar>(0.5) * (pmin + pmax);
}

template<typename Scalar, int dim>
BBox<Scalar, dim> BBox<Scalar, dim>::get_bbox() const noexcept { return *this; }

#endif // __SPATIAL_SEARCH_STRUCTURE_HPP__
