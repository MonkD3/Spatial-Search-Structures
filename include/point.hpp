#ifndef __SPATIAL_STRUCTURE_POINT_HPP__
#define __SPATIAL_STRUCTURE_POINT_HPP__

#include <cmath>
#include <limits>

#include "point.h"
#include "bbox.h"

template<typename Scalar, int dim>
Vec<Scalar, dim>::Vec() noexcept : coord({}) {};

template<typename Scalar, int dim>
Vec<Scalar, dim>::Vec(Scalar x, Scalar y) noexcept requires (dim == 2) : coord({x, y}) {};

template<typename Scalar, int dim>
Vec<Scalar, dim>::Vec(Scalar x, Scalar y, Scalar z) noexcept requires (dim == 3) : coord({x, y, z}) {};

// ========================= Copy / move assignments ======================
template<typename Scalar, int dim>
Vec<Scalar, dim>& Vec<Scalar, dim>::operator=(const VecT& other) noexcept {
    coord = other.coord;
    return *this;
};

template<typename Scalar, int dim>
Vec<Scalar, dim>& Vec<Scalar, dim>::operator=(const VecT&& other) noexcept {
    coord = std::move(other.coord);
    return *this;
};

// ================== In place binary operators ==================
template<typename Scalar, int dim>
Vec<Scalar, dim>& Vec<Scalar, dim>::operator+=(const VecT& other) noexcept {
    for (int i = 0; i < dim; ++i) coord[i] += other.coord[i];
    return *this;
};

template<typename Scalar, int dim>
Vec<Scalar, dim>& Vec<Scalar, dim>::operator-=(const VecT& other) noexcept {
    for (int i = 0; i < dim; ++i) coord[i] -= other.coord[i];
    return *this;
};

template<typename Scalar, int dim>
Vec<Scalar, dim>& Vec<Scalar, dim>::operator*=(Scalar const s) noexcept {
    for (int i = 0; i < dim; ++i) coord[i] *= s;
    return *this;
};

// ================ Accessor ==========================================
template<typename Scalar, int dim>
Scalar& Vec<Scalar, dim>::operator[](int i)       noexcept { return coord[i]; }

template<typename Scalar, int dim>
Scalar  Vec<Scalar, dim>::operator[](int i) const noexcept { return coord[i]; }

// ================== Geometric / linear algebra functions ============
template<typename Scalar, int dim>
Scalar Vec<Scalar, dim>::norm() const noexcept requires (dim == 2) {
    return std::hypot(coord[0], coord[1]);
};

template<typename Scalar, int dim>
Scalar Vec<Scalar, dim>::norm() const noexcept requires (dim == 3) {
    return std::hypot(coord[0], coord[1], coord[2]);
};

template<typename Scalar, int dim>
Vec<Scalar, dim> Vec<Scalar, dim>::unit() const noexcept {
    return (*this) / norm();
}

template<typename Scalar, int dim>
Scalar Vec<Scalar, dim>::sqnorm() const noexcept {
    Scalar sum = static_cast<Scalar>(0);
    for (int i = 0; i < dim; ++i) {
        sum += coord[i] * coord[i];
    }
    return sum;
}

template<typename Scalar, int dim>
Scalar Vec<Scalar, dim>::dot(const VecT& other) const noexcept {
    Scalar sum = static_cast<Scalar>(0);
    for (int i = 0; i < dim; ++i) {
        sum += coord[i] * other[i];
    }
    return sum;
}


// To be able to use them in the quadtree 
template<typename Scalar, int dim>
Vec<Scalar, dim> Vec<Scalar, dim>::get_centroid() const noexcept { return *this; }

template<typename Scalar, int dim>
BBox<Scalar, dim> Vec<Scalar, dim>::get_bbox() const noexcept {
    return BBox<Scalar, dim>(*this, *this);
}

#endif // __SPATIAL_STRUCTURE_POINT_HPP__
