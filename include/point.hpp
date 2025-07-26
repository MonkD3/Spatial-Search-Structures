#ifndef __SPATIAL_STRUCTURE_POINT_HPP
#define __SPATIAL_STRUCTURE_POINT_HPP

#include <array>
#include <cmath>
#include <limits>

template<typename Scalar, int dim> struct BBox; // forward declaration
template<typename Scalar, int dim> struct Vec;  // forward declaration

template<typename _Scalar, int dim>
struct Vec {
    using VecT = Vec<_Scalar, dim>;
    using Scalar = _Scalar;

    std::array<Scalar, dim> coord;

    Vec() : coord({}) {};

    Vec(Scalar x, Scalar y) requires (dim == 2) : coord({x, y}) {};
    Vec(Scalar x, Scalar y, Scalar z) requires (dim == 3) : coord({x, y, z}) {};

    // ========================= Copy / move constructors =====================
    Vec(const VecT& other) = default;
    Vec(VecT&& other) = default;

    // ========================= Copy / move assignments ======================
    VecT& operator=(const VecT& other) {
        coord = other.coord;
        return *this;
    };
    VecT& operator=(const VecT&& other) {
        coord = std::move(other.coord);
        return *this;
    };

    // ================== In place addition/subtraction ==================
    VecT& operator+=(const VecT& other) {
        for (int i = 0; i < dim; ++i) coord[i] += other.coord[i];
        return *this;
    };

    VecT& operator-=(const VecT& other) {
        for (int i = 0; i < dim; ++i) coord[i] -= other.coord[i];
        return *this;
    };

    // ================== In place multiplication by scalar ==============
    VecT& operator*=(Scalar const s) {
        for (int i = 0; i < dim; ++i) coord[i] *= s;
        return *this;
    };

    // ================== Out of place binary operators ==================
    friend VecT operator+(VecT lhs, const VecT& rhs) {
        lhs += rhs;
        return lhs;
    };
    friend VecT operator-(VecT lhs, const VecT& rhs) {
        lhs -= rhs;
        return lhs;
    };

    friend VecT operator*(VecT lhs, const Scalar rhs) {
        lhs *= rhs;
        return lhs;
    };
    friend VecT operator*(const Scalar lhs, VecT rhs) {
        rhs *= lhs;
        return rhs;
    };

    friend VecT operator/(VecT lhs, const Scalar rhs) {
        Scalar const inv = static_cast<Scalar>(1)/rhs;
        lhs *= inv;
        return lhs;
    };
    friend VecT operator/(const Scalar lhs, VecT rhs) {
        Scalar const inv = static_cast<Scalar>(1)/lhs;
        rhs *= inv;
        return rhs;
    };

    // ================= Unary - ==========================================
    friend VecT operator-(VecT p) {
        VecT ret;
        for (int i = 0; i < dim; ++i) ret[i] = -p[i];
        return ret;
    };

    // ================ Accessor ==========================================
    Scalar& operator[](int i)       { return coord[i]; }
    const Scalar& operator[](int i) const { return coord[i]; }

    // ================== Geometric / linear algebra functions ============
    Scalar norm() const requires (dim == 2) {
        return std::hypot(coord[0], coord[1]);
    };
    Scalar norm() const requires (dim == 3) {
        return std::hypot(coord[0], coord[1], coord[2]);
    };

    Scalar sqnorm() const {
        Scalar sum = Scalar(0);
        for (int i = 0; i < dim; ++i) {
            sum += coord[i] * coord[i];
        }
        return sum;
    }

    VecT unit() const {
        return (*this) / norm();
    }

    Scalar dot(const VecT& other) const {
        Scalar sum = Scalar(0);
        for (int i = 0; i < dim; ++i) {
            sum += coord[i] * other[i];
        }
        return sum;
    }

    // To be able to use them in the quadtree 
    VecT get_centroid() const { return *this; }

    BBox<Scalar, dim> get_bbox() const {
        BBox<Scalar, dim> bb;
        bb.pmin = *this;
        bb.pmax = *this;
        return bb;
    }

    static constexpr VecT minPoint() {
        VecT p;
        for (int i = 0; i < dim; ++i) {
            p[i] = - std::numeric_limits<Scalar>::max();
        }
        return p;
    }
    static constexpr VecT maxPoint() {
        VecT p;
        for (int i = 0; i < dim; ++i) {
            p[i] = std::numeric_limits<Scalar>::max();
        }
        return p;
    }
};

#endif // __SPATIAL_STRUCTURE_POINT_HPP
