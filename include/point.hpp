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

    Vec() noexcept : coord({}) {};

    Vec(Scalar x, Scalar y) noexcept requires (dim == 2) : coord({x, y}) {};
    Vec(Scalar x, Scalar y, Scalar z) noexcept requires (dim == 3) : coord({x, y, z}) {};

    // ========================= Copy / move constructors =====================
    Vec(const VecT& other) noexcept = default;
    Vec(VecT&& other) noexcept = default;

    // ========================= Copy / move assignments ======================
    VecT& operator=(const VecT& other) noexcept {
        coord = other.coord;
        return *this;
    };
    VecT& operator=(const VecT&& other) noexcept {
        coord = std::move(other.coord);
        return *this;
    };

    // ================== In place addition/subtraction ==================
    VecT& operator+=(const VecT& other) noexcept {
        for (int i = 0; i < dim; ++i) coord[i] += other.coord[i];
        return *this;
    };

    VecT& operator-=(const VecT& other) noexcept {
        for (int i = 0; i < dim; ++i) coord[i] -= other.coord[i];
        return *this;
    };

    // ================== In place multiplication by scalar ==============
    VecT& operator*=(Scalar const s) noexcept {
        for (int i = 0; i < dim; ++i) coord[i] *= s;
        return *this;
    };

    // ================== Out of place binary operators ==================
    friend VecT operator+(VecT lhs, const VecT& rhs) noexcept {
        lhs += rhs;
        return lhs;
    };
    friend VecT operator-(VecT lhs, const VecT& rhs) noexcept {
        lhs -= rhs;
        return lhs;
    };

    friend VecT operator*(VecT lhs, const Scalar rhs) noexcept {
        lhs *= rhs;
        return lhs;
    };
    friend VecT operator*(const Scalar lhs, VecT rhs) noexcept {
        rhs *= lhs;
        return rhs;
    };

    friend VecT operator/(VecT lhs, const Scalar rhs) noexcept {
        Scalar const inv = static_cast<Scalar>(1)/rhs;
        lhs *= inv;
        return lhs;
    };
    friend VecT operator/(const Scalar lhs, VecT rhs) noexcept {
        Scalar const inv = static_cast<Scalar>(1)/lhs;
        rhs *= inv;
        return rhs;
    };

    // ================= Unary - ==========================================
    friend VecT operator-(VecT p) noexcept {
        VecT ret;
        for (int i = 0; i < dim; ++i) ret[i] = -p[i];
        return ret;
    };

    // ================ Accessor ==========================================
    Scalar& operator[](int i)       noexcept { return coord[i]; }
    Scalar  operator[](int i) const noexcept { return coord[i]; }

    // ================== Geometric / linear algebra functions ============
    Scalar norm() const noexcept requires (dim == 2) {
        return std::hypot(coord[0], coord[1]);
    };
    Scalar norm() const noexcept requires (dim == 3) {
        return std::hypot(coord[0], coord[1], coord[2]);
    };

    VecT unit() const noexcept {
        return (*this) / norm();
    }

    Scalar sqnorm() const noexcept {
        Scalar sum = static_cast<Scalar>(0);
        for (int i = 0; i < dim; ++i) {
            sum += coord[i] * coord[i];
        }
        return sum;
    }

    Scalar dot(const VecT& other) const noexcept {
        Scalar sum = static_cast<Scalar>(0);
        for (int i = 0; i < dim; ++i) {
            sum += coord[i] * other[i];
        }
        return sum;
    }

    // Component wise multiplication
    friend VecT cMult(const VecT& p, const VecT& q) noexcept {
        VecT ret;
        for (int i = 0; i < dim; ++i) ret[i] = p[i]*q[i];
        return ret;
    }

    // To be able to use them in the quadtree 
    VecT get_centroid() const noexcept { return *this; }

    BBox<Scalar, dim> get_bbox() const noexcept {
        return BBox<Scalar, dim>(*this, *this);
    }

    static constexpr VecT minPoint() noexcept {
        VecT p;
        for (int i = 0; i < dim; ++i) {
            p[i] = - std::numeric_limits<Scalar>::max();
        }
        return p;
    }
    static constexpr VecT maxPoint() noexcept {
        VecT p;
        for (int i = 0; i < dim; ++i) {
            p[i] = std::numeric_limits<Scalar>::max();
        }
        return p;
    }
};

#endif // __SPATIAL_STRUCTURE_POINT_HPP
