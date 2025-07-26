#ifndef __SPATIAL_STRUCTURE_POINT_HPP
#define __SPATIAL_STRUCTURE_POINT_HPP

#include <array>
#include <cmath>
#include <limits>

template<typename Scalar> struct BBox2; // forward declaration

template<typename _Scalar>
struct Point2 {
    using Scalar = _Scalar;
    std::array<Scalar, 2> coord;

    Point2() : coord({0, 0}) {};
    Point2(Scalar x, Scalar y) : coord({x, y}) {};

    // ========================= Copy / move constructors =====================
    Point2(const Point2<Scalar>& other) = default;
    Point2(Point2<Scalar>&& other) = default;

    // ========================= Copy / move assignments ======================
    Point2<Scalar>& operator=(const Point2<Scalar>& other) {
        coord = other.coord;
        return *this;
    };
    Point2<Scalar>& operator=(const Point2<Scalar>&& other) {
        coord = std::move(other.coord);
        return *this;
    };

    // ================== In place addition/subtraction ==================
    Point2<Scalar>& operator+=(const Point2<Scalar>& other) {
        coord[0] += other.coord[0];
        coord[1] += other.coord[1];
        return *this;
    };
    Point2<Scalar>& operator-=(const Point2<Scalar>& other) {
        coord[0] -= other.coord[0];
        coord[1] -= other.coord[1];
        return *this;
    };

    // ================== In place multiplication by scalar ==============
    Point2<Scalar>& operator*=(Scalar const s) {
        coord[0] *= s;
        coord[1] *= s;
        return *this;
    };

    // ================== Out of place binary operators ==================
    friend Point2<Scalar> operator+(Point2<Scalar> lhs, const Point2<Scalar>& rhs) {
        lhs += rhs;
        return lhs;
    };
    friend Point2<Scalar> operator-(Point2<Scalar> lhs, const Point2<Scalar>& rhs) {
        lhs -= rhs;
        return lhs;
    };

    friend Point2<Scalar> operator*(Point2<Scalar> lhs, const Scalar rhs) {
        lhs *= rhs;
        return lhs;
    };
    friend Point2<Scalar> operator*(const Scalar lhs, Point2<Scalar> rhs) {
        rhs *= lhs;
        return rhs;
    };

    friend Point2<Scalar> operator/(Point2<Scalar> lhs, const Scalar rhs) {
        Scalar const inv = static_cast<Scalar>(1)/rhs;
        lhs *= inv;
        return lhs;
    };
    friend Point2<Scalar> operator/(const Scalar lhs, Point2<Scalar> rhs) {
        Scalar const inv = static_cast<Scalar>(1)/lhs;
        rhs *= inv;
        return rhs;
    };

    // ================= Unary - ==========================================
    friend Point2<Scalar> operator-(Point2<Scalar> p) {
        p[0] = -p[0];
        p[1] = -p[1];
        return p;
    };

    // ================ Accessor ==========================================
    Scalar& operator[](int i)       { return coord[i]; }
    const Scalar& operator[](int i) const { return coord[i]; }

    // ================== Geometric / linear algebra functions ============
    Scalar norm() const {
        return std::hypot(coord[0], coord[1]);
    };

    Scalar sqnorm() const {
        Scalar sum = Scalar(0);
        for (int i = 0; i < 2; ++i) {
            sum += coord[i] * coord[i];
        }
        return sum;
    }

    Point2<Scalar> unit() const {
        return (*this) / norm();
    }

    Scalar dot(const Point2<Scalar>& other) const {
        Scalar sum = Scalar(0);
        for (int i = 0; i < 2; ++i) {
            sum += coord[i] * other[i];
        }
        return sum;
    }

    // To be able to use them in the quadtree 
    Point2<Scalar> get_centroid() const { return *this; }
    BBox2<Scalar> get_bbox() const {
        BBox2<Scalar> bb;
        bb.pmin = *this;
        bb.pmax = *this;
        return bb;
    }

    static constexpr Point2<Scalar> minPoint() {
        Point2<Scalar> p;
        for (int i = 0; i < 2; ++i) {
            p[i] = - std::numeric_limits<Scalar>::max();
        }
        return p;
    }
    static constexpr Point2<Scalar> maxPoint() {
        Point2<Scalar> p;
        for (int i = 0; i < 2; ++i) {
            p[i] = std::numeric_limits<Scalar>::max();
        }
        return p;
    }
};

#endif // __SPATIAL_STRUCTURE_POINT_HPP
