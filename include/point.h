#ifndef __SPATIAL_STRUCTURE_POINT_H__
#define __SPATIAL_STRUCTURE_POINT_H__

#include <array>
#include <limits>

template<typename Scalar, int dim> struct BBox; // forward declaration
template<typename Scalar, int dim> struct Vec;  // forward declaration

template<typename _Scalar, int dim>
struct Vec {
    using VecT = Vec<_Scalar, dim>;
    using Scalar = _Scalar;

    std::array<Scalar, dim> coord;

    Vec() noexcept;

    Vec(Scalar x, Scalar y) noexcept requires (dim == 2);
    Vec(Scalar x, Scalar y, Scalar z) noexcept requires (dim == 3);

    // ========================= Copy / move constructors =====================
    Vec(const VecT& other) noexcept = default;
    Vec(VecT&& other) noexcept = default;

    // ========================= Copy / move assignments ======================
    VecT& operator=(const VecT& other) noexcept;
    VecT& operator=(const VecT&& other) noexcept;

    // ================== In place addition/subtraction ==================
    VecT& operator+=(const VecT& other) noexcept;

    VecT& operator-=(const VecT& other) noexcept;

    // ================== In place multiplication by scalar ==============
    VecT& operator*=(Scalar const s) noexcept;

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

    // Component wise multiplication
    friend VecT cMult(const VecT& p, const VecT& q) noexcept {
        VecT ret;
        for (int i = 0; i < dim; ++i) ret[i] = p[i]*q[i];
        return ret;
    }

    // ================= Unary - ==========================================
    friend VecT operator-(VecT p) noexcept {
        VecT ret;
        for (int i = 0; i < dim; ++i) ret[i] = -p[i];
        return ret;
    };

    // ================ Accessor ==========================================
    Scalar& operator[](int i)       noexcept;
    Scalar  operator[](int i) const noexcept;

    // ================== Geometric / linear algebra functions ============
    Scalar norm() const noexcept requires (dim == 2);
    Scalar norm() const noexcept requires (dim == 3);
    Scalar sqnorm() const noexcept;
    Scalar dot(const VecT& other) const noexcept;

    VecT unit() const noexcept;

    // To be able to use them in the quadtree 
    VecT get_centroid() const noexcept;
    BBox<Scalar, dim> get_bbox() const noexcept;

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
    };
};


#ifdef __cplusplus
extern "C" {
#endif

typedef Vec<float, 2>  Vec2f;
typedef Vec<float, 3>  Vec3f;
typedef Vec<double, 2> Vec2d;
typedef Vec<double, 3> Vec3d;

#ifdef __cplusplus
}
#endif

#endif // __SPATIAL_STRUCTURE_POINT_H__
