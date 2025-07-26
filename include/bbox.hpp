#ifndef __SPATIAL_SEARCH_STRUCTURE_HPP__
#define __SPATIAL_SEARCH_STRUCTURE_HPP__

#include <limits>
#include <array>

#include "point.hpp"

template<typename Scalar>
struct BBox2 {
    Point2<Scalar> pmin;
    Point2<Scalar> pmax;

    // Init an inverted bounding box
    BBox2() : pmin(Point2<Scalar>::maxPoint()), pmax(Point2<Scalar>::minPoint()) {};

    // Combine the bounding box with another
    inline void combineBox(const BBox2<Scalar>& other) {
        for (int i = 0; i < 2; i++)
            pmin[i] = std::min(pmin[i], other.pmin[i]);

        for (int i = 0; i < 2; i++)
            pmax[i] = std::max(pmax[i], other.pmax[i]);
    }
    inline void combinePoint(const Point2<Scalar>& other) {
        for (int i = 0; i < 2; i++)
            pmin[i] = std::min(pmin[i], other[i]);

        for (int i = 0; i < 2; i++)
            pmax[i] = std::max(pmax[i], other[i]);
    }


    inline Scalar& min(int i) {return pmin[i];};
    inline Scalar& max(int i) {return pmax[i];};

    inline const Scalar& min(int i) const {return pmin[i];};
    inline const Scalar& max(int i) const {return pmax[i];};

    inline Point2<Scalar> get_centroid() const {
        return  static_cast<Scalar>(0.5) * (pmin + pmax);
    }

    inline BBox2<Scalar> get_bbox() const {
        return *this;
    }
};

#endif // __SPATIAL_SEARCH_STRUCTURE_HPP__
