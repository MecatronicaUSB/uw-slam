/**
* This file is part of UW-SLAM.
* 
* Copyright 2018.
* Developed by Fabio Morales,
* Email: fabmoraleshidalgo@gmail.com; GitHub: @fmoralesh
*
* UW-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* UW-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with UW-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/
#pragma once
#include "Eigen/Core"
#include "ceres/ceres.h"
#include "sophus/sim3.hpp"
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

// This workaround creates a template specilization for Eigen's cast_impl,
// when casting from a ceres::Jet type. It relies on Eigen's internal API and
// might break with future versions of Eigen.
namespace Eigen {
namespace internal {

template <class T, int N, typename NewType>
struct cast_impl<ceres::Jet<T, N>, NewType> {
  EIGEN_DEVICE_FUNC
  static inline NewType run(ceres::Jet<T, N> const& x) {
    return static_cast<NewType>(x.a);
  }
};

}  // namespace internal
}  // namespace Eigen


namespace ceres {

// A jet traits class to make it easier to work with mixed auto / numeric diff.
template<typename T>
struct JetOps {
  static bool IsScalar() {
    return true;
  }
  static T GetScalar(const T& t) {
    return t;
  }
  static void SetScalar(const T& scalar, T* t) {
    *t = scalar;
  }
  static void ScaleDerivative(double /*scale_by*/, T * /*value*/) {
    // For double, there is no derivative to scale.
  }
};

template<typename T, int N>
struct JetOps<Jet<T, N> > {
  static bool IsScalar() {
    return false;
  }
  static T GetScalar(const Jet<T, N>& t) {
    return t.a;
  }
  static void SetScalar(const T& scalar, Jet<T, N>* t) {
    t->a = scalar;
  }
  static void ScaleDerivative(double scale_by, Jet<T, N> *value) {
    value->v *= scale_by;
  }
};

template<typename FunctionType, int kNumArgs, typename ArgumentType>
struct Chain {
  static ArgumentType Rule(const FunctionType &f,
                           const FunctionType /*dfdx*/[kNumArgs],
                           const ArgumentType /*x*/[kNumArgs]) {
    // In the default case of scalars, there's nothing to do since there are no
    // derivatives to propagate.
    return f;
  }
};

// XXX Add documentation here!
template<typename FunctionType, int kNumArgs, typename T, int N>
struct Chain<FunctionType, kNumArgs, Jet<T, N> > {
  static Jet<T, N> Rule(const FunctionType &f,
                        const FunctionType dfdx[kNumArgs],
                        const Jet<T, N> x[kNumArgs]) {
    // x is itself a function of another variable ("z"); what this function
    // needs to return is "f", but with the derivative with respect to z
    // attached to the jet. So combine the derivative part of x's jets to form
    // a Jacobian matrix between x and z (i.e. dx/dz).
    Eigen::Matrix<T, kNumArgs, N> dxdz;
    for (int i = 0; i < kNumArgs; ++i) {
      dxdz.row(i) = x[i].v.transpose();
    }

    // Map the input gradient dfdx into an Eigen row vector.
    Eigen::Map<const Eigen::Matrix<FunctionType, 1, kNumArgs> >
        vector_dfdx(dfdx, 1, kNumArgs);

    // Now apply the chain rule to obtain df/dz. Combine the derivative with
    // the scalar part to obtain f with full derivative information.
    Jet<T, N> jet_f;
    jet_f.a = f;
    jet_f.v = vector_dfdx.template cast<T>() * dxdz;  // Also known as dfdz.
    return jet_f;
  }
};

}  // namespace ceres

namespace uw
{

#define SSEE(val,idx) (*(((float*)&val)+idx))
#define ALIGN __attribute__((__aligned__(16)))

typedef Sophus::SE3f SE3;
typedef Sophus::Sim3f Sim3;
typedef Sophus::SO3f SO3;

typedef Eigen::Vector4d QuaternionVector;
typedef Eigen::Vector3d TranslationVector;
typedef Eigen::Quaternion<SE3::Scalar> Quaternion;
typedef Eigen::Matrix<double,3,1> Mat31d;
typedef Eigen::Matrix<double,3,3> Mat33d;
typedef Eigen::Matrix<double,4,1> Mat41d;
typedef Eigen::Matrix<double,4,4> Mat44d;
typedef Eigen::Matrix<double,6,1> Mat61d;
typedef Eigen::Matrix<double,6,6> Mat66d;
typedef Eigen::Matrix<double,6,7> Mat67d;

typedef Eigen::Matrix<float,3,1> Mat31f;
typedef Eigen::Matrix<float,3,3> Mat33f;
typedef Eigen::Matrix<float,3,4> Mat34f;
typedef Eigen::Matrix<float,4,1> Mat41f;
typedef Eigen::Matrix<float,4,4> Mat44f;
typedef Eigen::Matrix<float,6,1> Mat61f;
typedef Eigen::Matrix<float,6,6> Mat66f;
typedef Eigen::Matrix<float,6,7> Mat67f;

// Global constants
extern const int PYRAMID_LEVELS;
extern int BLOCK_SIZE;
extern double GRADIENT_THRESHOLD;

}