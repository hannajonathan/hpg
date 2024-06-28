// Copyright 2021 Associated Universities, Inc. Washington DC, USA.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#pragma once

#include "hpg_config.hpp"
#include "hpg.hpp"
#include "hpg_error.hpp"
// #include "hpg_export.h"

#include <cassert>
#include <cmath>
#include <optional>
#include <string>
#include <tuple>
#include <iostream>

#include <Kokkos_Core.hpp>

#if defined(HPG_ENABLE_SERIAL) || defined(HPG_ENABLE_OPENMP)
# include <fftw3.h>
# ifdef HPG_ENABLE_OPENMP
#  include <omp.h>
# endif
#endif
#ifdef HPG_ENABLE_CUDA
# include <cufft.h>
#endif

namespace hpg::runtime::impl::core {

namespace K = Kokkos;

/** scoped Kokkos profiling region value */
struct /*HPG_EXPORT*/ ProfileRegion {
  inline ProfileRegion(const char* nm) {
    K::Profiling::pushRegion(nm);
  }

  inline ~ProfileRegion() {
    K::Profiling::popRegion();
  }
};

/** ordered Grid array axes */
enum class /*HPG_EXPORT*/ GridAxis {
  x,
  y,
  mrow,
  cube
};

/** ordered CF array axes */
enum class /*HPG_EXPORT*/ CFAxis {
  x_major,
  y_major,
  mueller,
  cube,
  x_minor,
  y_minor
};

/** scalar type for all polarization products of a visibility value
 *
 * Values of this type can be used in Kokkos reductions
 *
 * @tparam T floating point type of visibility values
 * @tparam N number of polarizations
 */
template<typename T, int N>
struct /*HPG_EXPORT*/ poln_array_type {

  K::complex<T> vals[N];

  KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
  poln_array_type() {
    for (int i = 0; i < N; ++i) {
      vals[i] = 0;
    }
  }
  KOKKOS_INLINE_FUNCTION   // Copy Constructor
  poln_array_type(const poln_array_type& rhs) {
    for (int i = 0; i < N; ++i) {
      vals[i] = rhs.vals[i];
    }
  }
  KOKKOS_INLINE_FUNCTION   // add operator
  poln_array_type&
  operator +=(const poln_array_type& src) {
    for (int i = 0; i < N; ++i) {
      vals[i] += src.vals[i];
    }
    return *this;
  }
  KOKKOS_INLINE_FUNCTION   // volatile add operator
  void
  operator +=(const volatile poln_array_type& src) volatile {
    for (int i = 0; i < N; ++i) {
      vals[i] += src.vals[i];
    }
  }
  KOKKOS_INLINE_FUNCTION
  void
  set_nan() {
    for (int i = 0; i < N; ++i)
      vals[i] = K::complex<T>{NAN, NAN};
  }
};

/** scalar type for all polarization products of visibility values and weights
 *
 * Values of this type can be used in Kokkos reductions
 *
 * @tparam T floating point type of values
 * @tparam N number of polarizations
 */
template<typename T, int N>
struct /*HPG_EXPORT*/ vis_array_type {

  K::Array<K::complex<T>, N> vis;
  K::Array<K::complex<T>, N> wgt;

  KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
  vis_array_type() {
    for (int i = 0; i < N; ++i) {
      vis[i] = 0;
      wgt[i] = 0;
    }
  }
  KOKKOS_INLINE_FUNCTION   // Copy Constructor
  vis_array_type(const vis_array_type& rhs) {
    for (int i = 0; i < N; ++i) {
      vis[i] = rhs.vis[i];
      wgt[i] = rhs.wgt[i];
    }
  }
  KOKKOS_INLINE_FUNCTION   // add operator
  vis_array_type&
  operator +=(const vis_array_type& src) {
    for (int i = 0; i < N; ++i) {
      vis[i] += src.vis[i];
      wgt[i] += src.wgt[i];
    }
    return *this;
  }
  KOKKOS_INLINE_FUNCTION   // volatile add operator
  void
  operator +=(const volatile vis_array_type& src) volatile {
    for (int i = 0; i < N; ++i) {
      vis[i] += src.vis[i];
      wgt[i] += src.wgt[i];
    }
  }
};
} // end namespace hpg::runtime::impl::core

namespace Kokkos { //reduction identities must be defined in Kokkos namespace
/** reduction identity of poln_array_type */
template<int N>
struct /*HPG_EXPORT*/ reduction_identity<
  hpg::runtime::impl::core::poln_array_type<float, N>> {

  KOKKOS_FORCEINLINE_FUNCTION static
  hpg::runtime::impl::core::poln_array_type<float, N> sum() {
    return hpg::runtime::impl::core::poln_array_type<float, N>();
  }
};
template<int N>
struct /*HPG_EXPORT*/ reduction_identity<
  hpg::runtime::impl::core::poln_array_type<double, N>> {

  KOKKOS_FORCEINLINE_FUNCTION static
  hpg::runtime::impl::core::poln_array_type<double, N> sum() {
    return hpg::runtime::impl::core::poln_array_type<double, N>();
  }
};

/** reduction identity of vis_array_type */
template<int N>
struct /*HPG_EXPORT*/ reduction_identity<
  hpg::runtime::impl::core::vis_array_type<float, N>> {

  KOKKOS_FORCEINLINE_FUNCTION static
  hpg::runtime::impl::core::vis_array_type<float, N> sum() {
    return hpg::runtime::impl::core::vis_array_type<float, N>();
  }
};
template<int N>
struct /*HPG_EXPORT*/ reduction_identity<
  hpg::runtime::impl::core::vis_array_type<double, N>> {

  KOKKOS_FORCEINLINE_FUNCTION static
  hpg::runtime::impl::core::vis_array_type<double, N> sum() {
    return hpg::runtime::impl::core::vis_array_type<double, N>();
  }
};
}

namespace hpg::runtime::impl::core {

/** accumulation value type for complex values
 *
 * @tparam C K::Complex<.> type
 */
template <typename C>
using acc_cpx_t =
  K::complex<
    std::conditional_t<
      std::is_same_v<typename C::value_type, float>,
      double,
      long double>>;

/** visibility value type */
using vis_t = K::complex<visibility_fp>;

using acc_vis_t = acc_cpx_t<vis_t>;

/** convolution function value type */
using cf_t = K::complex<cf_fp>;

using acc_cf_t = acc_cpx_t<cf_t>;

/** gridded value type */
using gv_t = K::complex<grid_value_fp>;

/** portable UVW coordinates type */
using uvw_t = K::Array<vis_uvw_fp, 3>;

/** visibilities plus metadata for gridding */
template <unsigned N>
struct /*HPG_EXPORT*/ VisData {

  static constexpr unsigned npol = N;

  KOKKOS_INLINE_FUNCTION VisData() {};

  KOKKOS_INLINE_FUNCTION VisData(
    const K::Array<vis_t, N>& values, /**< visibility values */
    const K::Array<vis_weight_fp, N> weights, /**< visibility weights */
    vis_frequency_fp freq, /**< frequency */
    vis_phase_fp d_phase, /**< phase angle */
    const uvw_t& uvw, /** < uvw coordinates */
    unsigned& grid_cube, /**< grid cube index */
    const K::Array<unsigned, 2>& cf_index, /**< cf (cube, grp) index */
    /** cf phase gradient */
    const K::Array<cf_phase_gradient_fp, 2>& cf_phase_gradient)
    : m_values(values)
    , m_weights(weights)
    , m_freq(freq)
    , m_d_phase(d_phase)
    , m_uvw(uvw)
    , m_grid_cube(grid_cube)
    , m_cf_index(cf_index)
    , m_cf_phase_gradient(cf_phase_gradient) {}

  VisData(VisData const&) = default;

  VisData(VisData&&) noexcept = default;

  ~VisData() = default;

  VisData& operator=(VisData const&) = default;

  VisData& operator=(VisData&&) noexcept = default;

  K::Array<vis_t, N> m_values;
  K::Array<vis_weight_fp, N> m_weights;
  vis_frequency_fp m_freq;
  vis_phase_fp m_d_phase;
  uvw_t m_uvw;
  unsigned m_grid_cube;
  K::Array<unsigned, 2> m_cf_index;
  K::Array<cf_phase_gradient_fp, 2> m_cf_phase_gradient;
};

/** View type for grid values */
template <typename Layout, typename memory_space>
using grid_view = K::View<gv_t****, Layout, memory_space>;

/** View type for constant grid values */
template <typename Layout, typename memory_space>
using const_grid_view = K::View<const gv_t****, Layout, memory_space>;

/** View type for weight values
 *
 * logical axis order: mrow, cube
 */
template <typename Layout, typename memory_space>
using weight_view = K::View<grid_value_fp**, Layout, memory_space>;

/** View type for constant weight values */
template <typename Layout, typename memory_space>
using const_weight_view = K::View<const grid_value_fp**, Layout, memory_space>;

/** View type for CF values */
template <typename Layout, typename memory_space>
using cf_view =
  K::View<cf_t******, Layout, memory_space, K::MemoryTraits<K::Unmanaged>>;

/** View type for constant CF values */
template <typename Layout, typename memory_space>
using const_cf_view =
  K::View<
    const cf_t******,
    Layout,
    memory_space,
    K::MemoryTraits<K::Unmanaged>>;

/** view type for unmanaged view of vector data on host */
template <typename T>
using vector_view = K::View<T*, K::HostSpace, K::MemoryTraits<K::Unmanaged>>;

/** view for VisData<N> */
template <unsigned N, typename memory_space>
using visdata_view =
  K::View<VisData<N>*, memory_space, K::MemoryTraits<K::Unmanaged>>;

/** View for constant VisData<N> */
template <unsigned N, typename memory_space>
using const_visdata_view =
  K::View<const VisData<N>*, memory_space, K::MemoryTraits<K::Unmanaged>>;

/** view for backing buffer of visdata views in ExecSpace */
template <typename memory_space>
using visbuff_view = K::View<VisData<4>*, memory_space>;

/** view for backing buffer of gvisvals views in ExecSpace */
template <typename memory_space>
using gvisbuff_view = K::View<poln_array_type<visibility_fp, 4>*, memory_space>;

/** view for Mueller element index matrix */
template <typename memory_space>
using mindex_view = K::View<int[4][4], memory_space>;

/** view for constant Mueller element index matrix */
template <typename memory_space>
using const_mindex_view =
  K::View<const int[4][4], memory_space, K::MemoryTraits<K::RandomAccess>>;

/** almost atomic complex addition
 *
 * We want atomic addition to avoid the use of locks, that is, to be implemented
 * only by either a real atomic addition function or a compare-and-swap loop. We
 * define this function since 128 bit CAS functions don't exist in CUDA or HIP,
 * and we want at least a chance of compiling to the actual atomic add
 * functions. This function is almost atomic since the real and imaginary
 * components are updated sequentially.
 */
template <typename execution_space, typename T>
/*HPG_EXPORT*/ KOKKOS_FORCEINLINE_FUNCTION void
pseudo_atomic_add(volatile K::complex<T>& acc, const K::complex<T>& val) {
  K::atomic_add(&acc, val);
}

#ifdef HPG_ENABLE_CUDA
template <>
/*HPG_EXPORT*/ KOKKOS_FORCEINLINE_FUNCTION void
pseudo_atomic_add<K::Cuda, double>(
  volatile K::complex<double>& acc, const K::complex<double>& val) {

  K::atomic_add(&acc.real(), val.real());
  K::atomic_add(&acc.imag(), val.imag());
}

template <>
/*HPG_EXPORT*/ KOKKOS_FORCEINLINE_FUNCTION void
pseudo_atomic_add<K::Cuda, float>(
  volatile K::complex<float>& acc, const K::complex<float>& val) {

  K::atomic_add(&acc.real(), val.real());
  K::atomic_add(&acc.imag(), val.imag());
}
#endif // HPG_ENABLE_CUDA

/** convert UV coordinate to major and minor grid coordinates, and CF
 * coordinates
 *
 * Computed coordinates may refer to the points in the domain of the CF function
 * translated to the position of the visibility on the fine (oversampled) grid.
 *
 * The four returned coordinates are as follows
 * - leftmost grid coordinate of (visibility-centered) CF support
 * - leftmost major CF coordinate within CF support
 * - the "minor" coordinate of the CF, always non-negative
 * - offset of visibility (on fine grid) to nearest major grid
 *   point (positive or negative)
 *
 * For negative grid_scale values, in the above description, change "left" to
 * "right"
 *
 * The value of the minor coordinate must be between 0 and oversampling - 1
 * (inclusive); it's computation proceeds as follows:
 *
 * - G is the grid coordinate nearest position
 * - fine_offset is the distance from (visibility) position to nearest grid
 * coordinate, (G - position) * oversampling
 * - points at which CF are evaluated are {(I - (position - G)) * oversampling}
 * or {I * oversampling + fine_offset} for I in some range of integers
 * - the left edge of the support of CF is nominally at CFArray::padding
 * - CFArray employs a decomposed form of 1d index i as (i / oversampling, i %
 * oversampling), where the second component is always between 0 and
 * oversampling - 1
 * - if fine_offset >= 0, {I * oversampling + fine_offset}, and the CF indexes
 * are (I, fine_offset)
 * - if fine_offset <= 0, {I * oversampling + fine_offset} = {(I - 1) *
 *  oversampling + (fine_offset + oversampling)}, and the CF indexes are (I - 1,
 *  fine_offset + oversampling)
 *
 * @return tuple comprising four integer coordinates
 */
/*HPG_EXPORT*/ KOKKOS_INLINE_FUNCTION std::tuple<int, int, int, int>
compute_vis_coord(
  int g_size,
  int oversampling,
  int cf_radius,
  vis_uvw_fp coord,
  vis_frequency_fp inv_lambda,
  grid_scale_fp grid_scale) {

  const double position = grid_scale * coord * inv_lambda + g_size / 2.0;
  long grid_coord = std::lrint(position);
  const long fine_offset = std::lrint((grid_coord - position) * oversampling);
  grid_coord -= cf_radius;
  long cf_minor;
  long cf_major;
  if (fine_offset >= 0) {
    cf_minor = fine_offset;
    cf_major = CFArray::padding;
  } else {
    cf_minor = oversampling + fine_offset;
    cf_major = CFArray::padding - 1;
  }
  assert(0 <= cf_minor && cf_minor < oversampling);
  return {grid_coord, cf_major, cf_minor, fine_offset};
}

/** portable sincos()
 */
#pragma nv_exec_check_disable
template <typename execution_space, typename T>
/*HPG_EXPORT*/ KOKKOS_FORCEINLINE_FUNCTION void
sincos(T ph, T* sn, T* cs) {
  *sn = std::sin(ph);
  *cs = std::cos(ph);
}

#ifdef KOKKOS_ENABLE_CUDA
template <>
/*HPG_EXPORT*/ __device__ __forceinline__ void
sincos<K::Cuda, float>(float ph, float* sn, float* cs) {
  ::sincosf(ph, sn, cs);
}
template <>
/*HPG_EXPORT*/ __device__ __forceinline__ void
sincos<K::Cuda, double>(double ph, double* sn, double* cs) {
  ::sincos(ph, sn, cs);
}
#endif

/** convert phase to complex value
 */
template <typename execution_space, typename T>
/*HPG_EXPORT*/ KOKKOS_FORCEINLINE_FUNCTION K::complex<T>
cphase(T ph) {
  K::complex<T> result;
  sincos<execution_space, T>(ph, &result.imag(), &result.real());
  return result;
}

/** magnitude of K::complex<T> value */
template <typename T>
/*HPG_EXPORT*/ KOKKOS_FORCEINLINE_FUNCTION T
mag(const K::complex<T>& v) {
  return std::hypot(v.real(), v.imag());
}

/** helper class for computing visibility value and index metadata
 *
 * Basically exists to encapsulate conversion from a Visibility value to several
 * visibility metadata values needed by gridding kernel
 */
template <unsigned N, typename execution_space>
struct /*HPG_EXPORT*/ Vis final {

  int m_grid_coord[2]; /**< grid coordinate */
  int m_cf_minor[2]; /**< CF minor coordinate */
  int m_cf_major[2]; /**< CF major coordinate */
  int m_fine_offset[2]; /**< visibility position - nearest major grid */
  int m_cf_size[2]; /**< cf size */
  K::Array<vis_t, N> m_values; /**< visibility values */
  K::Array<vis_weight_fp, N> m_weights; /**< visibility weights */
  K::complex<vis_phase_fp> m_phasor;
  int m_grid_cube; /**< grid cube index */
  int m_cf_cube; /**< cf cube index */
  int m_cf_grp; /**< cf group index */
  bool m_pos_w; /**< true iff W coordinate is strictly positive */
  cf_phase_gradient_fp m_phi0[2]; /**< phase screen value origin */
  cf_phase_gradient_fp m_dphi[2]; /**< phase screen value increment */

  KOKKOS_INLINE_FUNCTION Vis() {};

  KOKKOS_INLINE_FUNCTION Vis(
    const VisData<N>& vis,
    const K::Array<vis_t, N>& vals,
    const K::Array<int, 2>& grid_size,
    const K::Array<grid_scale_fp, 2>& grid_scale,
    const K::Array<K::Array<int, 2>, HPG_MAX_NUM_CF_GROUPS>& cf_radii,
    const K::Array<int, 2>& oversampling)
    : m_values(vals)
    , m_weights(vis.m_weights)
    , m_phasor(cphase<execution_space>(vis.m_d_phase))
    , m_grid_cube(vis.m_grid_cube)
    , m_cf_cube(vis.m_cf_index[0])
    , m_cf_grp(vis.m_cf_index[1])
    , m_pos_w(vis.m_uvw[2] > 0) {

    static const vis_frequency_fp c = 299792458.0;
    auto inv_lambda = vis.m_freq / c;

    for (const auto d : {0, 1}) {
      m_cf_size[d] = 2 * cf_radii[m_cf_grp][d] + 1;
      auto [g, maj, min, f] =
        compute_vis_coord(
          grid_size[d],
          oversampling[d],
          m_cf_size[d] / 2,
          vis.m_uvw[d],
          inv_lambda,
          grid_scale[d]);
      m_grid_coord[d] = g;
      m_cf_major[d] = maj;
      m_cf_minor[d] = min;
      m_fine_offset[d] = f;
      m_phi0[d] =
        -vis.m_cf_phase_gradient[d]
        * ((m_cf_size[d] / 2) * oversampling[d] - m_fine_offset[d]);
      m_dphi[d] = vis.m_cf_phase_gradient[d] * oversampling[d];
    }
  }

  Vis(Vis const&) = default;

  Vis(Vis&&) noexcept = default;

  Vis& operator=(Vis const&) = default;

  Vis& operator=(Vis&&) noexcept = default;
};

// we're wrapping each kernel in a class in order to support partial
// specialization of the kernel functions by execution space

/** gridding kernel
 *
 * Note that the default implementation probably is optimal only for many-core
 * devices, probably not OpenMP (although it is correct on all devices).
 */
template <int N, typename execution_space, unsigned version>
struct /*HPG_EXPORT*/ VisibilityGridder final {

  using member_type = typename K::TeamPolicy<execution_space>::member_type;

  using scratch_phscr_view =
    K::View<
      cf_phase_gradient_fp*,
    typename execution_space::scratch_memory_space>;

  template <typename cf_layout, typename grid_layout, typename memory_space>
  static KOKKOS_FUNCTION poln_array_type<visibility_fp, N>
  degrid_vis(
    // gv_t = complex gridded value type
    // cf_t = convolution function value type
    const member_type& team_member, //using member_type = typename K::TeamPolicy<execution_space>::member_type
    const Vis<N, execution_space>& vis, //line 495, struct Vis
    const cf_view<cf_layout, memory_space>& cf, //using cf_view = K::View<cf_t******, Layout, memory_space, K::MemoryTraits<K::Unmanaged>>;
    const const_mindex_view<memory_space>& mueller_indexes, //using const_mindex_view = K::View<const int[4][4], memory_space, K::MemoryTraits<K::RandomAccess>>;
    const const_mindex_view<memory_space>& conjugate_mueller_indexes,
    const const_grid_view<grid_layout, memory_space>& model, //using const_grid_view = K::View<const gv_t****, Layout, memory_space>;
    const scratch_phscr_view& phi_Y) { // using scratch_phscr_view = K::View<cf_phase_gradient_fp*, typename execution_space::scratch_memory_space>;

    const auto& N_X = vis.m_cf_size[0]; // first index of cf_size array (how many pixels along u dim.)
    const auto& N_Y = vis.m_cf_size[1]; // second index of cf_size array (along v dim.)
    const auto N_R = model.extent_int(int(GridAxis::mrow)); // Number of elements in mrow of GridAxis?

    std::cout << "Using VisibilityGridder case 0 degrid_vis" << std::endl;

    // vis.m_pos_w = true iff W coordinate is strictly positive
    // mindex is index of the Mueller matrix
    auto degridding_mindex =
      vis.m_pos_w ? conjugate_mueller_indexes : mueller_indexes;
    cf_fp cf_im_factor = (vis.m_pos_w ? 1 : -1);

    // compute the values of the phase screen along the Y axis now and store the
    // results in scratch memory because gridding on the Y axis accesses the
    // phase screen values for every row of the Mueller matrix column
    K::parallel_for(
      K::TeamVectorRange(team_member, N_Y),
      [=](const int Y) {
        phi_Y(Y) = vis.m_phi0[1] + Y * vis.m_dphi[1]; // Yth index in phi_Y (phase gradient) = phase screen value origin[1] + Y * phase screen value increment [1]
      });
    team_member.team_barrier();

    poln_array_type<visibility_fp, N> result; // Create array for all polarization products of a visibility value

    if (model.is_allocated()) {
      // model degridding
      static_assert(std::is_same_v<acc_vis_t, acc_cf_t>);
      vis_array_type<acc_vis_t::value_type, N> vis_array;
      // Create vis_array_type (all polarization products of visibility values and weights)
      // acc_vis_t::value_type is floating point type of values, N is num of polarizations

      // 3d (X, Y, Mueller) subspace of CF for this visibility
      // Degrid: we have Stokes value, convert to Feed thru Mueller matrix
      auto cf_vis =
        K::subview(
          cf,
          K::pair<int, int>(vis.m_cf_major[0], vis.m_cf_major[0] + N_X), // X axis: from CF major coordinate to CF size array
          K::pair<int, int>(vis.m_cf_major[1], vis.m_cf_major[1] + N_Y), // Y axis: from CF major coordinate to CF size array
          K::ALL,
          vis.m_cf_cube, // CF cube index (is this the Z/W coordinate?)
          vis.m_cf_minor[0], // X axis of CF minor coordinate
          vis.m_cf_minor[1]); // Y axis of CF minor coordinate

      // 3d (X, Y, pol) subspace of model for this visibility
      auto model_vis =
        K::subview(
          model,
          K::pair<int, int>(vis.m_grid_coord[0], vis.m_grid_coord[0] + N_X),
          K::pair<int, int>(vis.m_grid_coord[1], vis.m_grid_coord[1] + N_Y),
          K::ALL,
          vis.m_grid_cube); // Grid cube index

      // loop over model polarizations
      for (int gpol = 0; gpol < N_R; ++gpol) {
        decltype(vis_array) va; // Create vis_array va (?)
        // parallel loop over grid X
        K::parallel_reduce(
          K::TeamThreadRange(team_member, N_X),
          [=](const int X, decltype(vis_array)& vis_array_l) {
            auto phi_X = vis.m_phi0[0] + X * vis.m_dphi[0]; // phi_X (phase gradient) = phase screen value origin[0] + X * phase screen value increment [0]
            // loop over grid Y
            for (int Y = 0; Y < N_Y; ++Y) {
              auto screen = cphase<execution_space>(-phi_X - phi_Y(Y)); // screen = complex value conversion of phase (-phi_X - phy_Y(Y))
              const auto mv = model_vis(X, Y, gpol) * screen; // Access (X, Y, gpol)'th element, multiply by screen (this undoes the correction)
              // loop over visibility polarizations
              for (int vpol = 0; vpol < N; ++vpol) {
                if (const auto mindex = degridding_mindex(gpol, vpol); // IDK
                    mindex >= 0) {
                  cf_t cfv = cf_vis(X, Y, mindex); // conv. func. cfv = (X,Y,mindex)th element of cf_vis
                  cfv.imag() *= cf_im_factor; // Im(cfv) = Im(cfv) * cf_im_factor
                  vis_array_l.vis[vpol] += cfv * mv;
                  vis_array_l.wgt[vpol] += cfv;
                }
              }
            }
          },
          K::Sum<decltype(va)>(va)); // return sum as the same type as va
        vis_array += va;
      }

      // apply weights and phasor to compute predicted visibilities
      auto conj_phasor = vis.m_phasor;
      conj_phasor.imag() *= -1;
      for (int vpol = 0; vpol < N; ++vpol)
        result.vals[vpol] =
          (vis_array.vis[vpol]
           / ((vis_array.wgt[vpol] != (acc_cf_t)0) // if weight != (acc_cf_t) 0, then use weight. Else, set weight = (acc_cf_t) 1, then use.
              ? vis_array.wgt[vpol]
              : (acc_cf_t)1))
          * conj_phasor;
    }
    return result;
  }

  // function for gridding a single visibility with sum of weights
  template <typename cf_layout, typename grid_layout, typename memory_space>
  static KOKKOS_FUNCTION void
  grid_vis(
    const member_type& team_member,
    const Vis<N, execution_space>& vis,
    const unsigned gpol,
    const cf_view<cf_layout, memory_space>& cf,
    const const_mindex_view<memory_space>& mueller_indexes,
    const const_mindex_view<memory_space>& conjugate_mueller_indexes,
    const grid_view<grid_layout, memory_space>& grid,
    const weight_view<typename execution_space::array_layout, memory_space>&
    weights,
    const scratch_phscr_view& phi_Y) {

    std::cout << "Using VisibilityGridder case 0 grid_vis" << std::endl;

    const auto& N_X = vis.m_cf_size[0]; // Number of pixels along X/U dimension
    const auto& N_Y = vis.m_cf_size[1]; // Number of pixels along Y/V dimension

    auto gridding_mindex =
      K::subview(
        (vis.m_pos_w ? mueller_indexes : conjugate_mueller_indexes),
        gpol,
        K::ALL);
    // Create subview of either Mueller indexes or their conjugates (depending on W coordinate)
    // Take slice at gpol-th index (the given grid polarization)
    // Subview contains entire extent of the remaining dimension

    cf_fp cf_im_factor = (vis.m_pos_w ? -1 : 1); // If W. pos., -1. Else, 1.

    // compute the values of the phase screen along the Y axis now and store the
    // results in scratch memory because gridding on the Y axis accesses the
    // phase screen values for every row of the Mueller matrix column
    K::parallel_for(
      K::TeamVectorRange(team_member, N_Y),
      [=](const int Y) {
        phi_Y(Y) = vis.m_phi0[1] + Y * vis.m_dphi[1]; // phase gradient (Y) = phase screen origin value + Y * phase screen increment
      });
    team_member.team_barrier(); // Stop each thread that finishes until all other threads catch up (?)

    // 3d (X, Y, Mueller) subspace of CF for this visibility
    auto cf_vis =
      K::subview(
        cf,
        K::pair<int, int>(vis.m_cf_major[0], vis.m_cf_major[0] + N_X),
        K::pair<int, int>(vis.m_cf_major[1], vis.m_cf_major[1] + N_Y),
        K::ALL,
        vis.m_cf_cube,
        vis.m_cf_minor[0],
        vis.m_cf_minor[1]);
    // Create a 3D subview of cf (convolution function types)
    // range: (From X coord of major CF until end of axis, same for Y, all Mueller indexes(?), CF cube index (W/Z), X coord of minor CF, Y coord of minor CF)

    // 2d (X, Y) subspace of grid for this visibility and grid polarization
    // (gpol)
    auto grd_vis =
      K::subview(
        grid,
        K::pair<int, int>(vis.m_grid_coord[0], vis.m_grid_coord[0] + N_X),
        K::pair<int, int>(vis.m_grid_coord[1], vis.m_grid_coord[1] + N_Y),
        gpol,
        vis.m_grid_cube);

    // accumulate to grid, and CF weights per visibility polarization
    poln_array_type<acc_cf_t::value_type, N> grid_wgt;
    // parallel loop over grid X
    K::parallel_reduce(
      K::TeamThreadRange(team_member, N_X),
      [=](const int X, decltype(grid_wgt)& grid_wgt_l) {
        auto phi_X = vis.m_phi0[0] + X * vis.m_dphi[0]; // phase gradient
        // loop over grid Y
        for (int Y = 0; Y < N_Y; ++Y) {
          const cf_t screen = cphase<execution_space>(phi_X + phi_Y(Y)); // complex conversion of phase gradient
          gv_t gv(0);
          // loop over visibility polarizations
          for (int vpol = 0; vpol < N; ++vpol) {
            if (const auto mindex = gridding_mindex(vpol); mindex >= 0) { // if vpol'th Mueller index/conjugate >= 0
              cf_t cfv = cf_vis(X, Y, mindex);
              cfv.imag() *= cf_im_factor;
              gv += gv_t(cfv * screen * vis.m_values[vpol]); // gv_t gv = (X,Y,mindex)'th index of subspace of CF for this vis * phase screen * vis for this pol
              grid_wgt_l.vals[vpol] += cfv; // weight for this vis pol = (X,Y,mindex)'th index of subspace of CF for this vis (never used again?)
            }
          }
          pseudo_atomic_add<execution_space>(grd_vis(X, Y), gv); // add gv to grd_vis(X,Y)
        }
      },
      K::Sum<decltype(grid_wgt)>(grid_wgt));
    // compute final weight and add it to weights
    K::single(
      K::PerTeam(team_member), // restricts lambda to execute once per team
      [&]() { // initialize as reference
        grid_value_fp twgt = 0;
        for (int vpol = 0; vpol < N; ++vpol)
          twgt +=
            grid_value_fp(mag(grid_wgt.vals[vpol]) * vis.m_weights[vpol]); // magnitude of complex vpol'th value of grid weights * vpol'th value of vis weights
        K::atomic_add(&weights(gpol, vis.m_grid_cube), twgt); // add total weight to weights(grid polarization, grid cube index)
      });
  }

  // function for gridding a single visibility without sum of weights
  template <typename cf_layout, typename grid_layout, typename memory_space>
  static KOKKOS_FUNCTION void
  grid_vis_no_weights(
    const member_type& team_member,
    const Vis<N, execution_space>& vis,
    const unsigned gpol,
    const cf_view<cf_layout, memory_space>& cf,
    const const_mindex_view<memory_space>& mueller_indexes,
    const const_mindex_view<memory_space>& conjugate_mueller_indexes,
    const grid_view<grid_layout, memory_space>& grid,
    const scratch_phscr_view& phi_Y) {

    const auto& N_X = vis.m_cf_size[0];
    const auto& N_Y = vis.m_cf_size[1];

    auto gridding_mindex =
      K::subview(
        (vis.m_pos_w ? mueller_indexes : conjugate_mueller_indexes),
        gpol,
        K::ALL);
    cf_fp cf_im_factor = (vis.m_pos_w ? -1 : 1);

    // compute the values of the phase screen along the Y axis now and store the
    // results in scratch memory because gridding on the Y axis accesses the
    // phase screen values for every row of the Mueller matrix column
    K::parallel_for(
      K::TeamVectorRange(team_member, N_Y),
      [=](const int Y) {
        phi_Y(Y) = vis.m_phi0[1] + Y * vis.m_dphi[1];
      });
    team_member.team_barrier();

    // 3d (X, Y, Mueller) subspace of CF for this visibility
    auto cf_vis =
      K::subview(
        cf,
        K::pair<int, int>(vis.m_cf_major[0], vis.m_cf_major[0] + N_X),
        K::pair<int, int>(vis.m_cf_major[1], vis.m_cf_major[1] + N_Y),
        K::ALL,
        vis.m_cf_cube,
        vis.m_cf_minor[0],
        vis.m_cf_minor[1]);

    // 2d (X, Y) subspace of grid for this visibility and grid polarization
    // (gpol)
    auto grd_vis =
      K::subview(
        grid,
        K::pair<int, int>(vis.m_grid_coord[0], vis.m_grid_coord[0] + N_X),
        K::pair<int, int>(vis.m_grid_coord[1], vis.m_grid_coord[1] + N_Y),
        gpol,
        vis.m_grid_cube);

    // parallel loop over grid X
    K::parallel_for(
      K::TeamThreadRange(team_member, N_X),
      [=](const int X) {
        auto phi_X = vis.m_phi0[0] + X * vis.m_dphi[0];
        // loop over grid Y
        for (int Y = 0; Y < N_Y; ++Y) {
          const cf_t screen = cphase<execution_space>(phi_X + phi_Y(Y));
          gv_t gv(0);
          // loop over visibility polarizations
          for (int vpol = 0; vpol < N; ++vpol) {
            if (const auto mindex = gridding_mindex(vpol); mindex >= 0) {
              cf_t cfv = cf_vis(X, Y, mindex);
              cfv.imag() *= cf_im_factor;
              gv += gv_t(cfv * screen * vis.m_values[vpol]);
            }
          }
          pseudo_atomic_add<execution_space>(grd_vis(X, Y), gv);
        }
      });
  }

  static KOKKOS_INLINE_FUNCTION bool
  all_within_grid(
    const Vis<N, execution_space>& vis,
    const K::Array<int, 2>& grid_size) {

    return
      (0 <= vis.m_grid_coord[0])
      && (vis.m_grid_coord[0] + vis.m_cf_size[0] <= grid_size[0])
      && (0 <= vis.m_grid_coord[1])
      && (vis.m_grid_coord[1] + vis.m_cf_size[1] <= grid_size[1]);
  }

  template <typename cf_layout, typename grid_layout, typename memory_space>
  static void
  kernel(
    execution_space exec,
    const K::Array<cf_view<cf_layout, memory_space>, HPG_MAX_NUM_CF_GROUPS>&
      cfs,
    const K::Array<K::Array<int, 2>, HPG_MAX_NUM_CF_GROUPS>& cf_radii,
    unsigned max_cf_extent_y,
    const_mindex_view<memory_space> mueller_indexes,
    const_mindex_view<memory_space> conjugate_mueller_indexes,
    bool update_grid_weights,
    bool do_degrid,
    bool do_grid,
    int num_visibilities,
    visdata_view<N, memory_space> visibilities,
    gvisbuff_view<memory_space>& gvisbuff,
    const K::Array<grid_scale_fp, 2>& grid_scale,
    const const_grid_view<grid_layout, memory_space>& model,
    const grid_view<grid_layout, memory_space>& grid,
    const weight_view<typename execution_space::array_layout, memory_space>&
      weights) {

    ProfileRegion region("VisibilityGridder");

    const K::Array<int, 2>
      grid_size{
      grid.extent_int(int(GridAxis::x)),
      grid.extent_int(int(GridAxis::y))};
    const K::Array<int, 2>
      oversampling{
      cfs[0].extent_int(int(CFAxis::x_minor)),
      cfs[0].extent_int(int(CFAxis::y_minor))};

    auto shmem_size = scratch_phscr_view::shmem_size(max_cf_extent_y);

    if (do_degrid) {
      K::parallel_for(
        "degridding",
        K::TeamPolicy<execution_space>(exec, num_visibilities, K::AUTO)
        .set_scratch_size(0, K::PerTeam(shmem_size)),
        KOKKOS_LAMBDA(const member_type& team_member) {
          auto i = team_member.league_rank();
          auto& visibility = visibilities(i);

          Vis<N, execution_space> vis(
            visibility,
            visibility.m_values,
            grid_size,
            grid_scale,
            cf_radii,
            oversampling);
          poln_array_type<visibility_fp, N> gvis;
          // skip this visibility if all of the updated grid points are not
          // within grid bounds
          if (all_within_grid(vis, grid_size)) {
            gvis =
              degrid_vis<cf_layout, grid_layout, memory_space>(
                team_member,
                vis,
                cfs[vis.m_cf_grp],
                mueller_indexes,
                conjugate_mueller_indexes,
                model,
                scratch_phscr_view(
                  team_member.team_scratch(0),
                  max_cf_extent_y));
            if (do_grid)
              // return residual visibilities, prepare values for gridding
              K::single(
                K::PerTeam(team_member),
                [&]() {
                  for (int vpol = 0; vpol < N; ++vpol) {
                    visibility.m_values[vpol] -= gvis.vals[vpol];
                    gvisbuff(i).vals[vpol] =
                      visibility.m_values[vpol]
                      * vis.m_phasor
                      * vis.m_weights[vpol];
                  }
                });
            else
              // return predicted visibilities
              K::single(
                K::PerTeam(team_member),
                [&]() {
                  for (int vpol = 0; vpol < N; ++vpol)
                    visibility.m_values[vpol] = gvis.vals[vpol];
                });
          }
        });
    } else {
      K::parallel_for(
        "gvis_init",
        K::RangePolicy<execution_space>(exec, 0, num_visibilities),
        KOKKOS_LAMBDA(const int i) {
          auto& vis = visibilities(i);
          auto phasor = cphase<execution_space>(vis.m_d_phase);
          for (int vpol = 0; vpol < N; ++vpol) {
            gvisbuff(i).vals[vpol] =
              vis.m_values[vpol] * phasor * vis.m_weights[vpol];
          }
        });
    }

    if (do_grid) {
      const auto N_R = grid.extent_int(int(GridAxis::mrow));
      if (update_grid_weights)
        K::parallel_for(
          "gridding",
          K::TeamPolicy<execution_space>(exec, N_R * num_visibilities, K::AUTO)
          .set_scratch_size(0, K::PerTeam(shmem_size)),
          KOKKOS_LAMBDA(const member_type& team_member) {
            auto i = team_member.league_rank() / N_R;
            auto gpol = team_member.league_rank() % N_R;

            Vis<N, execution_space> vis(
              visibilities(i),
              reinterpret_cast<K::Array<vis_t, N>&>(gvisbuff(i).vals),
              grid_size,
              grid_scale,
              cf_radii,
              oversampling);
            // skip this visibility if all of the updated grid points are not
            // within grid bounds
            if (all_within_grid(vis, grid_size)) {
              grid_vis<cf_layout, grid_layout, memory_space>(
                team_member,
                vis,
                gpol,
                cfs[vis.m_cf_grp],
                mueller_indexes,
                conjugate_mueller_indexes,
                grid,
                weights,
                scratch_phscr_view(
                  team_member.team_scratch(0),
                  max_cf_extent_y));
            }
          });
      else
        K::parallel_for(
          "gridding_no_weights",
          K::TeamPolicy<execution_space>(exec, N_R * num_visibilities, K::AUTO)
          .set_scratch_size(0, K::PerTeam(shmem_size)),
          KOKKOS_LAMBDA(const member_type& team_member) {
            auto i = team_member.league_rank() / N_R;
            auto gpol = team_member.league_rank() % N_R;

            Vis<N, execution_space> vis(
              visibilities(i),
              reinterpret_cast<K::Array<vis_t, N>&>(gvisbuff(i).vals),
              grid_size,
              grid_scale,
              cf_radii,
              oversampling);
            // skip this visibility if all of the updated grid points are not
            // within grid bounds
            if (all_within_grid(vis, grid_size)) {
              grid_vis_no_weights<cf_layout, grid_layout, memory_space>(
                team_member,
                vis,
                gpol,
                cfs[vis.m_cf_grp],
                mueller_indexes,
                conjugate_mueller_indexes,
                grid,
                scratch_phscr_view(
                  team_member.team_scratch(0),
                  max_cf_extent_y));
            }
          });
    }
  }
};

//VisibilityGridder version 2 used for mean_grid
template <int N, typename execution_space>
struct /*HPG_EXPORT*/ VisibilityGridder<N, execution_space, 2> final {

  using member_type = typename K::TeamPolicy<execution_space>::member_type;

  using scratch_phscr_view =
    K::View<
      cf_phase_gradient_fp*,
    typename execution_space::scratch_memory_space>;

  template <typename cf_layout, typename grid_layout, typename memory_space>
  static KOKKOS_FUNCTION poln_array_type<visibility_fp, N>
  degrid_vis(
    // gv_t = complex gridded value type
    // cf_t = convolution function value type
    const member_type& team_member, //using member_type = typename K::TeamPolicy<execution_space>::member_type
    const Vis<N, execution_space>& vis, //line 495, struct Vis
    const cf_view<cf_layout, memory_space>& cf, //using cf_view = K::View<cf_t******, Layout, memory_space, K::MemoryTraits<K::Unmanaged>>;
    const const_mindex_view<memory_space>& mueller_indexes, //using const_mindex_view = K::View<const int[4][4], memory_space, K::MemoryTraits<K::RandomAccess>>;
    const const_mindex_view<memory_space>& conjugate_mueller_indexes,
    const const_grid_view<grid_layout, memory_space>& model, //using const_grid_view = K::View<const gv_t****, Layout, memory_space>;
    const scratch_phscr_view& phi_Y) { // using scratch_phscr_view = K::View<cf_phase_gradient_fp*, typename execution_space::scratch_memory_space>;

    std::cout << "Using VisibilityGridder case 2 degrid_vis" << std::endl;

    const auto& N_X = vis.m_cf_size[0]; // first index of cf_size array (how many pixels along u dim.)
    const auto& N_Y = vis.m_cf_size[1]; // second index of cf_size array (along v dim.)
    const auto N_R = model.extent_int(int(GridAxis::mrow)); // Number of elements in mrow of GridAxis?

    // vis.m_pos_w = true iff W coordinate is strictly positive
    // mindex is index of the Mueller matrix
    auto degridding_mindex =
      vis.m_pos_w ? conjugate_mueller_indexes : mueller_indexes;
    cf_fp cf_im_factor = (vis.m_pos_w ? 1 : -1);

    // compute the values of the phase screen along the Y axis now and store the
    // results in scratch memory because gridding on the Y axis accesses the
    // phase screen values for every row of the Mueller matrix column
    K::parallel_for(
      K::TeamVectorRange(team_member, N_Y),
      [=](const int Y) {
        phi_Y(Y) = vis.m_phi0[1] + Y * vis.m_dphi[1]; // Yth index in phi_Y (phase gradient) = phase screen value origin[1] + Y * phase screen value increment [1]
      });
    team_member.team_barrier();

    poln_array_type<visibility_fp, N> result; // Create array for all polarization products of a visibility value

    if (model.is_allocated()) {
      // model degridding
      static_assert(std::is_same_v<acc_vis_t, acc_cf_t>);
      vis_array_type<acc_vis_t::value_type, N> vis_array;
      // Create vis_array_type (all polarization products of visibility values and weights)
      // acc_vis_t::value_type is floating point type of values, N is num of polarizations

      // 3d (X, Y, Mueller) subspace of CF for this visibility
      // Degrid: we have Stokes value, convert to Feed thru Mueller matrix
      auto cf_vis =
        K::subview(
          cf,
          K::pair<int, int>(vis.m_cf_major[0], vis.m_cf_major[0] + N_X), // X axis: from CF major coordinate to CF size array
          K::pair<int, int>(vis.m_cf_major[1], vis.m_cf_major[1] + N_Y), // Y axis: from CF major coordinate to CF size array
          K::ALL,
          vis.m_cf_cube, // CF cube index (is this the Z/W coordinate?)
          vis.m_cf_minor[0], // X axis of CF minor coordinate
          vis.m_cf_minor[1]); // Y axis of CF minor coordinate

      // 3d (X, Y, pol) subspace of model for this visibility
      auto model_vis =
        K::subview(
          model,
          K::pair<int, int>(vis.m_grid_coord[0], vis.m_grid_coord[0] + N_X),
          K::pair<int, int>(vis.m_grid_coord[1], vis.m_grid_coord[1] + N_Y),
          K::ALL,
          vis.m_grid_cube); // Grid cube index

      // loop over model polarizations
      for (int gpol = 0; gpol < N_R; ++gpol) {
        decltype(vis_array) va; // Create vis_array va (?)
        // parallel loop over grid X
        K::parallel_reduce(
          K::TeamThreadRange(team_member, N_X),
          [=](const int X, decltype(vis_array)& vis_array_l) {
            auto phi_X = vis.m_phi0[0] + X * vis.m_dphi[0]; // phi_X (phase gradient) = phase screen value origin[0] + X * phase screen value increment [0]
            // loop over grid Y
            for (int Y = 0; Y < N_Y; ++Y) {
              auto screen = cphase<execution_space>(-phi_X - phi_Y(Y)); // screen = complex value conversion of phase (-phi_X - phy_Y(Y))
              const auto mv = model_vis(X, Y, gpol) * screen; // Access (X, Y, gpol)'th element, multiply by screen (this undoes the correction)
              // loop over visibility polarizations
              for (int vpol = 0; vpol < N; ++vpol) {
                if (const auto mindex = degridding_mindex(gpol, vpol); // IDK
                    mindex >= 0) {
                  cf_t cfv = cf_vis(X, Y, mindex); // conv. func. cfv = (X,Y,mindex)th element of cf_vis
                  cfv.imag() *= cf_im_factor; // Im(cfv) = Im(cfv) * cf_im_factor
                  vis_array_l.vis[vpol] += cfv * mv;
                  vis_array_l.wgt[vpol] += cfv;
                }
              }
            }
          },
          K::Sum<decltype(va)>(va)); // return sum as the same type as va
        vis_array += va;
      }

      // apply weights and phasor to compute predicted visibilities
      auto conj_phasor = vis.m_phasor;
      conj_phasor.imag() *= -1;
      for (int vpol = 0; vpol < N; ++vpol)
        result.vals[vpol] =
          (vis_array.vis[vpol]
           / ((vis_array.wgt[vpol] != (acc_cf_t)0) // if weight != (acc_cf_t) 0, then use weight. Else, set weight = (acc_cf_t) 1, then use.
              ? vis_array.wgt[vpol]
              : (acc_cf_t)1))
          * conj_phasor;
    }
    return result;
  }

  // function for gridding a single visibility with sum of weights
  template <typename cf_layout, typename grid_layout, typename memory_space>
  static KOKKOS_FUNCTION void
  grid_vis(
    const member_type& team_member,
    const Vis<N, execution_space>& vis,
    const unsigned gpol,
    const cf_view<cf_layout, memory_space>& cf,
    const const_mindex_view<memory_space>& mueller_indexes,
    const const_mindex_view<memory_space>& conjugate_mueller_indexes,
    const grid_view<grid_layout, memory_space>& grid,
    const weight_view<typename execution_space::array_layout, memory_space>&
    weights,
    const scratch_phscr_view& phi_Y) {

    std::cout << "Using VisibilityGridder case 2 grid_vis" << std::endl;

    const auto& N_X = vis.m_cf_size[0]; // Number of pixels along X/U dimension
    const auto& N_Y = vis.m_cf_size[1]; // Number of pixels along Y/V dimension

    auto gridding_mindex =
      K::subview(
        (vis.m_pos_w ? mueller_indexes : conjugate_mueller_indexes),
        gpol,
        K::ALL);
    // Create subview of either Mueller indexes or their conjugates (depending on W coordinate)
    // Take slice at gpol-th index (the given grid polarization)
    // Subview contains entire extent of the remaining dimension

    cf_fp cf_im_factor = (vis.m_pos_w ? -1 : 1); // If W. pos., -1. Else, 1.

    // compute the values of the phase screen along the Y axis now and store the
    // results in scratch memory because gridding on the Y axis accesses the
    // phase screen values for every row of the Mueller matrix column
    K::parallel_for(
      K::TeamVectorRange(team_member, N_Y),
      [=](const int Y) {
        phi_Y(Y) = vis.m_phi0[1] + Y * vis.m_dphi[1]; // phase gradient (Y) = phase screen origin value + Y * phase screen increment
      });
    team_member.team_barrier(); // Stop each thread that finishes until all other threads catch up (?)

    // 3d (X, Y, Mueller) subspace of CF for this visibility
    auto cf_vis =
      K::subview(
        cf,
        K::pair<int, int>(vis.m_cf_major[0], vis.m_cf_major[0] + N_X),
        K::pair<int, int>(vis.m_cf_major[1], vis.m_cf_major[1] + N_Y),
        K::ALL,
        vis.m_cf_cube,
        vis.m_cf_minor[0],
        vis.m_cf_minor[1]);
    // Create a 3D subview of cf (convolution function types)
    // range: (From X coord of major CF until end of axis, same for Y, all Mueller indexes(?), CF cube index (W/Z), X coord of minor CF, Y coord of minor CF)

    // 2d (X, Y) subspace of grid for this visibility and grid polarization
    // (gpol)
    auto grd_vis =
      K::subview(
        grid,
        K::pair<int, int>(vis.m_grid_coord[0], vis.m_grid_coord[0] + N_X),
        K::pair<int, int>(vis.m_grid_coord[1], vis.m_grid_coord[1] + N_Y),
        gpol,
        vis.m_grid_cube);

    // accumulate to grid, and CF weights per visibility polarization
    poln_array_type<acc_cf_t::value_type, N> grid_wgt;
    // parallel loop over grid X
    K::parallel_reduce(
      K::TeamThreadRange(team_member, N_X),
      [=](const int X, decltype(grid_wgt)& grid_wgt_l) {
        auto phi_X = vis.m_phi0[0] + X * vis.m_dphi[0]; // phase gradient
        // loop over grid Y
        for (int Y = 0; Y < N_Y; ++Y) {
          const cf_t screen = cphase<execution_space>(phi_X + phi_Y(Y)); // complex conversion of phase gradient
          gv_t gv(0);
          // loop over visibility polarizations
          for (int vpol = 0; vpol < N; ++vpol) {
            if (const auto mindex = gridding_mindex(vpol); mindex >= 0) { // if vpol'th Mueller index/conjugate >= 0
              cf_t cfv = cf_vis(X, Y, mindex);
              cfv.imag() *= cf_im_factor;
              gv += gv_t(cfv * screen * vis.m_values[vpol]); // gv_t gv = (X,Y,mindex)'th index of subspace of CF for this vis * phase screen * vis for this pol
              grid_wgt_l.vals[vpol] += cfv; // weight for this vis pol = (X,Y,mindex)'th index of subspace of CF for this vis (never used again?)
            }
          }
          pseudo_atomic_add<execution_space>(grd_vis(X, Y), gv); // add gv to grd_vis(X,Y)
        }
      },
      K::Sum<decltype(grid_wgt)>(grid_wgt));
    // compute final weight and add it to weights
    K::single(
      K::PerTeam(team_member), // restricts lambda to execute once per team
      [&]() { // initialize as reference
        grid_value_fp twgt = 0;
        for (int vpol = 0; vpol < N; ++vpol)
          twgt +=
            grid_value_fp(mag(grid_wgt.vals[vpol]) * vis.m_weights[vpol]); // magnitude of complex vpol'th value of grid weights * vpol'th value of vis weights
        K::atomic_add(&weights(gpol, vis.m_grid_cube), twgt); // add total weight to weights(grid polarization, grid cube index)
      });
  }

  // degrid_vis clone with weighted mean
  template <typename cf_layout, typename grid_layout, typename memory_space>
  static KOKKOS_FUNCTION poln_array_type<visibility_fp, N>
  degrid_vis_weighted_mean(
    // gv_t = complex gridded value type
    // cf_t = convolution function value type
    const member_type& team_member, //using member_type = typename K::TeamPolicy<execution_space>::member_type
    const Vis<N, execution_space>& vis, //line 495, struct Vis
    const cf_view<cf_layout, memory_space>& cf, //using cf_view = K::View<cf_t******, Layout, memory_space, K::MemoryTraits<K::Unmanaged>>;
    const const_mindex_view<memory_space>& mueller_indexes, //using const_mindex_view = K::View<const int[4][4], memory_space, K::MemoryTraits<K::RandomAccess>>;
    const const_mindex_view<memory_space>& conjugate_mueller_indexes,
    const const_grid_view<grid_layout, memory_space>& model, //using const_grid_view = K::View<const gv_t****, Layout, memory_space>;
    const grid_view<grid_layout, memory_space>& mean_grid,
    const scratch_phscr_view& phi_Y) { // using scratch_phscr_view = K::View<cf_phase_gradient_fp*, typename execution_space::scratch_memory_space>;

    std::cout << "degrid_vis_weighted_mean in visibilitygridder 2" << std::endl;

    const auto& N_X = vis.m_cf_size[0]; // first index of cf_size array (how many pixels along u dim.)
    const auto& N_Y = vis.m_cf_size[1]; // second index of cf_size array (along v dim.)
    const auto N_R = model.extent_int(int(GridAxis::mrow)); // Number of elements in mrow of GridAxis?

    // vis.m_pos_w = true iff W coordinate is strictly positive
    // mindex is index of the Mueller matrix
    auto degridding_mindex =
      vis.m_pos_w ? conjugate_mueller_indexes : mueller_indexes;
    cf_fp cf_im_factor = (vis.m_pos_w ? 1 : -1);

    // compute the values of the phase screen along the Y axis now and store the
    // results in scratch memory because gridding on the Y axis accesses the
    // phase screen values for every row of the Mueller matrix column
    K::parallel_for(
      K::TeamVectorRange(team_member, N_Y),
      [=](const int Y) {
        phi_Y(Y) = vis.m_phi0[1] + Y * vis.m_dphi[1]; // Yth index in phi_Y (phase gradient) = phase screen value origin[1] + Y * phase screen value increment [1]
      });
    team_member.team_barrier();

    poln_array_type<visibility_fp, N> result; // Create array for all polarization products of a visibility value

    if (model.is_allocated()) {
      // model degridding
      static_assert(std::is_same_v<acc_vis_t, acc_cf_t>);
      vis_array_type<acc_vis_t::value_type, N> vis_array;
      // Create vis_array_type (all polarization products of visibility values and weights)
      // acc_vis_t::value_type is floating point type of values, N is num of polarizations

      // 3d (X, Y, Mueller) subspace of CF for this visibility
      // Degrid: we have Stokes value, convert to Feed thru Mueller matrix
      auto cf_vis =
        K::subview(
          cf,
          K::pair<int, int>(vis.m_cf_major[0], vis.m_cf_major[0] + N_X), // X axis: from CF major coordinate to CF size array
          K::pair<int, int>(vis.m_cf_major[1], vis.m_cf_major[1] + N_Y), // Y axis: from CF major coordinate to CF size array
          K::ALL,
          vis.m_cf_cube, // CF cube index (is this the Z/W coordinate?)
          vis.m_cf_minor[0], // X axis of CF minor coordinate
          vis.m_cf_minor[1]); // Y axis of CF minor coordinate

      // 3d (X, Y, pol) subspace of model for this visibility
      auto model_vis =
        K::subview(
          model,
          K::pair<int, int>(vis.m_grid_coord[0], vis.m_grid_coord[0] + N_X),
          K::pair<int, int>(vis.m_grid_coord[1], vis.m_grid_coord[1] + N_Y),
          K::ALL,
          vis.m_grid_cube); // Grid cube index

      // loop over model polarizations
      for (int gpol = 0; gpol < N_R; ++gpol) {
        decltype(vis_array) va; // Create vis_array va (?)
        // parallel loop over grid X
        K::parallel_reduce(
          K::TeamThreadRange(team_member, N_X),
          [=](const int X, decltype(vis_array)& vis_array_l) {
            auto phi_X = vis.m_phi0[0] + X * vis.m_dphi[0]; // phi_X (phase gradient) = phase screen value origin[0] + X * phase screen value increment [0]
            // loop over grid Y
            for (int Y = 0; Y < N_Y; ++Y) {
              auto screen = cphase<execution_space>(-phi_X - phi_Y(Y)); // screen = complex value conversion of phase (-phi_X - phy_Y(Y))
              const auto mv = model_vis(X, Y, gpol) * screen; // Access (X, Y, gpol)'th element, multiply by screen (this undoes the correction)
              // loop over visibility polarizations
              for (int vpol = 0; vpol < N; ++vpol) {
                if (const auto mindex = degridding_mindex(gpol, vpol); // IDK
                    mindex >= 0) {
                  cf_t cfv = cf_vis(X, Y, mindex); // conv. func. cfv = (X,Y,mindex)th element of cf_vis
                  cfv.imag() *= cf_im_factor; // Im(cfv) = Im(cfv) * cf_im_factor
                  vis_array_l.vis[vpol] += cfv * mv;
                  vis_array_l.wgt[vpol] += cfv;
                }
              }
            }
          },
          K::Sum<decltype(va)>(va)); // return sum as the same type as va
        vis_array += va;
      }

      // apply weights and phasor to compute predicted visibilities
      auto conj_phasor = vis.m_phasor;
      conj_phasor.imag() *= -1;
      for (int vpol = 0; vpol < N; ++vpol)
        result.vals[vpol] =
          (vis_array.vis[vpol]
           / ((vis_array.wgt[vpol] != (acc_cf_t)0) // if weight != (acc_cf_t) 0, then use weight. Else, set weight = (acc_cf_t) 1, then use.
              ? vis_array.wgt[vpol]
              : (acc_cf_t)1))
          * conj_phasor;
    }
    return result;
  }
  
  template <typename cf_layout, typename grid_layout, typename memory_space>
  static KOKKOS_FUNCTION void
  grid_vis_weighted_mean(
    const member_type& team_member,
    const Vis<N, execution_space>& vis,
    const unsigned gpol,
    const cf_view<cf_layout, memory_space>& cf,
    const const_mindex_view<memory_space>& mueller_indexes,
    const const_mindex_view<memory_space>& conjugate_mueller_indexes,
    const grid_view<grid_layout, memory_space>& grid,
    const grid_view<grid_layout, memory_space>& mean_grid,
    const weight_view<typename execution_space::array_layout, memory_space>&
    weights,
    const scratch_phscr_view& phi_Y) {

    std::cout << "grid_vis_weighted_mean in visibilitygridder 2" << std::endl;

    const auto& N_X = vis.m_cf_size[0]; // Number of pixels along X/U dimension
    const auto& N_Y = vis.m_cf_size[1]; // Number of pixels along Y/V dimension

    auto gridding_mindex =
      K::subview(
        (vis.m_pos_w ? mueller_indexes : conjugate_mueller_indexes),
        gpol,
        K::ALL);
    // Create subview of either Mueller indexes or their conjugates (depending on W coordinate)
    // Take slice at gpol-th index (the given grid polarization)
    // Subview contains entire extent of the remaining dimension

    cf_fp cf_im_factor = (vis.m_pos_w ? -1 : 1); // If W. pos., -1. Else, 1.

    std::cout << "grid_vis_weighted_mean in visibilitygridder 2 outside parallel_for" << std::endl;

    // compute the values of the phase screen along the Y axis now and store the
    // results in scratch memory because gridding on the Y axis accesses the
    // phase screen values for every row of the Mueller matrix column
    K::parallel_for(
      K::TeamVectorRange(team_member, N_Y),
      [=](const int Y) {
        phi_Y(Y) = vis.m_phi0[1] + Y * vis.m_dphi[1]; // phase gradient (Y) = phase screen origin value + Y * phase screen increment
      });
    team_member.team_barrier(); // Stop each thread that finishes until all other threads catch up (?)

    // 3d (X, Y, Mueller) subspace of CF for this visibility
    auto cf_vis =
      K::subview(
        cf,
        K::pair<int, int>(vis.m_cf_major[0], vis.m_cf_major[0] + N_X),
        K::pair<int, int>(vis.m_cf_major[1], vis.m_cf_major[1] + N_Y),
        K::ALL,
        vis.m_cf_cube,
        vis.m_cf_minor[0],
        vis.m_cf_minor[1]);
    // Create a 3D subview of cf (convolution function types)
    // range: (From X coord of major CF until end of axis, same for Y, all Mueller indexes(?), CF cube index (W/Z), X coord of minor CF, Y coord of minor CF)

    // 2d (X, Y) subspace of grid for this visibility and grid polarization
    // (gpol)
    auto grd_vis =
      K::subview(
        grid,
        K::pair<int, int>(vis.m_grid_coord[0], vis.m_grid_coord[0] + N_X),
        K::pair<int, int>(vis.m_grid_coord[1], vis.m_grid_coord[1] + N_Y),
        gpol,
        vis.m_grid_cube);

    auto mean_grd_vis =
      K::subview(
        mean_grid,
        K::pair<int, int>(vis.m_grid_coord[0], vis.m_grid_coord[0] + N_X),
        K::pair<int, int>(vis.m_grid_coord[1], vis.m_grid_coord[1] + N_Y),
        gpol,
        vis.m_grid_cube);

    // accumulate to grid, and CF weights per visibility polarization
    poln_array_type<acc_cf_t::value_type, N> grid_wgt;
    // parallel loop over grid X
    K::parallel_reduce(
      K::TeamThreadRange(team_member, N_X),
      [=](const int X, decltype(grid_wgt)& grid_wgt_l) {
        auto phi_X = vis.m_phi0[0] + X * vis.m_dphi[0]; // phase gradient
        // loop over grid Y
        for (int Y = 0; Y < N_Y; ++Y) {
          const cf_t screen = cphase<execution_space>(phi_X + phi_Y(Y)); // complex conversion of phase gradient
          gv_t gv(0);
          // loop over visibility polarizations
          for (int vpol = 0; vpol < N; ++vpol) {
            if (const auto mindex = gridding_mindex(vpol); mindex >= 0) { // if vpol'th Mueller index/conjugate >= 0
              cf_t cfv = cf_vis(X, Y, mindex);
              cfv.imag() *= cf_im_factor;
              gv += gv_t(cfv * screen * vis.m_values[vpol]); // gv_t gv = (X,Y,mindex)'th index of subspace of CF for this vis * phase screen * vis for this pol
              grid_wgt_l.vals[vpol] += cfv; // weight for this vis pol = (X,Y,mindex)'th index of subspace of CF for this vis
            }
          }
          pseudo_atomic_add<execution_space>(grd_vis(X, Y), gv); // add gv to grd_vis(X,Y)
          pseudo_atomic_add<execution_space>(mean_grd_vis(X,Y), gv);
        }
      },
      K::Sum<decltype(grid_wgt)>(grid_wgt)); // add grid_wgt_l to grid_wgt?
    // compute final weight and add it to weights

    std::cout << "grid_vis_weighted_mean in visibilitygridder 2 outside single" << std::endl;
    K::single(
      K::PerTeam(team_member), // restricts lambda to execute once per team
      [&]() { // initialize as reference
        grid_value_fp twgt = 0;
        for (int vpol = 0; vpol < N; ++vpol)
          twgt +=
            grid_value_fp(mag(grid_wgt.vals[vpol]) * vis.m_weights[vpol]); // magnitude of complex vpol'th value of grid weights * vpol'th value of vis weights
        K::atomic_add(&weights(gpol, vis.m_grid_cube), twgt); // add total weight to weights(grid polarization, grid cube index)
      });
    
    // Weighted average (using both grid weights and vis weights--is this correct?) applied only to mean_grid
    K::parallel_for(
      K::TeamThreadRange(team_member, N_X),
      [=] (const int X) {
        for (int Y = 0; Y < N_Y; ++Y){
          mean_grd_vis(X, Y) /= weights(gpol, vis.m_grid_cube);
        }
      });
  }

  // function for gridding a single visibility without sum of weights
  template <typename cf_layout, typename grid_layout, typename memory_space>
  static KOKKOS_FUNCTION void
  grid_vis_no_weights(
    const member_type& team_member,
    const Vis<N, execution_space>& vis,
    const unsigned gpol,
    const cf_view<cf_layout, memory_space>& cf,
    const const_mindex_view<memory_space>& mueller_indexes,
    const const_mindex_view<memory_space>& conjugate_mueller_indexes,
    const grid_view<grid_layout, memory_space>& grid,
    const scratch_phscr_view& phi_Y) {

    std::cout << "grid_vis_no_weights in visibilitygridder 2" << std::endl;

    const auto& N_X = vis.m_cf_size[0];
    const auto& N_Y = vis.m_cf_size[1];

    auto gridding_mindex =
      K::subview(
        (vis.m_pos_w ? mueller_indexes : conjugate_mueller_indexes),
        gpol,
        K::ALL);
    cf_fp cf_im_factor = (vis.m_pos_w ? -1 : 1);

    // compute the values of the phase screen along the Y axis now and store the
    // results in scratch memory because gridding on the Y axis accesses the
    // phase screen values for every row of the Mueller matrix column
    K::parallel_for(
      K::TeamVectorRange(team_member, N_Y),
      [=](const int Y) {
        phi_Y(Y) = vis.m_phi0[1] + Y * vis.m_dphi[1];
      });
    team_member.team_barrier();

    // 3d (X, Y, Mueller) subspace of CF for this visibility
    auto cf_vis =
      K::subview(
        cf,
        K::pair<int, int>(vis.m_cf_major[0], vis.m_cf_major[0] + N_X),
        K::pair<int, int>(vis.m_cf_major[1], vis.m_cf_major[1] + N_Y),
        K::ALL,
        vis.m_cf_cube,
        vis.m_cf_minor[0],
        vis.m_cf_minor[1]);

    // 2d (X, Y) subspace of grid for this visibility and grid polarization
    // (gpol)
    auto grd_vis =
      K::subview(
        grid,
        K::pair<int, int>(vis.m_grid_coord[0], vis.m_grid_coord[0] + N_X),
        K::pair<int, int>(vis.m_grid_coord[1], vis.m_grid_coord[1] + N_Y),
        gpol,
        vis.m_grid_cube);

    // parallel loop over grid X
    K::parallel_for(
      K::TeamThreadRange(team_member, N_X),
      [=](const int X) {
        auto phi_X = vis.m_phi0[0] + X * vis.m_dphi[0];
        // loop over grid Y
        for (int Y = 0; Y < N_Y; ++Y) {
          const cf_t screen = cphase<execution_space>(phi_X + phi_Y(Y));
          gv_t gv(0);
          // loop over visibility polarizations
          for (int vpol = 0; vpol < N; ++vpol) {
            if (const auto mindex = gridding_mindex(vpol); mindex >= 0) {
              cf_t cfv = cf_vis(X, Y, mindex);
              cfv.imag() *= cf_im_factor;
              gv += gv_t(cfv * screen * vis.m_values[vpol]);
            }
          }
          pseudo_atomic_add<execution_space>(grd_vis(X, Y), gv);
        }
      });
  }

  static KOKKOS_INLINE_FUNCTION bool
  all_within_grid(
    const Vis<N, execution_space>& vis,
    const K::Array<int, 2>& grid_size) {

    return
      (0 <= vis.m_grid_coord[0])
      && (vis.m_grid_coord[0] + vis.m_cf_size[0] <= grid_size[0])
      && (0 <= vis.m_grid_coord[1])
      && (vis.m_grid_coord[1] + vis.m_cf_size[1] <= grid_size[1]);
  }

  template <typename cf_layout, typename grid_layout, typename memory_space>
  static void
  kernel(
    execution_space exec,
    const K::Array<cf_view<cf_layout, memory_space>, HPG_MAX_NUM_CF_GROUPS>&
      cfs,
    const K::Array<K::Array<int, 2>, HPG_MAX_NUM_CF_GROUPS>& cf_radii,
    unsigned max_cf_extent_y,
    const_mindex_view<memory_space> mueller_indexes,
    const_mindex_view<memory_space> conjugate_mueller_indexes,
    bool update_grid_weights,
    bool do_degrid,
    bool do_grid,
    int num_visibilities,
    visdata_view<N, memory_space> visibilities,
    gvisbuff_view<memory_space>& gvisbuff,
    const K::Array<grid_scale_fp, 2>& grid_scale,
    const const_grid_view<grid_layout, memory_space>& model,
    const grid_view<grid_layout, memory_space>& grid,
    const grid_view<grid_layout, memory_space>& mean_grid,
    const weight_view<typename execution_space::array_layout, memory_space>&
      weights) {

    ProfileRegion region("VisibilityGridder");

    const K::Array<int, 2>
      grid_size{
      grid.extent_int(int(GridAxis::x)),
      grid.extent_int(int(GridAxis::y))};
    const K::Array<int, 2>
      oversampling{
      cfs[0].extent_int(int(CFAxis::x_minor)),
      cfs[0].extent_int(int(CFAxis::y_minor))};

    auto shmem_size = scratch_phscr_view::shmem_size(max_cf_extent_y);

    if (do_degrid) {
      K::parallel_for(
        "degridding",
        K::TeamPolicy<execution_space>(exec, num_visibilities, K::AUTO)
        .set_scratch_size(0, K::PerTeam(shmem_size)),
        KOKKOS_LAMBDA(const member_type& team_member) {
          auto i = team_member.league_rank();
          auto& visibility = visibilities(i);

          Vis<N, execution_space> vis(
            visibility,
            visibility.m_values,
            grid_size,
            grid_scale,
            cf_radii,
            oversampling);
          poln_array_type<visibility_fp, N> gvis;
          // skip this visibility if all of the updated grid points are not
          // within grid bounds
          if (all_within_grid(vis, grid_size)) {
            gvis =
              degrid_vis<cf_layout, grid_layout, memory_space>(
                team_member,
                vis,
                cfs[vis.m_cf_grp],
                mueller_indexes,
                conjugate_mueller_indexes,
                model,
                scratch_phscr_view(
                  team_member.team_scratch(0),
                  max_cf_extent_y));
            if (do_grid)
              // return residual visibilities, prepare values for gridding
              K::single(
                K::PerTeam(team_member),
                [&]() {
                  for (int vpol = 0; vpol < N; ++vpol) {
                    visibility.m_values[vpol] -= gvis.vals[vpol];
                    gvisbuff(i).vals[vpol] =
                      visibility.m_values[vpol]
                      * vis.m_phasor
                      * vis.m_weights[vpol];
                  }
                });
            else
              // return predicted visibilities
              K::single(
                K::PerTeam(team_member),
                [&]() {
                  for (int vpol = 0; vpol < N; ++vpol)
                    visibility.m_values[vpol] = gvis.vals[vpol];
                });
          }
        });
    } else {
      K::parallel_for(
        "gvis_init",
        K::RangePolicy<execution_space>(exec, 0, num_visibilities),
        KOKKOS_LAMBDA(const int i) {
          auto& vis = visibilities(i);
          auto phasor = cphase<execution_space>(vis.m_d_phase);
          for (int vpol = 0; vpol < N; ++vpol) {
            gvisbuff(i).vals[vpol] =
              vis.m_values[vpol] * phasor * vis.m_weights[vpol];
          }
        });
    }

    if (do_grid) {
      const auto N_R = grid.extent_int(int(GridAxis::mrow));
      if (update_grid_weights)
        K::parallel_for(
          "gridding",
          K::TeamPolicy<execution_space>(exec, N_R * num_visibilities, K::AUTO)
          .set_scratch_size(0, K::PerTeam(shmem_size)),
          KOKKOS_LAMBDA(const member_type& team_member) {
            auto i = team_member.league_rank() / N_R;
            auto gpol = team_member.league_rank() % N_R;

            Vis<N, execution_space> vis(
              visibilities(i),
              reinterpret_cast<K::Array<vis_t, N>&>(gvisbuff(i).vals),
              grid_size,
              grid_scale,
              cf_radii,
              oversampling);
            // skip this visibility if all of the updated grid points are not
            // within grid bounds
            if (all_within_grid(vis, grid_size)) {
              grid_vis<cf_layout, grid_layout, memory_space>(
                team_member,
                vis,
                gpol,
                cfs[vis.m_cf_grp],
                mueller_indexes,
                conjugate_mueller_indexes,
                grid,
                weights,
                scratch_phscr_view(
                  team_member.team_scratch(0),
                  max_cf_extent_y));
            }
          });
      else
        K::parallel_for(
          "gridding_no_weights",
          K::TeamPolicy<execution_space>(exec, N_R * num_visibilities, K::AUTO)
          .set_scratch_size(0, K::PerTeam(shmem_size)),
          KOKKOS_LAMBDA(const member_type& team_member) {
            auto i = team_member.league_rank() / N_R;
            auto gpol = team_member.league_rank() % N_R;

            Vis<N, execution_space> vis(
              visibilities(i),
              reinterpret_cast<K::Array<vis_t, N>&>(gvisbuff(i).vals),
              grid_size,
              grid_scale,
              cf_radii,
              oversampling);
            // skip this visibility if all of the updated grid points are not
            // within grid bounds
            if (all_within_grid(vis, grid_size)) {
              grid_vis_weighted_mean<cf_layout, grid_layout, memory_space>(
                team_member,
                vis,
                gpol,
                cfs[vis.m_cf_grp],
                mueller_indexes,
                conjugate_mueller_indexes,
                grid,
                mean_grid,
                weights,
                scratch_phscr_view(
                  team_member.team_scratch(0),
                  max_cf_extent_y));
            }
          });
    }
  }
};

#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
template <int N, typename execution_space>
struct /*HPG_EXPORT*/ VisibilityGridder<N, execution_space, 1> final {

  using member_type =
    typename VisibilityGridder<N, execution_space, 0>::member_type;

  using scratch_phscr_view =
    typename VisibilityGridder<N, execution_space, 0>::scratch_phscr_view;

  template <typename cf_layout, typename grid_layout, typename memory_space>
  static KOKKOS_FUNCTION poln_array_type<visibility_fp, N>
  degrid_vis(
    const member_type& team_member,
    const Vis<N, execution_space>& vis,
    const cf_view<cf_layout, memory_space>& cf,
    const const_mindex_view<memory_space>& mueller_indexes,
    const const_mindex_view<memory_space>& conjugate_mueller_indexes,
    const const_grid_view<grid_layout, memory_space>& model,
    const scratch_phscr_view& phi_Y) {

    const auto& N_X = vis.m_cf_size[0];
    const auto& N_Y = vis.m_cf_size[1];
    const auto N_R = model.extent_int(int(GridAxis::mrow));

    auto degridding_mindex =
      vis.m_pos_w ? conjugate_mueller_indexes : mueller_indexes;
    cf_fp cf_im_factor = (vis.m_pos_w ? 1 : -1);

    // compute the values of the phase screen along the Y axis now and store the
    // results in scratch memory because gridding on the Y axis accesses the
    // phase screen values for every row of the Mueller matrix column
    K::parallel_for(
      K::TeamVectorRange(team_member, N_Y),
      [=](const int Y) {
        phi_Y(Y) = vis.m_phi0[1] + Y * vis.m_dphi[1];
      });
    team_member.team_barrier();

    poln_array_type<visibility_fp, N> result;

    if (model.is_allocated()) {
      // model degridding
      static_assert(std::is_same_v<acc_vis_t, acc_cf_t>);
      vis_array_type<acc_vis_t::value_type, N> vis_array;

      // 3d (X, Y, Mueller) subspace of CF for this visibility
      auto cf_vis =
        K::subview(
          cf,
          K::pair<int, int>(vis.m_cf_major[0], vis.m_cf_major[0] + N_X),
          K::pair<int, int>(vis.m_cf_major[1], vis.m_cf_major[1] + N_Y),
          K::ALL,
          vis.m_cf_cube,
          vis.m_cf_minor[0],
          vis.m_cf_minor[1]);

      // 3d (X, Y, pol) subspace of model for this visibility
      auto model_vis =
        K::subview(
          model,
          K::pair<int, int>(vis.m_grid_coord[0], vis.m_grid_coord[0] + N_X),
          K::pair<int, int>(vis.m_grid_coord[1], vis.m_grid_coord[1] + N_Y),
          K::ALL,
          vis.m_grid_cube);

      // loop over model polarizations
      for (int gpol = 0; gpol < N_R; ++gpol) {
        decltype(vis_array) va;
        // parallel loop over grid X
        K::parallel_reduce(
          K::TeamThreadRange(team_member, N_X),
          [=](const int X, decltype(vis_array)& vis_array_l) {
            auto phi_X = vis.m_phi0[0] + X * vis.m_dphi[0];
            // loop over grid Y
            for (int Y = 0; Y < N_Y; ++Y) {
              auto screen = cphase<execution_space>(-phi_X - phi_Y(Y));
              const auto mv = model_vis(X, Y, gpol) * screen;
              // loop over visibility polarizations
              for (int vpol = 0; vpol < N; ++vpol) {
                if (const auto mindex = degridding_mindex(gpol, vpol);
                    mindex >= 0) {
                  cf_t cfv = cf_vis(X, Y, mindex);
                  cfv.imag() *= cf_im_factor;
                  vis_array_l.vis[vpol] += cfv * mv;
                  vis_array_l.wgt[vpol] += cfv;
                }
              }
            }
          },
          K::Sum<decltype(va)>(va));
        vis_array += va;
      }

      // apply weights and phasor to compute predicted visibilities
      auto conj_phasor = vis.m_phasor;
      conj_phasor.imag() *= -1;
      for (int vpol = 0; vpol < N; ++vpol)
        result.vals[vpol] =
          (vis_array.vis[vpol]
           / ((vis_array.wgt[vpol] != (acc_cf_t)0)
              ? vis_array.wgt[vpol]
              : (acc_cf_t)1))
          * conj_phasor;
    }
    return result;
  }

  template <typename cf_layout, typename grid_layout, typename memory_space>
  static void
  degrid_all(
    execution_space exec,
    const K::Array<cf_view<cf_layout, memory_space>, HPG_MAX_NUM_CF_GROUPS>&
      cfs,
    const K::Array<K::Array<int, 2>, HPG_MAX_NUM_CF_GROUPS>& cf_radii,
    unsigned max_cf_extent_y,
    const_mindex_view<memory_space> mueller_indexes,
    const_mindex_view<memory_space> conjugate_mueller_indexes,
    int num_visibilities,
    const visdata_view<N, memory_space>& visibilities,
    gvisbuff_view<memory_space>& gvisbuff,
    const K::Array<grid_scale_fp, 2>& grid_scale,
    const const_grid_view<grid_layout, memory_space>& model,
    const grid_view<grid_layout, memory_space>& grid) {

    const K::Array<int, 2>
      grid_size{
      grid.extent_int(int(GridAxis::x)),
      grid.extent_int(int(GridAxis::y))};
    const K::Array<int, 2>
      oversampling{
      cfs[0].extent_int(int(CFAxis::x_minor)),
      cfs[0].extent_int(int(CFAxis::y_minor))};

    auto shmem_size = scratch_phscr_view::shmem_size(max_cf_extent_y);

    K::parallel_for(
      "degrid_all",
      K::TeamPolicy<execution_space>(exec, num_visibilities, K::AUTO)
      .set_scratch_size(0, K::PerTeam(shmem_size)),
      KOKKOS_LAMBDA(const member_type& team_member) {
        auto i = team_member.league_rank();
        auto& visibility = visibilities(i);

        Vis<N, execution_space> vis(
          visibility,
          visibility.m_values,
          grid_size,
          grid_scale,
          cf_radii,
          oversampling);
        poln_array_type<visibility_fp, N> gvis;
        if (VisibilityGridder<N, execution_space, 0>::all_within_grid(vis, grid_size))
          gvis =
            degrid_vis<cf_layout, grid_layout, memory_space>(
              team_member,
              vis,
              cfs[vis.m_cf_grp],
              mueller_indexes,
              conjugate_mueller_indexes,
              model,
              scratch_phscr_view(
                team_member.team_scratch(0),
                max_cf_extent_y));
        else
          gvis.set_nan();
        K::single(
          K::PerTeam(team_member),
          [&]() {
            reinterpret_cast<poln_array_type<visibility_fp, N>&>(gvisbuff(i)) =
              gvis;
          });
      });
  }

  // compute residual visibilities, prepare values for gridding
  template <typename memory_space>
  static KOKKOS_FUNCTION void
  vis_copy_residual_and_rescale(
    execution_space exec,
    int num_visibilities,
    visdata_view<N, memory_space>& visibilities,
    gvisbuff_view<memory_space>& gvisbuff) {
    K::parallel_for(
      "vis_copy_residual_and_rescale",
      K::TeamPolicy<execution_space>(exec, num_visibilities, K::AUTO),
      KOKKOS_LAMBDA(const member_type& team_member) {
        auto i = team_member.league_rank();
        auto& vis = visibilities(i);
        auto phasor = cphase<execution_space>(vis.m_d_phase);
        auto& gvis = gvisbuff(i);
        K::parallel_for(
          K::TeamThreadRange(team_member, N),
          [&](const int vpol) {
            vis.m_values[vpol] -= gvis.vals[vpol];
            gvis.vals[vpol] = vis.m_values[vpol] * phasor * vis.m_weights[vpol];
          });
      });
  }

  // prepare values for gridding
  template <typename memory_space>
  static KOKKOS_FUNCTION void
  vis_rescale(
    execution_space exec,
    int num_visibilities,
    const visdata_view<N, memory_space>& visibilities,
    gvisbuff_view<memory_space>& gvisbuff) {
    K::parallel_for(
      "vis_rescale",
      K::TeamPolicy<execution_space>(exec, num_visibilities, K::AUTO),
      KOKKOS_LAMBDA(const member_type& team_member) {
        auto i = team_member.league_rank();
        auto& vis = visibilities(i);
        auto phasor = cphase<execution_space>(vis.m_d_phase);
        auto& gvis = gvisbuff(i);
        K::parallel_for(
          K::TeamThreadRange(team_member, N),
          [&](const int vpol) {
            gvis.vals[vpol] = vis.m_values[vpol] * phasor * vis.m_weights[vpol];
          });
      });
  }

  // copy predicted visibilities
  template <typename memory_space>
  static KOKKOS_FUNCTION void
  vis_copy_predicted(
    execution_space exec,
    int num_visibilities,
    visdata_view<N, memory_space>& visibilities,
    gvisbuff_view<memory_space>& gvisbuff) {
    K::parallel_for(
      "vis_copy_predicted",
      K::TeamPolicy<execution_space>(exec, num_visibilities, K::AUTO),
      KOKKOS_LAMBDA(const member_type& team_member) {
        auto i = team_member.league_rank();
        auto& vis = visibilities(i);
        auto& gvis = gvisbuff(i);
        K::parallel_for(
          K::TeamThreadRange(team_member, N),
          [&](const int vpol) {
            vis.m_values[vpol] = gvis.vals[vpol];
          });
      });
  }

  // function for gridding a single visibility with sum of weights
  template <typename cf_layout, typename grid_layout, typename memory_space>
  static KOKKOS_FUNCTION void
  grid_vis(
    const member_type& team_member,
    const Vis<N, execution_space>& vis,
    const unsigned gpol,
    const cf_view<cf_layout, memory_space>& cf,
    const const_mindex_view<memory_space>& mueller_indexes,
    const const_mindex_view<memory_space>& conjugate_mueller_indexes,
    const grid_view<grid_layout, memory_space>& grid,
    const weight_view<typename execution_space::array_layout, memory_space>&
    weights,
    const scratch_phscr_view& phi_Y) {

    const auto& N_X = vis.m_cf_size[0];
    const auto& N_Y = vis.m_cf_size[1];

    auto gridding_mindex =
      K::subview(
        (vis.m_pos_w ? mueller_indexes : conjugate_mueller_indexes),
        gpol,
        K::ALL);
    cf_fp cf_im_factor = (vis.m_pos_w ? -1 : 1);

    // compute the values of the phase screen along the Y axis now and store the
    // results in scratch memory because gridding on the Y axis accesses the
    // phase screen values for every row of the Mueller matrix column
    K::parallel_for(
      K::TeamVectorRange(team_member, N_Y),
      [=](const int Y) {
        phi_Y(Y) = vis.m_phi0[1] + Y * vis.m_dphi[1];
      });
    team_member.team_barrier();

    // 3d (X, Y, Mueller) subspace of CF for this visibility
    auto cf_vis =
      K::subview(
        cf,
        K::pair<int, int>(vis.m_cf_major[0], vis.m_cf_major[0] + N_X),
        K::pair<int, int>(vis.m_cf_major[1], vis.m_cf_major[1] + N_Y),
        K::ALL,
        vis.m_cf_cube,
        vis.m_cf_minor[0],
        vis.m_cf_minor[1]);

    // 2d (X, Y) subspace of grid for this visibility and grid polarization
    // (gpol)
    auto grd_vis =
      K::subview(
        grid,
        K::pair<int, int>(vis.m_grid_coord[0], vis.m_grid_coord[0] + N_X),
        K::pair<int, int>(vis.m_grid_coord[1], vis.m_grid_coord[1] + N_Y),
        gpol,
        vis.m_grid_cube);

    // accumulate to grid, and CF weights per visibility polarization
    poln_array_type<acc_cf_t::value_type, N> grid_wgt;
    // parallel loop over grid X
    K::parallel_reduce(
      K::TeamThreadRange(team_member, N_X),
      [=](const int X, decltype(grid_wgt)& grid_wgt_l) {
        auto phi_X = vis.m_phi0[0] + X * vis.m_dphi[0];
        // loop over grid Y
        for (int Y = 0; Y < N_Y; ++Y) {
          const cf_t screen = cphase<execution_space>(phi_X + phi_Y(Y));
          gv_t gv(0);
          // loop over visibility polarizations
          for (int vpol = 0; vpol < N; ++vpol) {
            if (const auto mindex = gridding_mindex(vpol); mindex >= 0) {
              cf_t cfv = cf_vis(X, Y, mindex);
              cfv.imag() *= cf_im_factor;
              gv += gv_t(cfv * screen * vis.m_values[vpol]);
              grid_wgt_l.vals[vpol] += cfv;
            }
          }
          pseudo_atomic_add<execution_space>(grd_vis(X, Y), gv);
        }
      },
      K::Sum<decltype(grid_wgt)>(grid_wgt));
    // compute final weight and add it to weights
    K::single(
      K::PerTeam(team_member),
      [&]() {
        grid_value_fp twgt = 0;
        for (int vpol = 0; vpol < N; ++vpol)
          twgt +=
            grid_value_fp(mag(grid_wgt.vals[vpol]) * vis.m_weights[vpol]);
        K::atomic_add(&weights(gpol, vis.m_grid_cube), twgt);
      });
  }

  // function for gridding a single visibility without sum of weights
  template <typename cf_layout, typename grid_layout, typename memory_space>
  static KOKKOS_FUNCTION void
  grid_vis_no_weights(
    const member_type& team_member,
    const Vis<N, execution_space>& vis,
    const unsigned gpol,
    const cf_view<cf_layout, memory_space>& cf,
    const const_mindex_view<memory_space>& mueller_indexes,
    const const_mindex_view<memory_space>& conjugate_mueller_indexes,
    const grid_view<grid_layout, memory_space>& grid,
    const scratch_phscr_view& phi_Y) {

    const auto& N_X = vis.m_cf_size[0];
    const auto& N_Y = vis.m_cf_size[1];

    auto gridding_mindex =
      K::subview(
        (vis.m_pos_w ? mueller_indexes : conjugate_mueller_indexes),
        gpol,
        K::ALL);
    cf_fp cf_im_factor = (vis.m_pos_w ? -1 : 1);

    // compute the values of the phase screen along the Y axis now and store the
    // results in scratch memory because gridding on the Y axis accesses the
    // phase screen values for every row of the Mueller matrix column
    K::parallel_for(
      K::TeamVectorRange(team_member, N_Y),
      [=](const int Y) {
        phi_Y(Y) = vis.m_phi0[1] + Y * vis.m_dphi[1];
      });
    team_member.team_barrier();

    // 3d (X, Y, Mueller) subspace of CF for this visibility
    auto cf_vis =
      K::subview(
        cf,
        K::pair<int, int>(vis.m_cf_major[0], vis.m_cf_major[0] + N_X),
        K::pair<int, int>(vis.m_cf_major[1], vis.m_cf_major[1] + N_Y),
        K::ALL,
        vis.m_cf_cube,
        vis.m_cf_minor[0],
        vis.m_cf_minor[1]);

    // 2d (X, Y) subspace of grid for this visibility and grid polarization
    // (gpol)
    auto grd_vis =
      K::subview(
        grid,
        K::pair<int, int>(vis.m_grid_coord[0], vis.m_grid_coord[0] + N_X),
        K::pair<int, int>(vis.m_grid_coord[1], vis.m_grid_coord[1] + N_Y),
        gpol,
        vis.m_grid_cube);

    // parallel loop over grid X
    K::parallel_for(
      K::TeamThreadRange(team_member, N_X),
      [=](const int X) {
        auto phi_X = vis.m_phi0[0] + X * vis.m_dphi[0];
        // loop over grid Y
        for (int Y = 0; Y < N_Y; ++Y) {
          const cf_t screen = cphase<execution_space>(phi_X + phi_Y(Y));
          gv_t gv(0);
          // loop over visibility polarizations
          for (int vpol = 0; vpol < N; ++vpol) {
            if (const auto mindex = gridding_mindex(vpol); mindex >= 0) {
              cf_t cfv = cf_vis(X, Y, mindex);
              cfv.imag() *= cf_im_factor;
              gv += gv_t(cfv * screen * vis.m_values[vpol]);
            }
          }
          pseudo_atomic_add<execution_space>(grd_vis(X, Y), gv);
        }
      });
  }

  template <
    typename cf_layout,
    typename grid_layout,
    typename memory_space,
    typename F>
  static KOKKOS_FUNCTION void
  for_all_vis_on_grid(
    execution_space exec,
    const K::Array<cf_view<cf_layout, memory_space>, HPG_MAX_NUM_CF_GROUPS>&
    cfs,
    const K::Array<K::Array<int, 2>, HPG_MAX_NUM_CF_GROUPS>& cf_radii,
    unsigned max_cf_extent_y,
    int num_visibilities,
    visdata_view<N, memory_space>& visibilities,
    gvisbuff_view<memory_space>& gvisbuff,
    const K::Array<grid_scale_fp, 2>& grid_scale,
    const grid_view<grid_layout, memory_space>& grid,
    const F& grid_one) {

    const K::Array<int, 2>
      grid_size{
      grid.extent_int(int(GridAxis::x)),
      grid.extent_int(int(GridAxis::y))};
    const K::Array<int, 2>
      oversampling{
      cfs[0].extent_int(int(CFAxis::x_minor)),
      cfs[0].extent_int(int(CFAxis::y_minor))};

    auto shmem_size = scratch_phscr_view::shmem_size(max_cf_extent_y);

    const auto N_R = grid.extent_int(int(GridAxis::mrow));

    K::parallel_for(
      "gridding_no_weights",
      K::TeamPolicy<execution_space>(exec, N_R * num_visibilities, K::AUTO)
      .set_scratch_size(0, K::PerTeam(shmem_size)),
      KOKKOS_LAMBDA(const member_type& team_member) {
        auto i = team_member.league_rank() / N_R;
        auto gpol = team_member.league_rank() % N_R;

        Vis<N, execution_space> vis(
          visibilities(i),
          reinterpret_cast<K::Array<vis_t, N>&>(gvisbuff(i).vals),
          grid_size,
          grid_scale,
          cf_radii,
          oversampling);
        // skip this visibility if all of the updated grid points are not
        // within grid bounds
        if (VisibilityGridder<N, execution_space, 0>::all_within_grid(
              vis,
              grid_size))
          grid_one(
            team_member,
            vis,
            gpol,
            cfs[vis.m_cf_grp],
            scratch_phscr_view(
              team_member.team_scratch(0),
              max_cf_extent_y));
      });
  }

  template <typename cf_layout, typename grid_layout, typename memory_space>
  static KOKKOS_FUNCTION void
  grid_all_no_weights(
    execution_space exec,
    const K::Array<cf_view<cf_layout, memory_space>, HPG_MAX_NUM_CF_GROUPS>&
    cfs,
    const K::Array<K::Array<int, 2>, HPG_MAX_NUM_CF_GROUPS>& cf_radii,
    unsigned max_cf_extent_y,
    const_mindex_view<memory_space> mueller_indexes,
    const_mindex_view<memory_space> conjugate_mueller_indexes,
    int num_visibilities,
    visdata_view<N, memory_space>& visibilities,
    gvisbuff_view<memory_space>& gvisbuff,
    const K::Array<grid_scale_fp, 2>& grid_scale,
    const grid_view<grid_layout, memory_space>& grid) {

    for_all_vis_on_grid(
      exec,
      cfs,
      cf_radii,
      max_cf_extent_y,
      num_visibilities,
      visibilities,
      gvisbuff,
      grid_scale,
      grid,
      KOKKOS_LAMBDA(
        const member_type& team_member,
        const Vis<N, execution_space>& vis,
        int gpol,
        const cf_view<cf_layout, memory_space>& cf,
        const scratch_phscr_view& phi_Y) {
        grid_vis_no_weights<cf_layout, grid_layout, memory_space>(
          team_member,
          vis,
          gpol,
          cf,
          mueller_indexes,
          conjugate_mueller_indexes,
          grid,
          phi_Y);
      });
  }

  template <typename cf_layout, typename grid_layout, typename memory_space>
  static KOKKOS_FUNCTION void
  grid_all(
    execution_space exec,
    const K::Array<cf_view<cf_layout, memory_space>, HPG_MAX_NUM_CF_GROUPS>&
    cfs,
    const K::Array<K::Array<int, 2>, HPG_MAX_NUM_CF_GROUPS>& cf_radii,
    unsigned max_cf_extent_y,
    const_mindex_view<memory_space> mueller_indexes,
    const_mindex_view<memory_space> conjugate_mueller_indexes,
    int num_visibilities,
    visdata_view<N, memory_space>& visibilities,
    gvisbuff_view<memory_space>& gvisbuff,
    const K::Array<grid_scale_fp, 2>& grid_scale,
    const grid_view<grid_layout, memory_space>& grid,
    const weight_view<typename execution_space::array_layout, memory_space>&
    weights) {

    for_all_vis_on_grid(
      exec,
      cfs,
      cf_radii,
      max_cf_extent_y,
      num_visibilities,
      visibilities,
      gvisbuff,
      grid_scale,
      grid,
      KOKKOS_LAMBDA(
        const member_type& team_member,
        const Vis<N, execution_space>& vis,
        int gpol,
        const cf_view<cf_layout, memory_space>& cf,
        const scratch_phscr_view& phi_Y) {
        grid_vis<cf_layout, grid_layout, memory_space>(
          team_member,
          vis,
          gpol,
          cf,
          mueller_indexes,
          conjugate_mueller_indexes,
          grid,
          weights,
          phi_Y);
      });
  }
};
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS

/** grid normalization kernel
 */
template <typename execution_space, unsigned version>
struct /*HPG_EXPORT*/ GridNormalizer final {

  template <typename grid_layout, typename memory_space>
  static void
  kernel(
    execution_space exec,
    const grid_view<grid_layout, memory_space>& grid,
    const const_weight_view<
    typename execution_space::array_layout, memory_space>& weights,
      const grid_value_fp& wfactor) {

    static_assert(
      int(GridAxis::x) == 0
      && int(GridAxis::y) == 1
      && int(GridAxis::mrow) == 2
      && int(GridAxis::cube) == 3);
    static_assert(
      GridWeightArray::Axis::mrow == 0 && GridWeightArray::Axis::cube == 1);

    K::parallel_for(
      "normalization",
      K::MDRangePolicy<K::Rank<4>, execution_space>(
        exec,
        {0, 0, 0, 0},
        {grid.extent_int(int(GridAxis::x)),
         grid.extent_int(int(GridAxis::y)),
         grid.extent_int(int(GridAxis::mrow)),
         grid.extent_int(int(GridAxis::cube))}),
      KOKKOS_LAMBDA(int x, int y, int mrow, int cube) {
        grid(x, y, mrow, cube) /= (wfactor * weights(mrow, cube));
      });
  }

  template <typename grid_layout, typename memory_space>
  static void
  kernel(
    execution_space exec,
    const grid_view<grid_layout, memory_space>& grid,
    const grid_value_fp& norm) {

    static_assert(
      int(GridAxis::x) == 0
      && int(GridAxis::y) == 1
      && int(GridAxis::mrow) == 2
      && int(GridAxis::cube) == 3);

    grid_value_fp inv_norm = (grid_value_fp)(1.0) / norm;
    K::parallel_for(
      "normalization",
      K::MDRangePolicy<K::Rank<4>, execution_space>(
        exec,
        {0, 0, 0, 0},
        {grid.extent_int(int(GridAxis::x)),
         grid.extent_int(int(GridAxis::y)),
         grid.extent_int(int(GridAxis::mrow)),
         grid.extent_int(int(GridAxis::cube))}),
      KOKKOS_LAMBDA(int x, int y, int mrow, int cube) {
        grid(x, y, mrow, cube) *= inv_norm;
      });
  }
};

/** fftw function class templated on fp precision */
template <typename T>
struct /*HPG_EXPORT*/ FFTW {

  using complex_t = void;
  using plan_t = void;

  // static void
  // exec(const plan_t plan, K::complex<T>* in, K::complex<T>* out) {
  // }

#ifdef HPG_ENABLE_OPENMP
  static void
  plan_with_nthreads(int n) {
  }
#endif // HPG_ENABLE_OPENMP

  // static std::tuple<plan_t, plan_t>
  // plan_many(
  //   int rank, const int *n, int howmany,
  //   const K::complex<T> *in, const int *inembed,
  //   int istride, int idist,
  //   K::complex<T> *out, const int *onembed,
  //   int ostride, int odist,
  //   int sign, unsigned flags);

  // static void
  // destroy_plan(std::tuple<plan_t, plan_t> plan);
};

/** FFTW specialized for double precision */
template <>
struct /*HPG_EXPORT*/ FFTW<double> {

  using complex_t = fftw_complex;
  using plan_t = fftw_plan;

  static void
  exec(const plan_t plan, K::complex<double>* in, K::complex<double>* out) {
    fftw_execute_dft(
      plan,
      reinterpret_cast<complex_t*>(in),
      reinterpret_cast<complex_t*>(out));
  }

#ifdef HPG_ENABLE_OPENMP
  static void
  plan_with_nthreads(int n) {
    fftw_plan_with_nthreads(n);
  }
#endif // HPG_ENABLE_OPENMP

  static std::tuple<plan_t, plan_t>
  plan_many(
    int rank, const int *n, int howmany,
    K::complex<double> *in, const int *inembed,
    int istride, int idist,
    K::complex<double> *out, const int *onembed,
    int ostride, int odist,
    int /*sstride*/,
    int sign, unsigned flags) {

    static_assert(sizeof(*in) == 16);

    auto plan =
      fftw_plan_many_dft(
        rank, n, howmany,
        reinterpret_cast<complex_t*>(in), inembed, istride, idist,
        reinterpret_cast<complex_t*>(out), onembed, ostride, odist,
        sign, flags);
    return {plan, plan};
  }

  static void
  destroy_plan(std::tuple<plan_t, plan_t> plans) {
    fftw_destroy_plan(std::get<0>(plans));
  }
};

/** FFTW specialized for single precision */
template <>
struct /*HPG_EXPORT*/ FFTW<float> {

  using complex_t = fftwf_complex;
  using plan_t = fftwf_plan;

  static void
  exec(const plan_t plan, K::complex<float>* in, K::complex<float>* out) {
    fftwf_execute_dft(
      plan,
      reinterpret_cast<complex_t*>(in),
      reinterpret_cast<complex_t*>(out));
  }

#ifdef HPG_ENABLE_OPENMP
  static void
  plan_with_nthreads(int n) {
    fftwf_plan_with_nthreads(n);
  }
#endif // HPG_ENABLE_OPENMP

  static std::tuple<plan_t, plan_t>
  plan_many(
    int rank, const int *n, int howmany,
    K::complex<float> *in, const int *inembed,
    int istride, int idist,
    K::complex<float> *out, const int *onembed,
    int ostride, int odist,
    int sstride,
    int sign, unsigned flags) {

    static_assert(sizeof(*in) == 8);

    return
      {fftwf_plan_many_dft(
          rank, n, howmany,
          reinterpret_cast<complex_t*>(in), inembed, istride, idist,
          reinterpret_cast<complex_t*>(out), onembed, ostride, odist,
          sign, flags),
       fftwf_plan_many_dft(
         rank, n, howmany,
         reinterpret_cast<complex_t*>(in + sstride), inembed, istride, idist,
         reinterpret_cast<complex_t*>(out + sstride), onembed, ostride, odist,
         sign, flags)};
  }

  static void
  destroy_plan(std::tuple<plan_t, plan_t> plans) {
    fftwf_destroy_plan(std::get<0>(plans));
    fftwf_destroy_plan(std::get<1>(plans));
  }
};

/** FFT kernels
 *
 * Both in-place and out-of-place versions
 */
template <typename execution_space, unsigned version>
struct /*HPG_EXPORT*/ FFT final {

  // default implementation assumes FFTW3

  template <typename IG, typename OG>
  static auto
  grid_fft_handle(execution_space exec, FFTSign sign, IG& igrid, OG& ogrid) {

    using scalar_t = typename OG::value_type::value_type;

#ifdef HPG_ENABLE_OPENMP
    if constexpr (std::is_same_v<execution_space, K::Serial>)
                   FFTW<scalar_t>::plan_with_nthreads(1);
    else
      FFTW<scalar_t>::plan_with_nthreads(omp_get_max_threads());
#endif // HPG_ENABLE_OPENMP

    // this assumes there is no padding in grid
    assert(igrid.span() ==
           igrid.extent(0) * igrid.extent(1)
           * igrid.extent(2) * igrid.extent(3));
    static_assert(
      int(GridAxis::x) == 0
      && int(GridAxis::y) == 1
      && int(GridAxis::mrow) == 2
      && int(GridAxis::cube) == 3);
    int n[2]{igrid.extent_int(0), igrid.extent_int(1)};
    int stride = 1;
    int dist = igrid.extent_int(0) * igrid.extent_int(1) * igrid.extent_int(2);
    int nembed[2]{
                igrid.extent_int(0) * igrid.extent_int(2),
                  igrid.extent_int(1)};
    auto result =
      FFTW<scalar_t>::plan_many(
        2, n, igrid.extent_int(3),
        const_cast<K::complex<scalar_t>*>(&igrid(0, 0, 0, 0)),
        nembed, stride, dist,
        &ogrid(0, 0, 0, 0), nembed, stride, dist,
        igrid.extent_int(1),
        ((sign == FFTSign::NEGATIVE) ? FFTW_FORWARD : FFTW_BACKWARD),
        FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    return result;
  }

  /** in-place FFT kernel
   */
  template <typename grid_layout, typename memory_space>
  static std::optional<Error>
  in_place_kernel(
    execution_space exec,
    FFTSign sign,
    const grid_view<grid_layout, memory_space>& grid) {

    using scalar_t =
      typename grid_view<grid_layout, memory_space>::value_type::value_type;

    auto handles = grid_fft_handle(exec, sign, grid, grid);
    auto& [h0, h1] = handles;
    std::optional<Error> result;
    if (h0 == nullptr || h1 == nullptr)
      result = Error("fftw in_place_kernel() failed");
    if (!result) {
      for (int mrow = 0; mrow < grid.extent_int(2); ++mrow) {
        FFTW<scalar_t>::exec(h0, &grid(0, 0, mrow, 0), &grid(0, 0, mrow, 0));
        std::swap(h0, h1);
      }
      FFTW<scalar_t>::destroy_plan(handles);
    }
    return result;
  }

  /** out-of-place FFT kernel
   */
  template <typename grid_layout, typename memory_space>
  static std::optional<Error>
  out_of_place_kernel(
    execution_space exec,
    FFTSign sign,
    const const_grid_view<grid_layout, memory_space>& pre_grid,
    const grid_view<grid_layout, memory_space>& post_grid) {

    using scalar_t =
      typename grid_view<grid_layout, memory_space>::value_type::value_type;

    auto handles = grid_fft_handle(exec, sign, pre_grid, post_grid);
    auto& [h0, h1] = handles;
    std::optional<Error> result;
    if (h0 == nullptr || h1 == nullptr)
      result = Error("fftw in_place_kernel() failed");
    if (!result) {
      for (int mrow = 0; mrow < pre_grid.extent_int(2); ++mrow) {
        FFTW<scalar_t>::exec(
          h0,
          const_cast<K::complex<scalar_t>*>(&pre_grid(0, 0, mrow, 0)),
          &post_grid(0, 0, mrow, 0));
        std::swap(h0, h1);
      }
      FFTW<scalar_t>::destroy_plan(handles);
    }
    return result;
  }
};

#ifdef HPG_ENABLE_CUDA

/*HPG_EXPORT*/ Error
cufft_error(const std::string& prefix, cufftResult rc);

/** cufft function class templated on fp precision */
template <typename T>
struct /*HPG_EXPORT*/ CUFFT {
  //constexpr cufftType type;
  static cufftResult
  exec(cufftHandle, K::complex<T>*, K::complex<T>*, int) {
    assert(false);
    return CUFFT_NOT_SUPPORTED;
  }
};

template <>
struct /*HPG_EXPORT*/ CUFFT<double> {

  static constexpr cufftType type = CUFFT_Z2Z;

  static cufftResult
  exec(
    cufftHandle plan,
    FFTSign sign,
    K::complex<double>* idata,
    K::complex<double>* odata) {

    return
      cufftExecZ2Z(
        plan,
        reinterpret_cast<cufftDoubleComplex*>(idata),
        reinterpret_cast<cufftDoubleComplex*>(odata),
        ((sign == FFTSign::NEGATIVE) ? CUFFT_FORWARD : CUFFT_INVERSE));
  }
};

template <>
struct /*HPG_EXPORT*/ CUFFT<float> {

  static constexpr cufftType type = CUFFT_C2C;

  static cufftResult
  exec(
    cufftHandle plan,
    FFTSign sign,
    K::complex<float>* idata,
    K::complex<float>* odata) {
    return
      cufftExecC2C(
        plan,
        reinterpret_cast<cufftComplex*>(idata),
        reinterpret_cast<cufftComplex*>(odata),
        ((sign == FFTSign::NEGATIVE) ? CUFFT_FORWARD : CUFFT_INVERSE));
  }
};

/** fft kernels for Cuda
 */
template <>
struct /*HPG_EXPORT*/ FFT<K::Cuda, 0> final {

  template <typename G>
  static std::tuple<cufftResult_t, cufftHandle>
  grid_fft_handle(K::Cuda exec, G& grid) {

    using scalar_t = typename G::value_type::value_type;

    // this assumes there is no padding in grid
    assert(grid.span() ==
           grid.extent(0) * grid.extent(1) * grid.extent(2) * grid.extent(3));
    static_assert(
      int(GridAxis::x) == 0
      && int(GridAxis::y) == 1
      && int(GridAxis::mrow) == 2
      && int(GridAxis::cube) == 3);
    int n[2]{grid.extent_int(1), grid.extent_int(0)};
    cufftHandle result;
    auto rc =
      cufftPlanMany(
        &result, 2, n,
        NULL, 1, 1,
        NULL, 1, 1,
        CUFFT<scalar_t>::type,
        grid.extent_int(2) * grid.extent_int(3));
    if (rc == CUFFT_SUCCESS)
      rc = cufftSetStream(result, exec.cuda_stream());
    return {rc, result};
  }

  /** in-place FFT kernel
   */
  template <typename grid_layout, typename memory_space>
  static std::optional<Error>
  in_place_kernel(
      K::Cuda exec,
      FFTSign sign,
      const grid_view<grid_layout, memory_space>& grid) {

    using scalar_t =
      typename grid_view<grid_layout, memory_space>::value_type::value_type;

    auto [rc, handle] = grid_fft_handle(exec, grid);
    if (rc == CUFFT_SUCCESS) {
      rc = CUFFT<scalar_t>::exec(handle, sign, grid.data(), grid.data());
      auto rc0 = cufftDestroy(handle);
      assert(rc0 == CUFFT_SUCCESS);
    }
    std::optional<Error> result;
    if (rc != CUFFT_SUCCESS)
      result = cufft_error("Cuda in_place_kernel() failed: ", rc);
    return result;
  }

  /** out-of-place FFT kernel
   */
  template <typename grid_layout, typename memory_space>
  static std::optional<Error>
  out_of_place_kernel(
    K::Cuda exec,
    FFTSign sign,
    const const_grid_view<grid_layout, memory_space>& pre_grid,
    const grid_view<grid_layout, memory_space>& post_grid) {

    using scalar_t =
      typename grid_view<grid_layout, memory_space>::value_type::value_type;

    auto [rc, handle] = grid_fft_handle(exec, post_grid);
    if (rc == CUFFT_SUCCESS) {
      rc =
        CUFFT<scalar_t>::exec(
          handle,
          sign,
          const_cast<K::complex<scalar_t>*>(pre_grid.data()),
          post_grid.data());
      auto rc0 = cufftDestroy(handle);
      assert(rc0 == CUFFT_SUCCESS);
    }
    std::optional<Error> result;
    if (rc != CUFFT_SUCCESS)
      result = cufft_error("cuda out_of_place_kernel() failed: ", rc);
    return result;
  }
};
#endif // HPG_ENABLE_CUDA

/** swap visibility values */
#pragma nv_exec_check_disable
template <typename execution_space>
/*HPG_EXPORT*/ KOKKOS_FORCEINLINE_FUNCTION void
swap_gv(gv_t& a, gv_t&b) {
  std::swap(a, b);
}

#ifdef HPG_ENABLE_CUDA
template <>
/*HPG_EXPORT*/ KOKKOS_FORCEINLINE_FUNCTION void
swap_gv<K::Cuda>(gv_t& a, gv_t&b) {
  gv_t tmp;
  tmp = a;
  a = b;
  b = tmp;
}
#endif // HPG_ENABLE_CUDA

/** grid rotation kernel
 *
 * Useful after FFT to shift grid planes by half the grid plane size in each
 * dimension
 */
template <typename execution_space, unsigned version>
struct /*HPG_EXPORT*/ GridShifter final {

  template <typename grid_layout, typename memory_space>
  static void
  kernel(
    ShiftDirection direction,
    execution_space exec,
    const grid_view<grid_layout, memory_space>& grid) {

    using member_type = typename K::TeamPolicy<execution_space>::member_type;

    // TODO: is this kernel valid for all possible GridAxis definitions?
    static_assert(
      int(GridAxis::x) == 0
      && int(GridAxis::y) == 1
      && int(GridAxis::mrow) == 2
      && int(GridAxis::cube) == 3);
    int n_x = grid.extent_int(0);
    int n_y = grid.extent_int(1);
    int n_mrow = grid.extent_int(2);
    int n_cube = grid.extent_int(3);

    int mid_x = n_x / 2;
    int mid_y = n_y / 2;

    if (n_x % 2 == 0 && n_y % 2 == 0) {
      // simpler (faster?) algorithm when both grid side lengths are even

      K::parallel_for(
        "grid_shift_ee",
        K::TeamPolicy<execution_space>(exec, n_mrow * n_cube, K::AUTO),
        KOKKOS_LAMBDA(const member_type& team_member) {
          auto gplane =
            K::subview(
              grid,
              K::ALL,
              K::ALL,
              team_member.league_rank() % n_mrow,
              team_member.league_rank() / n_mrow);
          K::parallel_for(
            K::TeamVectorRange(team_member, n_x / 2),
            [=](int x) {
              for (int y = 0; y < n_y / 2; ++y) {
                swap_gv<execution_space>(
                  gplane(x, y),
                  gplane(x + mid_x, y + mid_y));
                swap_gv<execution_space>(
                  gplane(x + mid_x, y),
                  gplane(x, y + mid_y));
              }
            });
        });
    } else if (n_x == n_y) {
      // single-pass algorithm for odd-length, square grid

      if (direction == ShiftDirection::FORWARD)
        K::parallel_for(
          "grid_rotation_oop",
          K::TeamPolicy<execution_space>(exec, n_mrow * n_cube, K::AUTO),
          KOKKOS_LAMBDA(const member_type& team_member) {
            auto gplane =
              K::subview(
                grid,
                K::ALL,
                K::ALL,
                team_member.league_rank() % n_mrow,
                team_member.league_rank() / n_mrow);
            K::parallel_for(
              K::TeamVectorRange(team_member, n_x),
              [=](int x) {
                gv_t tmp;
                int y = 0;
                for (int i = 0; i <= n_y; ++i) {
                  swap_gv<execution_space>(tmp, gplane(x, y));
                  x += mid_x;
                  if (x >= n_x)
                    x -= n_x;
                  y += mid_y;
                  if (y >= n_y)
                    y -= n_y;
                }
              });
          });
      else // direction == ShiftDirection::BACKWARD
        K::parallel_for(
          "grid_rotation_oon",
          K::TeamPolicy<execution_space>(exec, n_mrow * n_cube, K::AUTO),
          KOKKOS_LAMBDA(const member_type& team_member) {
            auto gplane =
              K::subview(
                grid,
                K::ALL,
                K::ALL,
                team_member.league_rank() % n_mrow,
                team_member.league_rank() / n_mrow);
            K::parallel_for(
              K::TeamVectorRange(team_member, n_x),
              [=](int x) {
                gv_t tmp;
                int y = 0;
                for (int i = 0; i <= n_y; ++i) {
                  swap_gv<execution_space>(tmp, gplane(x, y));
                  x -= mid_x;
                  if (x < 0)
                    x += n_x;
                  y -= mid_y;
                  if (y < 0)
                    y += n_y;
                }
              });
          });
    } else {
      // two-pass algorithm for the general case

      if (direction == ShiftDirection::FORWARD)
        K::parallel_for(
          "grid_rotation_genp",
          K::TeamPolicy<execution_space>(exec, n_mrow * n_cube, K::AUTO),
          KOKKOS_LAMBDA(const member_type& team_member) {
            auto gplane =
              K::subview(
                grid,
                K::ALL,
                K::ALL,
                team_member.league_rank() % n_mrow,
                team_member.league_rank() / n_mrow);

            // first pass, parallel over x
            if (n_y % 2 == 1)
              K::parallel_for(
                K::TeamThreadRange(team_member, n_x),
                [=](int x) {
                  gv_t tmp;
                  int y = 0;
                  for (int i = 0; i <= n_y; ++i) {
                    swap_gv<execution_space>(tmp, gplane(x, y));
                    y += mid_y;
                    if (y >= n_y)
                      y -= n_y;
                  }
                });
            else
              K::parallel_for(
                K::TeamThreadRange(team_member, n_x),
                [=](int x) {
                  for (int y = 0; y < mid_y; ++y)
                    swap_gv<execution_space>(
                      gplane(x, y),
                      gplane(x, y + mid_y));
                });

            // second pass, parallel over y
            if (n_x % 2 == 1)
              K::parallel_for(
                K::TeamThreadRange(team_member, n_y),
                [=](int y) {
                  gv_t tmp;
                  int x = 0;
                  for (int i = 0; i <= n_x; ++i) {
                    swap_gv<execution_space>(tmp, gplane(x, y));
                    x += mid_x;
                    if (x >= n_x)
                      x -= n_x;
                  }
                });
            else
              K::parallel_for(
                K::TeamThreadRange(team_member, n_y),
                [=](int y) {
                  for (int x = 0; x < mid_x; ++x)
                    swap_gv<execution_space>(
                      gplane(x, y),
                      gplane(x + mid_x, y));
                });
          });
      else // direction == ShiftDirection::BACKWARD
        K::parallel_for(
          "grid_rotation_genn",
          K::TeamPolicy<execution_space>(exec, n_mrow * n_cube, K::AUTO),
          KOKKOS_LAMBDA(const member_type& team_member) {
            auto gplane =
              K::subview(
                grid,
                K::ALL,
                K::ALL,
                team_member.league_rank() % n_mrow,
                team_member.league_rank() / n_mrow);

            // first pass, parallel over x
            if (n_y % 2 == 1)
              K::parallel_for(
                K::TeamThreadRange(team_member, n_x),
                [=](int x) {
                  gv_t tmp;
                  int y = 0;
                  for (int i = 0; i <= n_y; ++i) {
                    swap_gv<execution_space>(tmp, gplane(x, y));
                    y -= mid_y;
                    if (y < 0)
                      y += n_y;
                  }
                });
            else
              K::parallel_for(
                K::TeamThreadRange(team_member, n_x),
                [=](int x) {
                  for (int y = mid_y; y < n_y; ++y)
                    swap_gv<execution_space>(
                      gplane(x, y),
                      gplane(x, y - mid_y));
                });

            // second pass, parallel over y
            if (n_x % 2 == 1)
              K::parallel_for(
                K::TeamThreadRange(team_member, n_y),
                [=](int y) {
                  gv_t tmp;
                  int x = 0;
                  for (int i = 0; i <= n_x; ++i) {
                    swap_gv<execution_space>(tmp, gplane(x, y));
                    x -= mid_x;
                    if (x < 0)
                      x += n_x;
                  }
                });
            else
              K::parallel_for(
                K::TeamThreadRange(team_member, n_y),
                [=](int y) {
                  for (int x = mid_x; x < n_x; ++x)
                    swap_gv<execution_space>(
                      gplane(x, y),
                      gplane(x - mid_x, y));
                });
          });
    }
  }
};

} // ena namespace hpg::runtime:impl::core

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End: