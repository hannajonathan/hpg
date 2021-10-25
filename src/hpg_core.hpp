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

#include <Kokkos_Core.hpp>

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

/** ordered Grid array axes */
enum class /*HPG_EXPORT*/ GridAxis {
  x,
  y,
  mrow,
  channel
};

/** ordered CF array axes */
enum class /*HPG_EXPORT*/ CFAxis {
  x_major,
  y_major,
  mueller,
  channel,
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

/** visibilities plus metadata for gridding */
template <
  unsigned N,
  typename V,
  typename F,
  typename P,
  typename U,
  typename G>
struct /*HPG_EXPORT*/ VisData {

  static constexpr unsigned npol = N;

  enum : bool { is_visdata = true };

  using vis_t = V; // vis_t
  using freq_t = F; // vis_frequency_fp
  using ph_t = P; // vis_phase_fp
  using uvw_t = U; // impl::uvw_t
  using phgrad_t = G; // cf_phase_gradient_fp

  KOKKOS_INLINE_FUNCTION VisData() {};

  KOKKOS_INLINE_FUNCTION VisData(
    const K::Array<V, N>& values,
    F freq,
    P d_phase,
    const U uvw,
    const K::Array<unsigned, 2>& cf_index,
    const K::Array<G, 2>& cf_phase_gradient)
    : m_values(values)
    , m_freq(freq)
    , m_d_phase(d_phase)
    , m_uvw(uvw)
    , m_cf_index(cf_index)
    , m_cf_phase_gradient(cf_phase_gradient) {}

  VisData(VisData const&) = default;

  VisData(VisData&&) noexcept = default;

  ~VisData() = default;

  VisData& operator=(VisData const&) = default;

  VisData& operator=(VisData&&) noexcept = default;

  K::Array<vis_t, N> m_values;
  freq_t m_freq;
  ph_t m_d_phase;
  uvw_t m_uvw;
  K::Array<unsigned, 2> m_cf_index;
  K::Array<phgrad_t, 2> m_cf_phase_gradient;
};

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
template <typename T>
/*HPG_EXPORT*/ KOKKOS_INLINE_FUNCTION std::tuple<int, int, int, int>
compute_vis_coord(
  int g_size,
  int oversampling,
  int cf_radius,
  T coord,
  T inv_lambda,
  T grid_scale) {

  const T position = grid_scale * coord * inv_lambda + g_size / 2.0;
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
template <typename execution_space, typename VD, typename GS>
struct /*HPG_EXPORT*/ Vis final {

  static constexpr unsigned npol = VD::npol;
  using vis_t = typename VD::vis_t;
  using freq_t = typename VD::freq_t;
  using ph_t = typename VD::ph_t;
  using uvw_t = typename VD::uvw_t;
  using phgrad_t = typename VD::phgrad_t;

  int m_grid_coord[2]; /**< grid coordinate */
  int m_cf_minor[2]; /**< CF minor coordinate */
  int m_cf_major[2]; /**< CF major coordinate */
  int m_fine_offset[2]; /**< visibility position - nearest major grid */
  int m_cf_size[2]; /**< cf size */
  K::Array<vis_t, npol> m_values; /**< visibility values */
  K::complex<ph_t> m_phasor;
  int m_cf_channel; /**< cf channel index */
  int m_cf_grp; /**< cf group index */
  bool m_pos_w; /**< true iff W coordinate is strictly positive */
  phgrad_t m_phi0[2]; /**< phase screen value origin */
  phgrad_t m_dphi[2]; /**< phase screen value increment */

  KOKKOS_INLINE_FUNCTION Vis() {};

  KOKKOS_INLINE_FUNCTION Vis(
    const VD& vis,
    const K::Array<vis_t, npol>& vals,
    const K::Array<int, 2>& grid_size,
    const K::Array<GS, 2>& grid_scale,
    const K::Array<K::Array<int, 2>, HPG_MAX_NUM_CF_GROUPS>& cf_radii,
    const K::Array<int, 2>& oversampling)
    : m_values(vals)
    , m_phasor(cphase<execution_space>(vis.m_d_phase))
    , m_cf_channel(vis.m_cf_index[0])
    , m_cf_grp(vis.m_cf_index[1])
    , m_pos_w(vis.m_uvw[2] > 0) {

    static const freq_t c = 299792458.0;
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
template <
  unsigned N,
  typename execution_space,
  typename CFView,
  typename MIndexView,
  typename VisdataView,
  typename VisWgtValuesView,
  typename VisWgtColIndexView,
  typename VisWgtRowIndexView,
  typename GVisbuffView,
  typename GridScale,
  typename GridView,
  typename GridWeightView,
  typename ModelView>
struct /*HPG_EXPORT*/ VisibilityGridder final {

  static_assert(CFView::rank == 6);
  static_assert(VisdataView::value_type::is_visdata);
  static_assert(VisdataView::value_type::npol == N);
  static_assert(VisdataView::rank == 1);

  using visdata_t = typename VisdataView::value_type;
  using visibility_t = typename visdata_t::vis_t;
  using visibility_fp_t = typename visibility_t::value_type;

  static_assert(VisWgtValuesView::rank == 1);
  static_assert(VisWgtColIndexView::rank == 1);
  static_assert(VisWgtRowIndexView::rank == 1);

  static_assert(
    std::is_same_v<
      typename GVisbuffView::value_type,
      poln_array_type<visibility_fp_t, 4>>);
  static_assert(GVisbuffView::rank == 1);

  static_assert(
    std::is_same_v<typename MIndexView::non_const_data_type, int[4][4]>);
  static_assert(ModelView::rank == 4);
  static_assert(GridView::rank == 4);
  static_assert(GridWeightView::rank == 2);

  using member_type = typename K::TeamPolicy<execution_space>::member_type;

  using scratch_phscr_view =
    K::View<
      typename visdata_t::phgrad_t*,
      typename execution_space::scratch_memory_space>;

  using cf_const_view = typename CFView::const_type;
  using mindex_const_view = typename MIndexView::const_type;
  using model_const_view = typename ModelView::const_type;
  using viswgt_const_view = typename VisWgtValuesView::const_type;
  using viswgt_col_index_const_view = typename VisWgtColIndexView::const_type;
  using viswgt_row_index_const_view = typename VisWgtRowIndexView::const_type;

  using cf_value_t = typename CFView::non_const_value_type;
  using acc_vis_value_t = acc_cpx_t<visibility_t>;
  using acc_cf_value_t = acc_cpx_t<cf_value_t>;
  using viswgt_value_t = typename VisWgtValuesView::non_const_value_type;
  using viswgt_row_index_t = typename VisWgtRowIndexView::non_const_value_type;
  using grid_value_t = typename GridView::non_const_value_type;
  using model_value_t = typename ModelView::non_const_value_type;
  using grid_weight_value_t = typename GridWeightView::non_const_value_type;

  using vis_t = Vis<execution_space, visdata_t, GridScale>;

  execution_space m_exec;
  K::Array<cf_const_view, HPG_MAX_NUM_CF_GROUPS> m_cfs;
  K::Array<K::Array<int, 2>, HPG_MAX_NUM_CF_GROUPS> m_cf_radii;
  unsigned m_max_cf_extent_y;
  mindex_const_view m_mueller_indexes;
  mindex_const_view m_conjugate_mueller_indexes;
  int m_num_visibilities;
  VisdataView m_visibilities;
  viswgt_const_view m_viswgts;
  viswgt_col_index_const_view m_viswgt_col_index;
  viswgt_row_index_const_view m_viswgt_row_index;
  GVisbuffView m_gvisbuff;
  K::Array<GridScale, 2> m_grid_scale;
  GridView m_grid;
  GridWeightView m_grid_weights;
  model_const_view m_model;

  K::Array<int, 2> m_grid_size;
  K::Array<int, 2> m_oversampling;
  size_t m_shmem_size;

  unsigned m_cube_offset;
  /** constructor
   *
   * first argument is needed for complete class template argument deduction
   */
  VisibilityGridder(
    const std::integral_constant<unsigned, N>&,
    const execution_space& exec,
    const K::Array<CFView, HPG_MAX_NUM_CF_GROUPS>& cfs,
    const K::Array<K::Array<int, 2>, HPG_MAX_NUM_CF_GROUPS>& cf_radii,
    unsigned max_cf_extent_y,
    const MIndexView& mueller_indexes,
    const MIndexView& conjugate_mueller_indexes,
    int num_visibilities,
    const VisdataView& visibilities,
    const VisWgtValuesView& viswgts,
    const VisWgtColIndexView& viswgt_col_index,
    const VisWgtRowIndexView& viswgt_row_index,
    const GVisbuffView& gvisbuff,
    const K::Array<GridScale, 2>& grid_scale,
    const GridView& grid,
    const GridWeightView& grid_weights,
    const ModelView& model,
    unsigned cube_offset)
  : m_exec(exec)
  , m_cf_radii(cf_radii)
  , m_max_cf_extent_y(max_cf_extent_y)
  , m_mueller_indexes(mueller_indexes)
  , m_conjugate_mueller_indexes(conjugate_mueller_indexes)
  , m_num_visibilities(num_visibilities)
  , m_visibilities(visibilities)
  , m_viswgts(viswgts)
  , m_viswgt_col_index(viswgt_col_index)
  , m_viswgt_row_index(viswgt_row_index)
  , m_gvisbuff(gvisbuff)
  , m_grid_scale(grid_scale)
  , m_grid(grid)
  , m_grid_weights(grid_weights)
  , m_model(model) {

    for (size_t i = 0; i < HPG_MAX_NUM_CF_GROUPS; ++i)
      m_cfs[i] = cfs[i];
    m_grid_size = {
      m_grid.extent_int(int(GridAxis::x)),
      m_grid.extent_int(int(GridAxis::y))};
    m_oversampling = {
      m_cfs[0].extent_int(int(CFAxis::x_minor)),
      m_cfs[0].extent_int(int(CFAxis::y_minor))};
    m_shmem_size = scratch_phscr_view::shmem_size(m_max_cf_extent_y);
  }

  static KOKKOS_INLINE_FUNCTION bool
  all_within_grid(
    const vis_t& vis,
    const K::Array<int, 2>& grid_size) {

    return
      (0 <= vis.m_grid_coord[0])
      && (vis.m_grid_coord[0] + vis.m_cf_size[0] <= grid_size[0])
      && (0 <= vis.m_grid_coord[1])
      && (vis.m_grid_coord[1] + vis.m_cf_size[1] <= grid_size[1]);
  }

  static KOKKOS_INLINE_FUNCTION void
  compute_phase_screen(
    const member_type& team_member,
    const vis_t& vis,
    const scratch_phscr_view& phi_Y) {

    // compute the values of the phase screen along the Y axis now and store the
    // results in scratch memory because gridding on the Y axis accesses the
    // phase screen values for every row of the Mueller matrix column
    K::parallel_for(
      K::TeamVectorRange(team_member, vis.m_cf_size[1]),
      [=](const int Y) {
        phi_Y(Y) = vis.m_phi0[1] + Y * vis.m_dphi[1];
      });
    team_member.team_barrier();
  }

  static KOKKOS_INLINE_FUNCTION poln_array_type<visibility_fp_t, N>
  degrid_vis(
    const member_type& team_member,
    const vis_t& vis,
    const int& grid_channel,
    const cf_const_view& cf,
    const mindex_const_view& mueller_indexes,
    const mindex_const_view& conjugate_mueller_indexes,
    const model_const_view& model,
    const scratch_phscr_view& phi_Y) {

    const auto& N_X = vis.m_cf_size[0];
    const auto& N_Y = vis.m_cf_size[1];
    const auto N_R = model.extent_int(int(GridAxis::mrow));

    auto degridding_mindex =
      vis.m_pos_w ? conjugate_mueller_indexes : mueller_indexes;
    cf_fp cf_im_factor = (vis.m_pos_w ? 1 : -1);

    poln_array_type<visibility_fp_t, N> result;

    if (model.is_allocated()) {
      // model degridding
      static_assert(std::is_same_v<acc_vis_value_t, acc_cf_value_t>);
      vis_array_type<typename acc_vis_value_t::value_type, N> vis_array;

      // 3d (X, Y, Mueller) subspace of CF for this visibility
      auto cf_vis =
        K::subview(
          cf,
          K::pair<int, int>(vis.m_cf_major[0], vis.m_cf_major[0] + N_X),
          K::pair<int, int>(vis.m_cf_major[1], vis.m_cf_major[1] + N_Y),
          K::ALL,
          vis.m_cf_channel,
          vis.m_cf_minor[0],
          vis.m_cf_minor[1]);

      // 3d (X, Y, pol) subspace of model for this visibility
      auto model_vis =
        K::subview(
          model,
          K::pair<int, int>(vis.m_grid_coord[0], vis.m_grid_coord[0] + N_X),
          K::pair<int, int>(vis.m_grid_coord[1], vis.m_grid_coord[1] + N_Y),
          K::ALL,
          grid_channel);

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
                  cf_value_t cfv = cf_vis(X, Y, mindex);
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
           / ((vis_array.wgt[vpol] != acc_cf_value_t(0))
              ? vis_array.wgt[vpol]
              : acc_cf_value_t(1)))
          * conj_phasor;
    }
    return result;
  }

  //
  // DegridAll
  //
  // degrid all visibilities in m_visibilities to m_gvisbuff
  //

  struct DegridAll{};

  int
  league_size(const DegridAll&) const {
    return m_num_visibilities;
  }

  KOKKOS_INLINE_FUNCTION void
  operator()(const DegridAll&, const member_type& team_member) const {

    auto i = team_member.league_rank();
    auto& visibility = m_visibilities(i);

    vis_t vis(
      visibility,
      visibility.m_values,
      m_grid_size,
      m_grid_scale,
      m_cf_radii,
      m_oversampling);
    poln_array_type<visibility_fp_t, N> gvis;
    if (all_within_grid(vis, m_grid_size)) {
      auto phscr =
        scratch_phscr_view(
          team_member.team_scratch(0),
          m_max_cf_extent_y);
      compute_phase_screen(team_member, vis, phscr);
      for (int j = m_viswgt_row_index(i); j < m_viswgt_row_index(i + 1); ++j) {
        if (m_cube_offset <= m_viswgt_col_index(j)
            && m_viswgt_col_index(j) < m_cube_offset + m_cube_size FIXME) {
          auto v =
            degrid_vis(
              team_member,
              vis,
              m_viswgt_col_index(j),
              m_cfs[vis.m_cf_grp] - m_cube_offset,
              m_mueller_indexes,
              m_conjugate_mueller_indexes,
              m_model,
              phscr);
          K::single(K::PerTeam(team_member), [&](){ gvis += v; });
        }
      }
    } else {
      K::single(K::PerTeam(team_member), [&](){ gvis.set_nan(); });
    }
    K::single(
      K::PerTeam(team_member),
      [&]() {
        reinterpret_cast<poln_array_type<visibility_fp_t, N>&>(m_gvisbuff(i)) =
          gvis;
      });
  }

  void
  degrid_all() const {
    K::parallel_for(
      "degrid_all",
      K::TeamPolicy<execution_space, DegridAll>(
        m_exec,
        league_size(DegridAll()),
        K::AUTO)
      .set_scratch_size(0, K::PerTeam(m_shmem_size)),
      *this);
  }

  //
  // VisCopyResidualAndRescale
  //
  // copy residual visibilities to m_visibilities, then prepare m_gvisbuff for
  // gridding by rescaling visibilities
  //

  struct VisCopyResidualAndRescale{};

  int
  league_size(const VisCopyResidualAndRescale&) const {
    return m_num_visibilities;
  }

  // compute residual visibilities, prepare values for gridding
  KOKKOS_INLINE_FUNCTION void
  operator()(const VisCopyResidualAndRescale&, const member_type& team_member)
    const {
    auto i = team_member.league_rank();
    auto& vis = m_visibilities(i);
    auto phasor = cphase<execution_space>(vis.m_d_phase);
    auto& gvis = m_gvisbuff(i);
    K::parallel_for(
      K::TeamThreadRange(team_member, N),
      [&](const int vpol) {
        vis.m_values[vpol] -= gvis.vals[vpol];
        gvis.vals[vpol] = vis.m_values[vpol] * phasor;
      });
  }

  void
  vis_copy_residual_and_rescale() const {
    K::parallel_for(
      "vis_copy_residual_and_rescale",
      K::TeamPolicy<execution_space, VisCopyResidualAndRescale>(
        m_exec,
        league_size(VisCopyResidualAndRescale()),
        K::AUTO),
      *this);
  }

  //
  // VisRescale
  //
  // prepare m_gvisbuff for gridding by rescaling visibilities
  //

  struct VisRescale{};

  int
  league_size(const VisRescale&) const {
    return m_num_visibilities;
  }

  // prepare values for gridding
  KOKKOS_INLINE_FUNCTION void
  operator()(const VisRescale&, const member_type& team_member) const {
    auto i = team_member.league_rank();
    auto& vis = m_visibilities(i);
    auto phasor = cphase<execution_space>(vis.m_d_phase);
    auto& gvis = m_gvisbuff(i);
    K::parallel_for(
      K::TeamThreadRange(team_member, N),
      [&](const int vpol) {
        gvis.vals[vpol] = vis.m_values[vpol] * phasor;
      });
  }

  void
  vis_rescale() const {
    K::parallel_for(
      "vis_rescale",
      K::TeamPolicy<execution_space, VisRescale>(
        m_exec,
        league_size(VisRescale()),
        K::AUTO),
      *this);
  }

  //
  // VisCopyPredicted
  //
  // copy visibilities from m_gvisbuff to m_visibilities
  //

  struct VisCopyPredicted{};

  int
  league_size(const VisCopyPredicted&) const {
    return m_num_visibilities;
  }

  // copy predicted visibilities
  KOKKOS_INLINE_FUNCTION void
  operator()(const VisCopyPredicted&, const member_type& team_member) const {

    auto i = team_member.league_rank();
    auto& vis = m_visibilities(i);
    auto& gvis = m_gvisbuff(i);
    K::parallel_for(
      K::TeamThreadRange(team_member, N),
      [&](const int vpol) {
        vis.m_values[vpol] = gvis.vals[vpol];
      });
  }

  void
  vis_copy_predicted() const {
    K::parallel_for(
      "vis_copy_predicted",
      K::TeamPolicy<execution_space, VisCopyPredicted>(
        m_exec,
        league_size(VisCopyPredicted()),
        K::AUTO),
      *this);
  }

  // function for gridding a single visibility with sum of weights
  static KOKKOS_INLINE_FUNCTION void
  grid_vis(
    const member_type& team_member,
    const vis_t& vis,
    const int& grid_channel,
    const viswgt_value_t& viswgt,
    const int& gpol,
    const cf_const_view& cf,
    const mindex_const_view& mueller_indexes,
    const mindex_const_view& conjugate_mueller_indexes,
    const GridView& grid,
    const GridWeightView& grid_weights,
    const scratch_phscr_view& phi_Y) {

    const auto& N_X = vis.m_cf_size[0];
    const auto& N_Y = vis.m_cf_size[1];

    auto gridding_mindex =
      K::subview(
        (vis.m_pos_w ? mueller_indexes : conjugate_mueller_indexes),
        gpol,
        K::ALL);
    cf_fp cf_im_factor = (vis.m_pos_w ? -1 : 1);

    // 3d (X, Y, Mueller) subspace of CF for this visibility
    auto cf_vis =
      K::subview(
        cf,
        K::pair<int, int>(vis.m_cf_major[0], vis.m_cf_major[0] + N_X),
        K::pair<int, int>(vis.m_cf_major[1], vis.m_cf_major[1] + N_Y),
        K::ALL,
        vis.m_cf_channel,
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
        grid_channel);

    // accumulate to grid, and CF weights per visibility polarization
    poln_array_type<typename acc_cf_value_t::value_type, N> grid_wgt;
    // parallel loop over grid X
    K::parallel_reduce(
      K::TeamThreadRange(team_member, N_X),
      [=](const int X, decltype(grid_wgt)& grid_wgt_l) {
        auto phi_X = vis.m_phi0[0] + X * vis.m_dphi[0];
        // loop over grid Y
        for (int Y = 0; Y < N_Y; ++Y) {
          const auto wgt_screen =
            viswgt * cphase<execution_space>(phi_X + phi_Y(Y));
          grid_value_t gv(0);
          // loop over visibility polarizations
          for (int vpol = 0; vpol < N; ++vpol) {
            if (const auto mindex = gridding_mindex(vpol); mindex >= 0) {
              cf_value_t cfv = cf_vis(X, Y, mindex);
              cfv.imag() *= cf_im_factor;
              gv += grid_value_t(cfv * wgt_screen * vis.m_values[vpol]);
              grid_wgt_l.vals[vpol] += cfv;
            }
          }
          pseudo_atomic_add<execution_space>(grd_vis(X, Y), gv);
        }
      },
      K::Sum<decltype(grid_wgt)>(grid_wgt));
    // compute final weight and add it to grid weights
    K::single(
      K::PerTeam(team_member),
      [&]() {
        grid_weight_value_t twgt = 0;
        for (int vpol = 0; vpol < N; ++vpol)
          twgt += grid_weight_value_t(mag(grid_wgt.vals[vpol]) * viswgt);
        K::atomic_add(&grid_weights(gpol, grid_channel), twgt);
      });
  }

  // function for gridding a single visibility without sum of weights
  static KOKKOS_INLINE_FUNCTION void
  grid_vis_no_weights(
    const member_type& team_member,
    const vis_t& vis,
    const int& grid_channel,
    const viswgt_value_t& viswgt,
    const int& gpol,
    const cf_const_view& cf,
    const mindex_const_view& mueller_indexes,
    const mindex_const_view& conjugate_mueller_indexes,
    const GridView& grid,
    const scratch_phscr_view& phi_Y) {

    const auto& N_X = vis.m_cf_size[0];
    const auto& N_Y = vis.m_cf_size[1];

    auto gridding_mindex =
      K::subview(
        (vis.m_pos_w ? mueller_indexes : conjugate_mueller_indexes),
        gpol,
        K::ALL);
    cf_fp cf_im_factor = (vis.m_pos_w ? -1 : 1);

    // 3d (X, Y, Mueller) subspace of CF for this visibility
    auto cf_vis =
      K::subview(
        cf,
        K::pair<int, int>(vis.m_cf_major[0], vis.m_cf_major[0] + N_X),
        K::pair<int, int>(vis.m_cf_major[1], vis.m_cf_major[1] + N_Y),
        K::ALL,
        vis.m_cf_channel,
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
        grid_channel);

    // parallel loop over grid X
    K::parallel_for(
      K::TeamThreadRange(team_member, N_X),
      [=](const int X) {
        auto phi_X = vis.m_phi0[0] + X * vis.m_dphi[0];
        // loop over grid Y
        for (int Y = 0; Y < N_Y; ++Y) {
          const auto wgt_screen =
            viswgt * cphase<execution_space>(phi_X + phi_Y(Y));
          grid_value_t gv(0);
          // loop over visibility polarizations
          for (int vpol = 0; vpol < N; ++vpol) {
            if (const auto mindex = gridding_mindex(vpol); mindex >= 0) {
              cf_value_t cfv = cf_vis(X, Y, mindex);
              cfv.imag() *= cf_im_factor;
              gv += grid_value_t(cfv * wgt_screen * vis.m_values[vpol]);
            }
          }
          pseudo_atomic_add<execution_space>(grd_vis(X, Y), gv);
        }
      });
  }

  template <typename F>
  KOKKOS_INLINE_FUNCTION void
  for_all_vis_on_grid(const member_type& team_member, const F& grid_one) const {

    const auto N_R = m_grid.extent_int(int(GridAxis::mrow));
    auto i = team_member.league_rank() / N_R;
    auto gpol = team_member.league_rank() % N_R;

    vis_t vis(
      m_visibilities(i),
      reinterpret_cast<K::Array<visibility_t, N>&>(m_gvisbuff(i).vals),
      m_grid_size,
      m_grid_scale,
      m_cf_radii,
      m_oversampling);
    // skip this visibility if all of the updated grid points are not
    // within grid bounds
    if (all_within_grid(vis, m_grid_size)) {
      auto phscr =
        scratch_phscr_view(
          team_member.team_scratch(0),
          m_max_cf_extent_y);
      compute_phase_screen(team_member, vis, phscr);
      for (int j = m_viswgt_row_index(i); j < m_viswgt_row_index(i + 1); ++j)
        if (m_cube_offset <= m_viswgt_col_index(j)
            && m_viswgt_col_index(j) < m_cube_offset + m_cube_size FIXME)
          grid_one(
            vis,
            m_viswgt_col_index(j) - m_cube_offset,
            m_viswgts(j),
            gpol,
            m_cfs[vis.m_cf_grp],
            phscr);
      }
  }

  //
  // GridAllNoWeights
  //
  // grid visibilities in m_gvisbuff, without updating m_weights
  //

  struct GridAllNoWeights{};

  int
  league_size(const GridAllNoWeights&) const {
    return m_grid.extent_int(int(GridAxis::mrow)) * m_num_visibilities;
  }

  KOKKOS_INLINE_FUNCTION void
  operator()(const GridAllNoWeights&, const member_type& team_member) const {
    for_all_vis_on_grid(
      team_member,
      [&](
        const vis_t& vis,
        const int& channel,
        const viswgt_value_t& wgt,
        const int& gpol,
        const cf_const_view& cf,
        const scratch_phscr_view& phi_Y) {
        grid_vis_no_weights(
          team_member,
          vis,
          channel,
          wgt,
          gpol,
          cf,
          m_mueller_indexes,
          m_conjugate_mueller_indexes,
          m_grid,
          phi_Y);
      });
  }

  void
  grid_all_no_weights() const {
    K::parallel_for(
      "gridding_no_weights",
      K::TeamPolicy<execution_space, GridAllNoWeights>(
        m_exec,
        league_size(GridAllNoWeights()),
        K::AUTO)
      .set_scratch_size(0, K::PerTeam(m_shmem_size)),
      *this);
  }

  //
  // GridAll
  //
  // grid visibilities in m_gvisbuff with updates to m_weights
  //

  struct GridAll{};

  int
  league_size(const GridAll&) const {
    return m_grid.extent_int(int(GridAxis::mrow)) * m_num_visibilities;
  }

  KOKKOS_INLINE_FUNCTION void
  operator()(const GridAll&, const member_type team_member) const {
    for_all_vis_on_grid(
      team_member,
      [&](
        const vis_t& vis,
        const int& channel,
        const viswgt_value_t& wgt,
        const int& gpol,
        const cf_const_view& cf,
        const scratch_phscr_view& phi_Y) {
        grid_vis(
          team_member,
          vis,
          channel,
          wgt,
          gpol,
          cf,
          m_mueller_indexes,
          m_conjugate_mueller_indexes,
          m_grid,
          m_grid_weights,
          phi_Y);
      });
  }

  void
  grid_all() const {
    K::parallel_for(
      "gridding_weights",
      K::TeamPolicy<execution_space, GridAll>(
        m_exec,
        league_size(GridAll()),
        K::AUTO)
      .set_scratch_size(0, K::PerTeam(m_shmem_size)),
      *this);
  }
};

/** grid normalization
 */
template <
  typename execution_space,
  typename GridView,
  typename GridWeightView>
struct /*HPG_EXPORT*/ GridNormalizer final {

  static_assert(GridView::rank == 4);
  static_assert(GridWeightView::rank == 2);

  static_assert(
    int(GridAxis::x) == 0
    && int(GridAxis::y) == 1
    && int(GridAxis::mrow) == 2
    && int(GridAxis::channel) == 3);
  static_assert(
    GridWeightArray::Axis::mrow == 0 && GridWeightArray::Axis::channel == 1);

  using grid_weight_const_view_t = typename GridWeightView::const_type;
  using grid_value_t = typename GridView::non_const_value_type;
  using weight_value_t = typename GridWeightView::non_const_value_type;

  execution_space m_exec;
  GridView m_grid;
  grid_weight_const_view_t m_grid_weights;
  weight_value_t m_inv_norm;

  GridNormalizer(
    const execution_space& exec,
    const GridView& grid,
    const GridWeightView& grid_weights,
    weight_value_t wfactor)
    : m_exec(exec)
    , m_grid(grid)
    , m_grid_weights(grid_weights)
    , m_inv_norm(weight_value_t(1) / wfactor) {}

  GridNormalizer(
    const execution_space& exec,
    const GridView& grid,
    weight_value_t norm)
    : m_exec(exec)
    , m_grid(grid)
    , m_inv_norm(weight_value_t(1) / norm) {}

  //
  // ByWeights
  //
  // normalize grid by sum of weights
  //

  struct ByWeights{};

  KOKKOS_INLINE_FUNCTION void
  operator()(const ByWeights&, int x, int y, int mrow, int channel) const {

    m_grid(x, y, mrow, channel) =
      (m_grid(x, y, mrow, channel) * m_inv_norm)
      / m_grid_weights(mrow, channel);
  }

  //
  // ByValue
  //
  // normalize grid by a value
  //

  struct ByValue{};

  KOKKOS_INLINE_FUNCTION void
  operator()(const ByValue&, int x, int y, int mrow, int channel) const {

    m_grid(x, y, mrow, channel) *= m_inv_norm;
  }

  void
  normalize() const {
    const K::Array<int, 4> begin{0, 0, 0, 0};
    const K::Array<int, 4> end{
      m_grid.extent_int(int(GridAxis::x)),
      m_grid.extent_int(int(GridAxis::y)),
      m_grid.extent_int(int(GridAxis::mrow)),
      m_grid.extent_int(int(GridAxis::channel))};

    if (m_grid_weights.is_allocated())
      K::parallel_for(
        "normalize_by_weights",
        K::MDRangePolicy<K::Rank<4>, execution_space, ByWeights>(
          m_exec,
          begin,
          end),
        *this);
    else if (m_inv_norm != weight_value_t(1))
      K::parallel_for(
        "normalize_by_value",
        K::MDRangePolicy<K::Rank<4>, execution_space, ByValue>(
          m_exec,
          begin,
          end),
        *this);
  }
};

// deduction guide for "weights-less" constructor
template <typename execution_space, typename GridView, typename T>
GridNormalizer(const execution_space&, const GridView&, T) ->
  GridNormalizer<execution_space, GridView, K::View<T**>>;

/** swap visibility values */
#pragma nv_exec_check_disable
template <typename execution_space, typename T>
/*HPG_EXPORT*/ KOKKOS_FORCEINLINE_FUNCTION void
swap_gv(T& a, T&b) {
  std::swap(a, b);
}

#ifdef HPG_ENABLE_CUDA
template <>
/*HPG_EXPORT*/ KOKKOS_FORCEINLINE_FUNCTION void
swap_gv<K::Cuda, K::complex<float>>(
  K::complex<float>& a,
  K::complex<float>& b) {

  auto tmp = a;
  a = b;
  b = tmp;
}
template <>
/*HPG_EXPORT*/ KOKKOS_FORCEINLINE_FUNCTION void
swap_gv<K::Cuda, K::complex<double>>(
  K::complex<double>& a,
  K::complex<double>& b) {

  auto tmp = a;
  a = b;
  b = tmp;
}
#endif // HPG_ENABLE_CUDA

/** grid rotation
 *
 * Useful after FFT to shift grid planes by half the grid plane size in each
 * dimension
 */
template <typename execution_space, typename GridView>
struct /*HPG_EXPORT*/ GridShifter final {

  static_assert(GridView::rank == 4);

  using member_type = typename K::TeamPolicy<execution_space>::member_type;

  using grid_value_t = typename GridView::non_const_value_type;

  execution_space m_exec;
  ShiftDirection m_direction;
  GridView m_grid;

  int m_n_x;
  int m_n_y;
  int m_n_mrow;
  int m_n_channel;
  int m_mid_x;
  int m_mid_y;

  // TODO: is this kernel valid for all possible GridAxis definitions?
  static_assert(
    int(GridAxis::x) == 0
    && int(GridAxis::y) == 1
    && int(GridAxis::mrow) == 2
    && int(GridAxis::channel) == 3);

  GridShifter(
    const execution_space& exec,
    ShiftDirection direction,
    const GridView& grid)
    : m_exec(exec)
    , m_direction(direction)
    , m_grid(grid)
    , m_n_x(m_grid.extent_int(0))
    , m_n_y(m_grid.extent_int(1))
    , m_n_mrow(m_grid.extent_int(2))
    , m_n_channel(m_grid.extent_int(3))
    , m_mid_x(m_n_x / 2)
    , m_mid_y(m_n_y / 2) {}

  int
  league_size() const {
    return m_n_mrow * m_n_channel;
  }

  //
  // EvenEven
  //
  // both x and y dimensions of grid have even length
  //

  struct EvenEven{};

  KOKKOS_INLINE_FUNCTION void
  operator()(const EvenEven&, const member_type& team_member) const {
    auto i = team_member.league_rank();
    auto gplane =
      K::subview(m_grid, K::ALL, K::ALL, i % m_n_mrow, i / m_n_mrow);
    K::parallel_for(
      K::TeamVectorRange(team_member, m_n_x / 2),
      [=](int x) {
        for (int y = 0; y < m_n_y / 2; ++y) {
          swap_gv<execution_space>(
            gplane(x, y),
            gplane(x + m_mid_x, y + m_mid_y));
          swap_gv<execution_space>(
            gplane(x + m_mid_x, y),
            gplane(x, y + m_mid_y));
        }
      });
  }

  //
  // OddSquareForward
  //
  // both x and y dimensions of grid have same length, and are odd; rotate
  // "forward"
  //

  struct OddSquareForward{};

  KOKKOS_INLINE_FUNCTION void
  operator()(const OddSquareForward&, const member_type& team_member) const {
    auto i = team_member.league_rank();
    auto gplane =
      K::subview(m_grid, K::ALL, K::ALL, i % m_n_mrow, i / m_n_mrow);
    K::parallel_for(
      K::TeamVectorRange(team_member, m_n_x),
      [=](int x) {
        grid_value_t tmp;
        int y = 0;
        for (int i = 0; i <= m_n_y; ++i) {
          swap_gv<execution_space>(tmp, gplane(x, y));
          x += m_mid_x;
          if (x >= m_n_x)
            x -= m_n_x;
          y += m_mid_y;
          if (y >= m_n_y)
            y -= m_n_y;
        }
      });
  }

  //
  // OddSquareBackward
  //
  // both x and y dimensions of grid have same length, and are odd; rotate
  // "backward"
  //

  struct OddSquareBackward{};

  KOKKOS_INLINE_FUNCTION void
  operator()(const OddSquareBackward&, const member_type& team_member) const {
    auto i = team_member.league_rank();
    auto gplane =
      K::subview(m_grid, K::ALL, K::ALL, i % m_n_mrow, i / m_n_mrow);
    K::parallel_for(
      K::TeamVectorRange(team_member, m_n_x),
      [=](int x) {
        grid_value_t tmp;
        int y = 0;
        for (int i = 0; i <= m_n_y; ++i) {
          swap_gv<execution_space>(tmp, gplane(x, y));
          x -= m_mid_x;
          if (x < 0)
            x += m_n_x;
          y -= m_mid_y;
          if (y < 0)
            y += m_n_y;
        }
      });
  }

  //
  // GenForward
  //
  // generic shift "forward" algorithm
  //

  struct GenForward{};

  KOKKOS_INLINE_FUNCTION void
  operator()(const GenForward&, const member_type& team_member) const {
    auto i = team_member.league_rank();
    auto gplane =
      K::subview(m_grid, K::ALL, K::ALL, i % m_n_mrow, i / m_n_mrow);

    // first pass, parallel over x
    if (m_n_y % 2 == 1)
      K::parallel_for(
        K::TeamThreadRange(team_member, m_n_x),
        [=](int x) {
          grid_value_t tmp;
          int y = 0;
          for (int i = 0; i <= m_n_y; ++i) {
            swap_gv<execution_space>(tmp, gplane(x, y));
            y += m_mid_y;
            if (y >= m_n_y)
              y -= m_n_y;
          }
        });
    else
      K::parallel_for(
        K::TeamThreadRange(team_member, m_n_x),
        [=](int x) {
          for (int y = 0; y < m_mid_y; ++y)
            swap_gv<execution_space>(
              gplane(x, y),
              gplane(x, y + m_mid_y));
        });

    // second pass, parallel over y
    if (m_n_x % 2 == 1)
      K::parallel_for(
        K::TeamThreadRange(team_member, m_n_y),
        [=](int y) {
          grid_value_t tmp;
          int x = 0;
          for (int i = 0; i <= m_n_x; ++i) {
            swap_gv<execution_space>(tmp, gplane(x, y));
            x += m_mid_x;
            if (x >= m_n_x)
              x -= m_n_x;
          }
        });
    else
      K::parallel_for(
        K::TeamThreadRange(team_member, m_n_y),
        [=](int y) {
          for (int x = 0; x < m_mid_x; ++x)
            swap_gv<execution_space>(
              gplane(x, y),
              gplane(x + m_mid_x, y));
        });
  }

  //
  // GenBackward
  //
  // generic shift "backward" algorithm
  //

  struct GenBackward{};

  KOKKOS_INLINE_FUNCTION void
  operator()(const GenBackward&, const member_type& team_member) const {
    auto i = team_member.league_rank();
    auto gplane =
      K::subview(m_grid, K::ALL, K::ALL, i % m_n_mrow, i / m_n_mrow);

    // first pass, parallel over x
    if (m_n_y % 2 == 1)
      K::parallel_for(
        K::TeamThreadRange(team_member, m_n_x),
        [=](int x) {
          grid_value_t tmp;
          int y = 0;
          for (int i = 0; i <= m_n_y; ++i) {
            swap_gv<execution_space>(tmp, gplane(x, y));
            y -= m_mid_y;
            if (y < 0)
              y += m_n_y;
          }
        });
    else
      K::parallel_for(
        K::TeamThreadRange(team_member, m_n_x),
        [=](int x) {
          for (int y = m_mid_y; y < m_n_y; ++y)
            swap_gv<execution_space>(
              gplane(x, y),
              gplane(x, y - m_mid_y));
        });

    // second pass, parallel over y
    if (m_n_x % 2 == 1)
      K::parallel_for(
        K::TeamThreadRange(team_member, m_n_y),
        [=](int y) {
          grid_value_t tmp;
          int x = 0;
          for (int i = 0; i <= m_n_x; ++i) {
            swap_gv<execution_space>(tmp, gplane(x, y));
            x -= m_mid_x;
            if (x < 0)
              x += m_n_x;
          }
        });
    else
      K::parallel_for(
        K::TeamThreadRange(team_member, m_n_y),
        [=](int y) {
          for (int x = m_mid_x; x < m_n_x; ++x)
            swap_gv<execution_space>(
              gplane(x, y),
              gplane(x - m_mid_x, y));
        });
  }

  void
  shift() const {
    if (m_n_x % 2 == 0 && m_n_y % 2 == 0)
      K::parallel_for(
        "grid_shift_ee",
        K::TeamPolicy<execution_space, EvenEven>(
          m_exec,
          league_size(),
          K::AUTO),
        *this);
    else if (m_n_x == m_n_y) {
      if (m_direction == ShiftDirection::FORWARD)
        K::parallel_for(
          "grid_shift_osf",
          K::TeamPolicy<execution_space, OddSquareForward>(
            m_exec,
            league_size(),
            K::AUTO),
          *this);
      else // m_direction == ShiftDirection::BACKWARD
        K::parallel_for(
          "grid_shift_osb",
          K::TeamPolicy<execution_space, OddSquareBackward>(
            m_exec,
            league_size(),
            K::AUTO),
          *this);
    } else {
      if (m_direction == ShiftDirection::FORWARD)
        K::parallel_for(
          "grid_shift_genf",
          K::TeamPolicy<execution_space, GenForward>(
            m_exec,
            league_size(),
            K::AUTO),
          *this);
      else // m_direction == ShiftDirection::BACKWARD
        K::parallel_for(
          "grid_shift_genb",
          K::TeamPolicy<execution_space, GenBackward>(
            m_exec,
            league_size(),
            K::AUTO),
          *this);
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
