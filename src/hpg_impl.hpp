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
#include "hpg_core.hpp"
// #inlude "hpg_export.h"

#include <cassert>
#include <numeric>
#include <memory>
#include <optional>
#include <ostream>
#include <tuple>
#include <vector>

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

#ifdef __NVCC__
# define WORKAROUND_NVCC_IF_CONSTEXPR_BUG
#endif

/** @file hpg_impl.hpp
 *
 * HPG implementation header file
 */
namespace hpg {

// TODO: reimplement CFIndex bounds checking
// struct OutOfBoundsCFIndexError
//   : public Error {

//   OutOfBoundsCFIndexError(const vis_cf_index_t& idx)
//     : Error(
//       "vis_cf_index_t value (" + std::to_string(std::get<0>(idx))
//       + "," + std::to_string(std::get<1>(idx))
//       + ") is out of bounds for current CFArray",
//       ErrorType::OutOfBoundsCFIndex) {}
// };

/** invalid model grid size error
 *
 * Dimensions of model grid do not equal those of (visibility) value grid.
 */
struct InvalidModelGridSizeError
  : public Error {

  /** constructor */
  InvalidModelGridSizeError(
    const Kokkos::Array<int, GridValueArray::rank>& model_size,
    const Kokkos::Array<int, GridValueArray::rank>& grid_size)
    : Error(
      "model grid size " + sz2str(model_size)
      + " is different from visibility grid size " + sz2str(grid_size),
      ErrorType::InvalidModelGridSize) {}

  /** array extents as string */
  static std::string
  sz2str(const Kokkos::Array<int, GridValueArray::rank>& sz) {
    std::ostringstream oss;
    oss << "[" << sz[0]
        << "," << sz[1]
        << "," << sz[2]
        << "," << sz[3]
        << "]" << std::endl;
    return oss.str();
  }
};

namespace runtime::impl {

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

bool
is_initialized() noexcept;

bool
initialize(const InitArguments& args);

void
finalize();

/** type trait associating Kokkos device with hpg Device */
template <Device D>
struct /*HPG_EXPORT*/ DeviceT {
  using kokkos_device = void;

  static constexpr const char* const name = "";
};

#ifdef HPG_ENABLE_SERIAL
/** Serial device type trait */
template <>
struct /*HPG_EXPORT*/ DeviceT<Device::Serial> {
  using kokkos_device = K::Serial;

  static constexpr const char* const name = "Serial";
};
#endif // HPG_ENABLE_SERIAL

#ifdef HPG_ENABLE_OPENMP
/** OpenMP device type trait */
template <>
struct /*HPG_EXPORT*/ DeviceT<Device::OpenMP> {
  using kokkos_device = K::OpenMP;

  static constexpr const char* const name = "OpenMP";
};
#endif // HPG_ENABLE_OPENMP

#ifdef HPG_ENABLE_CUDA
/** Cuda device type trait */
template <>
struct /*HPG_EXPORT*/ DeviceT<Device::Cuda> {
  using kokkos_device = K::Cuda;

  static constexpr const char* const name = "Cuda";
};
#endif // HPG_ENABLE_CUDA

/** axis order for strided grid layout */
static const std::array<int, 4> strided_grid_layout_order{
  int(core::GridAxis::y),
  int(core::GridAxis::mrow),
  int(core::GridAxis::x),
  int(core::GridAxis::channel)};

/** device-specific grid array layout */
template <typename Device>
struct /*HPG_EXPORT*/ GridLayout {

  /** Kokkos layout type */
  using layout =
    std::conditional_t<
      std::is_same_v<typename Device::array_layout, K::LayoutLeft>,
      K::LayoutLeft,
      K::LayoutStride>;

  // an assumption throughout this module
  static_assert(
    int(core::GridAxis::x) == 0
    && int(core::GridAxis::y) == 1
    && int(core::GridAxis::mrow) == 2
    && int(core::GridAxis::channel) == 3);

  /** create Kokkos layout using given grid dimensions
   *
   * logical index order matches GridAxis definition
   */
  static layout
  dimensions(const K::Array<int, 4>& dims) {
    if constexpr (std::is_same_v<layout, K::LayoutLeft>) {
      return K::LayoutLeft(dims[0], dims[1], dims[2], dims[3]);
    } else {
      return
        K::LayoutStride::order_dimensions(
          4,
          strided_grid_layout_order.data(),
          dims.data());
    }
#ifdef WORKAROUND_NVCC_IF_CONSTEXPR_BUG
    return layout();
#endif
  }
};

/** CFLayout version number
 *
 * @todo make something useful of this, maybe add a value template parameter to
 * CFLayout?
 */
static const unsigned cf_layout_version_number = 0;

/** axis order for strided CF array layout */
static const std::array<int, 6> strided_cf_layout_order{
  int(core::CFAxis::mueller),
  int(core::CFAxis::y_major),
  int(core::CFAxis::x_major),
  int(core::CFAxis::channel),
  int(core::CFAxis::x_minor),
  int(core::CFAxis::y_minor)};

/** device-specific constant-support CF array layout */
template <typename Device>
struct /*HPG_EXPORT*/ CFLayout {

  /** Kokkos layout type */
  using layout =
    std::conditional_t<
      std::is_same_v<typename Device::array_layout, K::LayoutLeft>,
      K::LayoutLeft,
      K::LayoutStride>;

  // an assumption throughout this module
  static_assert(
    int(core::CFAxis::x_major) == 0
    && int(core::CFAxis::y_major) == 1
    && int(core::CFAxis::mueller) == 2
    && int(core::CFAxis::channel) == 3
    && int(core::CFAxis::x_minor) == 4
    && int(core::CFAxis::y_minor) == 5);

  /**
   * create Kokkos layout using given CFArray slice
   *
   * logical index order matches CFAxis definition
   */
  static layout
  dimensions(const CFArrayShape& cf, unsigned grp) {
    auto extents = cf.extents(grp);
    auto os = cf.oversampling();
    std::array<int, 6> dims{
      int((extents[CFArray::Axis::x] + os - 1) / os),
      int((extents[CFArray::Axis::y] + os - 1) / os),
      int(extents[CFArray::Axis::mueller]),
      int(extents[CFArray::Axis::channel]),
      int(os),
      int(os)
    };
    if constexpr (std::is_same_v<layout, K::LayoutLeft>) {
      return
        K::LayoutLeft(dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]);
    } else {
      return
        K::LayoutStride::order_dimensions(
          6,
          strided_cf_layout_order.data(),
          dims.data());
    }
#ifdef WORKAROUND_NVCC_IF_CONSTEXPR_BUG
    return layout();
#endif
  }
};

/** sign of a value
 *
 * @return -1, if less than 0; +1, if greater than 0; 0, if equal to 0
 */
template <typename T>
inline constexpr int
sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

/** visibility value type */
using vis_t = K::complex<visibility_fp>;

/** convolution function value type */
using cf_t = K::complex<cf_fp>;

/** gridded value type */
using gv_t = K::complex<grid_value_fp>;

/** portable UVW coordinates type */
using uvw_t = K::Array<vis_uvw_fp, 3>;

/** type for (plain) vector data */
template <typename T>
using vector_data = std::shared_ptr<std::vector<T>>;

/** View type for grid values */
template <typename Layout, typename memory_space>
using grid_view = K::View<gv_t****, Layout, memory_space>;

/** View type for grid weight values
 *
 * logical axis order: mrow, channel
 */
template <typename Layout, typename memory_space>
using grid_weight_view = K::View<grid_value_fp**, Layout, memory_space>;

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

template <unsigned N>
using visdata_t =
  core::VisData<
    N,
    vis_t,
    vis_frequency_fp,
    vis_phase_fp,
    uvw_t,
    cf_phase_gradient_fp>;

/** view for VisData<N> */
template <unsigned N, typename memory_space>
using visdata_view =
  K::View<visdata_t<N>*, memory_space, K::MemoryTraits<K::Unmanaged>>;

/** view for predicted/residual visibilities */
template <unsigned N, typename memory_space>
using gvis_view =
  K::View<core::poln_array_type<visibility_fp, N>*, memory_space>;

/** view for Mueller element index matrix */
template <typename memory_space>
using mindex_view = K::View<int[4][4], memory_space>;

/** view for constant Mueller element index matrix */
template <typename memory_space>
using const_mindex_view =
  K::View<const int[4][4], memory_space, K::MemoryTraits<K::RandomAccess>>;

template <typename memory_space>
using weight_view_memory_traits =
  std::conditional_t<
    std::is_same_v<memory_space, K::HostSpace>,
    K::MemoryUnmanaged,
    K::MemoryManaged>;

/** view for values in CRS representation of weights */
template <typename memory_space>
using weight_values_view = K::View<vis_weight_fp*, memory_space>;

/** view for column index in CRS representation of weights */
template <typename memory_space>
using weight_col_index_view = K::View<unsigned*, memory_space>;

/** view for row index in CRS representation of weights */
template <typename memory_space>
using weight_row_index_view = K::View<size_t*, memory_space>;

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
 *
 * Because the implementations depend on specific grid layouts, we leave this in
 * the impl namespace.
 */
template <typename execution_space>
struct /*HPG_EXPORT*/ FFT final {

  // default implementation assumes FFTW3

  template <typename IG, typename OG>
  static auto
  grid_fft_handle(execution_space exec, FFTSign sign, IG& igrid, OG& ogrid) {

    using scalar_t = typename OG::value_type::value_type;

#ifdef HPG_ENABLE_OPENMP
# ifdef HPG_ENABLE_SERIAL
    if constexpr (std::is_same_v<execution_space, K::Serial>)
      FFTW<scalar_t>::plan_with_nthreads(1);
    else
# endif
      FFTW<scalar_t>::plan_with_nthreads(omp_get_max_threads());
#endif // HPG_ENABLE_OPENMP

    // this assumes there is no padding in grid
    assert(igrid.span() ==
           igrid.extent(0) * igrid.extent(1)
           * igrid.extent(2) * igrid.extent(3));
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
  static std::optional<std::unique_ptr<Error>>
  in_place_kernel(
    execution_space exec,
    FFTSign sign,
    const grid_view<grid_layout, memory_space>& grid) {

    using scalar_t =
      typename grid_view<grid_layout, memory_space>::value_type::value_type;

    auto handles = grid_fft_handle(exec, sign, grid, grid);
    auto& [h0, h1] = handles;
    std::optional<std::unique_ptr<Error>> result;
    if (h0 == nullptr || h1 == nullptr)
      result = std::make_unique<Error>("fftw in_place_kernel() failed");
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
  static std::optional<std::unique_ptr<Error>>
  out_of_place_kernel(
    execution_space exec,
    FFTSign sign,
    const typename grid_view<grid_layout, memory_space>::const_type& pre_grid,
    const grid_view<grid_layout, memory_space>& post_grid) {

    using scalar_t =
      typename grid_view<grid_layout, memory_space>::value_type::value_type;

    auto handles = grid_fft_handle(exec, sign, pre_grid, post_grid);
    auto& [h0, h1] = handles;
    std::optional<std::unique_ptr<Error>> result;
    if (h0 == nullptr || h1 == nullptr)
      result = std::make_unique<Error>("fftw in_place_kernel() failed");
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

/*HPG_EXPORT*/ std::unique_ptr<Error>
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
struct /*HPG_EXPORT*/ FFT<K::Cuda> final {

  template <typename G>
  static std::tuple<cufftResult_t, cufftHandle>
  grid_fft_handle(K::Cuda exec, G& grid) {

    using scalar_t = typename G::value_type::value_type;

    // this assumes there is no padding in grid
    assert(grid.span() ==
           grid.extent(0) * grid.extent(1) * grid.extent(2) * grid.extent(3));
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
  static std::optional<std::unique_ptr<Error>>
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
    std::optional<std::unique_ptr<Error>> result;
    if (rc != CUFFT_SUCCESS)
      result = cufft_error("Cuda in_place_kernel() failed: ", rc);
    return result;
  }

  /** out-of-place FFT kernel
   */
  template <typename grid_layout, typename memory_space>
  static std::optional<std::unique_ptr<Error>>
  out_of_place_kernel(
    K::Cuda exec,
    FFTSign sign,
    const typename grid_view<grid_layout, memory_space>::const_type& pre_grid,
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
    std::optional<std::unique_ptr<Error>> result;
    if (rc != CUFFT_SUCCESS)
      result = cufft_error("cuda out_of_place_kernel() failed: ", rc);
    return result;
  }
};
#endif // HPG_ENABLE_CUDA

template <typename Layout, typename memory_space>
struct /*HPG_EXPORT*/ GridWeightPtr
  : public std::enable_shared_from_this<GridWeightPtr<Layout, memory_space>> {

  grid_weight_view<Layout, memory_space> m_gw;

  GridWeightPtr(const grid_weight_view<Layout, memory_space>& gw)
    : m_gw(gw) {}

  std::shared_ptr<GridWeightArray::value_type>
  ptr() const {
    return
      std::shared_ptr<GridWeightArray::value_type>(
        this->shared_from_this(),
        reinterpret_cast<GridWeightArray::value_type*>(m_gw.data()));
  }

  virtual ~GridWeightPtr() {}
};

template <typename Layout, typename memory_space>
struct /*HPG_EXPORT*/ GridValuePtr
  : public std::enable_shared_from_this<GridValuePtr<Layout, memory_space>> {

  grid_view<Layout, memory_space> m_gv;

  GridValuePtr(const grid_view<Layout, memory_space>& gv)
    : m_gv(gv) {}

  std::shared_ptr<GridValueArray::value_type>
  ptr() const {
    return
      std::shared_ptr<GridValueArray::value_type>(
        this->shared_from_this(),
        reinterpret_cast<GridValueArray::value_type*>(m_gv.data()));
  }

  virtual ~GridValuePtr() {}
};

/** concrete sub-class of abstract GridValueArray */
template <Device D>
class /*HPG_EXPORT*/ GridValueViewArray final
  : public GridValueArray {
public:

  using kokkos_device = typename DeviceT<D>::kokkos_device;
  using memory_space = typename kokkos_device::memory_space;
  using grid_layout = GridLayout<kokkos_device>;

  using grid_t =
    typename grid_view<typename grid_layout::layout, memory_space>::HostMirror;

  grid_t grid;

  GridValueViewArray() {}

  GridValueViewArray(const grid_t& grid_)
    : grid(grid_) {}

  virtual ~GridValueViewArray() {}

  unsigned
  extent(unsigned dim) const override {
    return grid.extent(dim);
  }

  static_assert(
    GridValueArray::Axis::x == 0
    && GridValueArray::Axis::y == 1
    && GridValueArray::Axis::mrow == 2
    && GridValueArray::Axis::channel == 3);

  const value_type&
  operator()(unsigned x, unsigned y, unsigned mrow, unsigned channel)
    const override {

    return reinterpret_cast<const value_type&>(grid(x, y, mrow, channel));
  }

  value_type&
  operator()(unsigned x, unsigned y, unsigned mrow, unsigned channel) override {

    return reinterpret_cast<value_type&>(grid(x, y, mrow, channel));
  }

  template <Device H>
  void
  copy_to(value_type* dst, Layout lyo) const {

    // we're assuming that a K::LayoutLeft or K::LayoutRight copy has no padding
    // (otherwise, the following is broken, not least because it may result in
    // an out-of-bounds access on dst)

    auto espace = typename DeviceT<H>::kokkos_device::execution_space();

    switch (lyo) {
    case Layout::Left: {
      K::View<
        typename grid_t::data_type,
        K::LayoutLeft,
        typename grid_t::memory_space,
        K::MemoryTraits<K::Unmanaged>> dstv(
          reinterpret_cast<typename grid_t::pointer_type>(dst),
          grid.extent(0), grid.extent(1), grid.extent(2), grid.extent(3));
      K::deep_copy(espace, dstv, grid);
      espace.fence();
      break;
    }
    case Layout::Right: {
      K::View<
        typename grid_t::data_type,
        K::LayoutRight,
        typename grid_t::memory_space,
        K::MemoryTraits<K::Unmanaged>> dstv(
          reinterpret_cast<typename grid_t::pointer_type>(dst),
          grid.extent(0), grid.extent(1), grid.extent(2), grid.extent(3));
      K::deep_copy(espace, dstv, grid);
      espace.fence();
      break;
    }
    default:
      assert(false);
      break;
    }
  }

protected:

  void
  unsafe_copy_to(Device host_device, value_type* dst, Layout lyo)
    const override {

    switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
    case Device::Serial:
      copy_to<Device::Serial>(dst, lyo);
      break;
#endif
#ifdef HPG_ENABLE_OPENMP
    case Device::OpenMP:
      copy_to<Device::OpenMP>(dst, lyo);
      break;
#endif
    default:
      assert(false);
      break;
    }
  }

public:

  template <Device H>
  static std::unique_ptr<GridValueViewArray>
  copy_from(
    const std::string& name,
    const value_type* src,
    const std::array<unsigned, rank>& extents,
    Layout lyo) {

    K::Array<int, rank>
      iext{int(extents[0]), int(extents[1]), int(extents[2]), int(extents[3])};
    grid_t grid(
      K::ViewAllocateWithoutInitializing(name),
      grid_layout::dimensions(iext));

    // we're assuming that a K::LayoutLeft or K::LayoutRight copy has no padding
    // (otherwise, the following is broken, not least because it may result in
    // an out-of-bounds access on dst)

    auto espace = typename DeviceT<H>::kokkos_device::execution_space();

    switch (lyo) {
    case Layout::Left: {
      K::View<
        typename grid_t::data_type,
        K::LayoutLeft,
        typename grid_t::memory_space,
        K::MemoryTraits<K::Unmanaged>> srcv(
          reinterpret_cast<typename grid_t::pointer_type>(
            const_cast<value_type *>(src)),
          grid.extent(0), grid.extent(1), grid.extent(2), grid.extent(3));
      K::deep_copy(espace, grid, srcv);
      espace.fence();
      break;
    }
    case Layout::Right: {
      K::View<
        typename grid_t::data_type,
        K::LayoutRight,
        typename grid_t::memory_space,
        K::MemoryTraits<K::Unmanaged>> srcv(
          reinterpret_cast<typename grid_t::pointer_type>(
            const_cast<value_type *>(src)),
          grid.extent(0), grid.extent(1), grid.extent(2), grid.extent(3));
      K::deep_copy(espace, grid, srcv);
      espace.fence();
      break;
    }
    default:
      assert(false);
      break;
    }
    return std::make_unique<GridValueViewArray>(grid);
  }

  static std::unique_ptr<GridValueViewArray>
  copy_from(
    const std::string& name,
    Device host_device,
    const value_type* src,
    const std::array<unsigned, rank>& extents,
    Layout lyo) {

    switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
    case Device::Serial:
      return copy_from<Device::Serial>(name, src, extents, lyo);
      break;
#endif
#ifdef HPG_ENABLE_OPENMP
    case Device::OpenMP:
      return copy_from<Device::OpenMP>(name, src, extents, lyo);
      break;
#endif
    default:
      assert(false);
      return nullptr;
      break;
    }
  }
};

/** concrete sub-class of abstract GridWeightArray */
template <Device D>
class /*HPG_EXPORT*/ GridWeightViewArray final
  : public GridWeightArray {
 public:

  using kokkos_device = typename DeviceT<D>::kokkos_device;
  using memory_space = typename kokkos_device::memory_space;
  using layout = typename kokkos_device::array_layout;
  using grid_weight_t =
    typename grid_weight_view<layout, memory_space>::HostMirror;

  grid_weight_t grid_weight;

  GridWeightViewArray(const grid_weight_t& grid_weight_)
    : grid_weight(grid_weight_) {}

  GridWeightViewArray() {}

  virtual ~GridWeightViewArray() {}

  unsigned
  extent(unsigned dim) const override {
    return grid_weight.extent(dim);
  }

  const value_type&
  operator()(unsigned mrow, unsigned channel) const override {

    return grid_weight(mrow, channel);
  }

  value_type&
  operator()(unsigned mrow, unsigned channel) override {

    return grid_weight(mrow, channel);
  }

  template <Device H>
  void
  copy_to(value_type* dst, Layout lyo) const {

    // we're assuming that a K::LayoutLeft or K::LayoutRight copy has no padding
    // (otherwise, the following is broken, not least because it may result in
    // an out-of-bounds access on dst)

    auto espace = typename DeviceT<H>::kokkos_device::execution_space();

    switch (lyo) {
    case Layout::Left: {
      K::View<
        typename grid_weight_t::data_type,
        K::LayoutLeft,
        typename grid_weight_t::memory_space,
        K::MemoryTraits<K::Unmanaged>> dstv(
          reinterpret_cast<typename grid_weight_t::pointer_type>(dst),
          grid_weight.extent(0), grid_weight.extent(1));
      K::deep_copy(espace, dstv, grid_weight);
      espace.fence();
      break;
    }
    case Layout::Right: {
      K::View<
        typename grid_weight_t::data_type,
        K::LayoutRight,
        typename grid_weight_t::memory_space,
        K::MemoryTraits<K::Unmanaged>> dstv(
          reinterpret_cast<typename grid_weight_t::pointer_type>(dst),
          grid_weight.extent(0), grid_weight.extent(1));
      K::deep_copy(espace, dstv, grid_weight);
      espace.fence();
      break;
    }
    default:
      assert(false);
      break;
    }
  }

protected:

  void
  unsafe_copy_to(Device host_device, value_type* dst, Layout lyo)
    const override {

    switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
    case Device::Serial:
      copy_to<Device::Serial>(dst, lyo);
      break;
#endif
#ifdef HPG_ENABLE_OPENMP
    case Device::OpenMP:
      copy_to<Device::OpenMP>(dst, lyo);
      break;
#endif
    default:
      assert(false);
      break;
    }
  }

public:

  template <Device H>
  static std::unique_ptr<GridWeightViewArray>
  copy_from(
    const std::string& name,
    const value_type* src,
    const std::array<unsigned, rank>& extents,
    Layout lyo) {

    grid_weight_t grid_weight(
      K::ViewAllocateWithoutInitializing(name),
      layout(extents[0], extents[1]));

    // we're assuming that a K::LayoutLeft or K::LayoutRight copy has no padding
    // (otherwise, the following is broken, not least because it may result in
    // an out-of-bounds access on dst)

    auto espace = typename DeviceT<H>::kokkos_device::execution_space();

    switch (lyo) {
    case Layout::Left: {
      K::View<
        typename grid_weight_t::data_type,
        K::LayoutLeft,
        typename grid_weight_t::memory_space,
        K::MemoryTraits<K::Unmanaged>> srcv(
          reinterpret_cast<typename grid_weight_t::pointer_type>(
            const_cast<value_type *>(src)),
          grid_weight.extent(0), grid_weight.extent(1));
      K::deep_copy(espace, grid_weight, srcv);
      espace.fence();
      break;
    }
    case Layout::Right: {
      K::View<
        typename grid_weight_t::data_type,
        K::LayoutRight,
        typename grid_weight_t::memory_space,
        K::MemoryTraits<K::Unmanaged>> srcv(
          reinterpret_cast<typename grid_weight_t::pointer_type>(
            const_cast<value_type *>(src)),
          grid_weight.extent(0), grid_weight.extent(1));
      K::deep_copy(espace, grid_weight, srcv);
      espace.fence();
      break;
    }
    default:
      assert(false);
      break;
    }
    return std::make_unique<GridWeightViewArray>(grid_weight);
  }

  static std::unique_ptr<GridWeightViewArray>
  copy_from(
    const std::string& name,
    Device host_device,
    const value_type* src,
    const std::array<unsigned, rank>& extents,
    Layout lyo) {

    switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
    case Device::Serial:
      return copy_from<Device::Serial>(name, src, extents, lyo);
      break;
#endif
#ifdef HPG_ENABLE_OPENMP
    case Device::OpenMP:
      return copy_from<Device::OpenMP>(name, src, extents, lyo);
      break;
#endif
    default:
      assert(false);
      return nullptr;
      break;
    }
  }
};

/** concrete sub-class of abstract GridValueArray for identically zero model
 * grids */
class /*HPG_EXPORT*/ UnallocatedModelValueArray final
  : public GridValueArray {
public:

  value_type m_zero;

  std::array<unsigned, rank> m_extents;

  UnallocatedModelValueArray(const std::array<unsigned, rank>& extents)
    : m_zero(0)
    , m_extents(extents) {}

  virtual ~UnallocatedModelValueArray() {}

  unsigned
  extent(unsigned dim) const override {
    return m_extents[dim];
  }

  const value_type&
  operator()(unsigned, unsigned, unsigned, unsigned)
    const override {

    return m_zero;
  }

  value_type&
  operator()(unsigned, unsigned, unsigned, unsigned) override {

    return m_zero;
  }

  template <Device H>
  void
  copy_to(value_type* dst, Layout lyo) const {

    // we're assuming that a K::LayoutLeft or K::LayoutRight copy has no padding
    // (otherwise, the following is broken, not least because it may result in
    // an out-of-bounds access on dst)

    auto espace = typename DeviceT<H>::kokkos_device::execution_space();

    switch (lyo) {
    case Layout::Left: {
      K::View<
        gv_t****,
        K::LayoutLeft,
        typename DeviceT<H>::kokkos_device::memory_space,
        K::MemoryTraits<K::Unmanaged>> dstv(
          reinterpret_cast<gv_t*>(dst),
          m_extents[0], m_extents[1], m_extents[2], m_extents[3]);
      K::deep_copy(espace, dstv, m_zero);
      espace.fence();
      break;
    }
    case Layout::Right: {
      K::View<
        gv_t****,
        K::LayoutRight,
        typename DeviceT<H>::kokkos_device::memory_space,
        K::MemoryTraits<K::Unmanaged>> dstv(
          reinterpret_cast<gv_t*>(dst),
          m_extents[0], m_extents[1], m_extents[2], m_extents[3]);
      K::deep_copy(espace, dstv, m_zero);
      espace.fence();
      break;
    }
    default:
      assert(false);
      break;
    }
  }

protected:

  void
  unsafe_copy_to(Device host_device, value_type* dst, Layout lyo)
    const override {

    switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
    case Device::Serial:
      copy_to<Device::Serial>(dst, lyo);
      break;
#endif
#ifdef HPG_ENABLE_OPENMP
    case Device::OpenMP:
      copy_to<Device::OpenMP>(dst, lyo);
      break;
#endif
    default:
      assert(false);
      break;
    }
  }
};

std::optional<std::tuple<unsigned, std::optional<Device>>>
parsed_cf_layout_version(const std::string& layout);

std::string
construct_cf_layout_version(unsigned vn, Device device);

/** initialize model visibilities view from GridValueArray instance */
template <Device D, typename GVH>
static void
init_model(GVH& gv_h, const GridValueArray& gv) {
  static_assert(
    K::SpaceAccessibility<
      typename DeviceT<D>::kokkos_device::memory_space,
      K::HostSpace>
    ::accessible);
  static_assert(
    GridValueArray::Axis::x == 0
    && GridValueArray::Axis::y == 1
    && GridValueArray::Axis::mrow == 2
    && GridValueArray::Axis::channel == 3);

  K::parallel_for(
    "init_model",
    K::MDRangePolicy<K::Rank<4>, typename DeviceT<D>::kokkos_device>(
      {0, 0, 0, 0},
      {int(gv.extent(0)),
       int(gv.extent(1)),
       int(gv.extent(2)),
       int(gv.extent(3))}),
    [&](int x, int y, int mr, int cb) {
      gv_h(x, y, mr, cb) = gv(x, y, mr, cb);
    });
}

rval_t<size_t>
min_cf_buffer_size(Device device, const CFArrayShape& cf, unsigned grp);

/** device-specific implementation sub-class of hpg::RWDeviceCFArray
 * class */
template <Device D>
class /*HPG_EXPORT*/ DeviceCFArray
  : public hpg::RWDeviceCFArray {
public:

  using kokkos_device = typename DeviceT<D>::kokkos_device;
  using cflayout = CFLayout<kokkos_device>;

  // notice layout for device D, but in HostSpace
  using cfd_view_h = cf_view<typename cflayout::layout, K::HostSpace>;

  /** layout version string */
  std::string m_version;
  /** oversampling factor */
  unsigned m_oversampling;
  /** extents by group */
  std::vector<std::array<unsigned, rank - 1>> m_extents;
  /** buffers in host memory with CF values */
  std::vector<std::vector<cf_t>> m_arrays;
  /** Views of host memory buffers */
  std::vector<cfd_view_h> m_views;

  DeviceCFArray()
    : m_version(construct_cf_layout_version(cf_layout_version_number, D)) {}

  DeviceCFArray(const CFArrayShape& shape)
    : m_version(construct_cf_layout_version(cf_layout_version_number, D))
    , m_oversampling(shape.oversampling()) {

    for (unsigned grp = 0; grp < shape.num_groups(); ++grp) {
      m_extents.push_back(shape.extents(grp));
      m_arrays.emplace_back(hpg::get_value(min_cf_buffer_size(D, shape, grp)));
      m_views.emplace_back(
        m_arrays.back().data(),
        cflayout::dimensions(*this, m_extents.size() - 1));
    }
  }

  DeviceCFArray(
    const std::string& version,
    unsigned oversampling,
    std::vector<
      std::tuple<std::array<unsigned, rank - 1>, std::vector<value_type>>>&&
      arrays)
    : m_version(version)
    , m_oversampling(oversampling) {

    for (auto& [e, v] : arrays) {
      m_extents.push_back(e);
      // we unfortunately must copy values from `arrays` because `cf_t` is
      // defined by the implementation, and is not the same type as `value_type`
      // (Kokkos::complex type has different alignment than std::complex)
      std::vector<cf_t> vals;
      vals.reserve(v.size());
      std::copy(v.begin(), v.end(), std::back_inserter(vals));
      m_arrays.push_back(std::move(vals));
      m_views.emplace_back(
        m_arrays.back().data(),
        cflayout::dimensions(*this, m_extents.size() - 1));
    }
  }

  virtual ~DeviceCFArray() {}

  unsigned
  oversampling() const override {
    return m_oversampling;
  }

  unsigned
  num_groups() const override {
    return m_arrays.size();
  }

  std::array<unsigned, rank - 1>
  extents(unsigned grp) const override {
    return m_extents[grp];
  }

  const char*
  layout() const override {
    return m_version.c_str();
  }

  static_assert(
    CFArray::Axis::x == 0
    && CFArray::Axis::y == 1
    && CFArray::Axis::mueller == 2
    && CFArray::Axis::channel == 3
    && CFArray::Axis::group == 4);

  std::complex<cf_fp>&
  operator()(
    unsigned x,
    unsigned y,
    unsigned mueller,
    unsigned channel,
    unsigned grp)
    override {

    return
      reinterpret_cast<std::complex<cf_fp>&>(
        m_views[grp](
          x / m_oversampling,
          y / m_oversampling,
          mueller,
          channel,
          x % m_oversampling,
          y % m_oversampling));
  }

  std::complex<cf_fp>
  operator()(
    unsigned x,
    unsigned y,
    unsigned mueller,
    unsigned channel,
    unsigned grp)
    const override {

    return
      m_views[grp](
        x / m_oversampling,
        y / m_oversampling,
        mueller,
        channel,
        x % m_oversampling,
        y % m_oversampling);
  }

  Device
  device() const override {
    return D;
  }
};

/** initialize CF array view from CFArray instance */
template <typename D, typename CFH>
static void
init_cf_host(CFH& cf_h, const CFArray& cf, unsigned grp) {
  static_assert(
    K::SpaceAccessibility<typename D::memory_space, K::HostSpace>
    ::accessible);

  static_assert(
    CFArray::Axis::x == 0
    && CFArray::Axis::y == 1
    && CFArray::Axis::mueller == 2
    && CFArray::Axis::channel == 3
    && CFArray::Axis::group == 4);

  auto extents = cf.extents(grp);
  auto oversampling = cf.oversampling();
  K::parallel_for(
    "cf_init",
    K::MDRangePolicy<K::Rank<4>, D>(
      {0, 0, 0, 0},
      {int(extents[0]), int(extents[1]), int(extents[2]), int(extents[3])}),
    [&](int i, int j, int mueller, int channel) {
      auto X = i / oversampling;
      auto x = i % oversampling;
      auto Y = j / oversampling;
      auto y = j % oversampling;
      cf_h(X, Y, mueller, channel, x, y) = cf(i, j, mueller, channel, grp);
    });
}

template <Device D>
static void
layout_for_device(
  Device host_device,
  const CFArray& cf,
  unsigned grp,
  CFArray::value_type* dst) {

  using kokkos_device = typename DeviceT<D>::kokkos_device;

  auto layout = CFLayout<kokkos_device>::dimensions(cf, grp);
  typename DeviceCFArray<D>::cfd_view_h
    cfd(reinterpret_cast<cf_t*>(dst), layout);
  switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial: {
    using host_device = DeviceT<Device::Serial>::kokkos_device;
    init_cf_host<host_device>(cfd, cf, grp);
    typename host_device::execution_space().fence();
    break;
  }
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
  case Device::OpenMP: {
    using host_device = DeviceT<Device::OpenMP>::kokkos_device;
    init_cf_host<host_device>(cfd, cf, grp);
    typename host_device::execution_space().fence();
    break;
  }
#endif // HPG_ENABLE_SERIAL
  default:
    assert(false);
    break;
  }
}

template <unsigned N>
size_t
max_number_cubes(const std::vector<::hpg::VisData<N>>& visdata) {
  auto max_gc =
    [](const size_t& acc, const auto& d) {
      return std::max(acc, d.m_grid_cubes.size());
    };
  return std::accumulate(visdata.begin(), visdata.end(), size_t(0), max_gc);
}

} // end namespace runtime::impl

} // end namespace hpg

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
