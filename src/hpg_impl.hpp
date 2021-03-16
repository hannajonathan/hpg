#pragma once

#include "hpg.hpp"

#include <algorithm>
#include <any>
#include <cassert>
#include <cfenv>
#include <cmath>
#include <deque>
#include <memory>
#include <set>
#include <type_traits>
#include <variant>

#ifndef NDEBUG
# include <iostream>
#endif

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

namespace K = Kokkos;
namespace KExp = Kokkos::Experimental;

// helper type for std::visit
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
// explicit deduction guide (not needed as of C++20)
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

/** @file hpg_impl.hpp
 *
 * HPG implementation header file
 */
namespace hpg {
namespace Impl {

template<int N>
struct poln_array_type {

  static_assert(std::is_same_v<visibility_fp, cf_fp>);

  K::complex<visibility_fp> vals[N];

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
};

template<int N>
struct vis_array_type {

  K::Array<K::complex<visibility_fp>, N> vis;
  K::Array<K::complex<cf_fp>, N> wgt;

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
}
}

namespace Kokkos { //reduction identity must be defined in Kokkos namespace
template<int N>
struct reduction_identity<hpg::Impl::poln_array_type<N>> {
  KOKKOS_FORCEINLINE_FUNCTION static hpg::Impl::poln_array_type<N> sum() {
    return hpg::Impl::poln_array_type<N>();
  }
};

template<int N>
struct reduction_identity<hpg::Impl::vis_array_type<N>> {
  KOKKOS_FORCEINLINE_FUNCTION static hpg::Impl::vis_array_type<N> sum() {
    return hpg::Impl::vis_array_type<N>();
  }
};
}

namespace hpg {

struct DisabledHostDeviceError
  : public Error {

  DisabledHostDeviceError()
    : Error(
      "Requested host device is not enabled",
      ErrorType::DisabledHostDevice) {}
};

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

struct InvalidModelGridSizeError
  : public Error {

  InvalidModelGridSizeError(
    const std::array<unsigned, GridValueArray::rank>& model_size,
    const std::array<unsigned, GridValueArray::rank>& grid_size)
    : Error(
      "model grid size " + sz2str(model_size)
      + " is different from visibility grid size " + sz2str(grid_size),
      ErrorType::InvalidModelGridSize) {}

  static std::string
  sz2str(const std::array<unsigned, GridValueArray::rank>& sz) {
    std::ostringstream oss;
    oss << "[" << sz[0]
        << "," << sz[1]
        << "," << sz[2]
        << "," << sz[3]
        << "]" << std::endl;
    return oss.str();
  }
};

namespace Impl {

/** visibility value type */
using vis_t = K::complex<visibility_fp>;

/** convolution function value type */
using cf_t = K::complex<cf_fp>;

/** gridded value type */
using gv_t = K::complex<grid_value_fp>;

/** portable UVW coordinates type */
using uvw_t = K::Array<vis_uvw_fp, 3>;

/** visibilities plus metadata for gridding */
template <unsigned N>
struct VisData {

  KOKKOS_INLINE_FUNCTION VisData() {};

  VisData(
    const K::Array<vis_t, N>& values, /**< visibility values */
    const K::Array<vis_weight_fp, N> weights, /**< visibility weights */
    vis_frequency_fp freq, /**< frequency */
    vis_phase_fp d_phase, /**< phase angle */
    const uvw_t& uvw, /** < uvw coordinates */
    unsigned& grid_cube, /**< grid cube index */
    const K::Array<unsigned, 2>& cf_index, /**< cf (cube, grp) index */
    const K::Array<cf_phase_gradient_fp, 2>& cf_phase_gradient/**< cf phase gradient */)
    : m_values(values)
    , m_weights(weights)
    , m_freq(freq)
    , m_d_phase(d_phase)
    , m_uvw(uvw)
    , m_grid_cube(grid_cube)
    , m_cf_index(cf_index)
    , m_cf_phase_gradient(cf_phase_gradient) {}

  VisData(VisData const&) = default;

  VisData(VisData&&) = default;

  KOKKOS_INLINE_FUNCTION ~VisData() = default;

  KOKKOS_INLINE_FUNCTION VisData& operator=(VisData const&) = default;

  KOKKOS_INLINE_FUNCTION VisData& operator=(VisData&&) = default;

  K::Array<vis_t, N> m_values;
  K::Array<vis_weight_fp, N> m_weights;
  vis_frequency_fp m_freq;
  vis_phase_fp m_d_phase;
  uvw_t m_uvw;
  unsigned m_grid_cube;
  K::Array<unsigned, 2> m_cf_index;
  K::Array<cf_phase_gradient_fp, 2> m_cf_phase_gradient;
};

static bool hpg_impl_initialized = false;

/** implementation initialization function */
bool
initialize() {
  bool result = true;
  K::initialize();
#ifdef HPG_ENABLE_OPENMP
  auto rc = fftw_init_threads();
  result = rc != 0;
#endif
#if defined(HPG_ENABLE_CUDA) \
  && (defined(HPG_ENABLE_OPENMP) || defined(HPG_ENABLE_SERIAL))
  if (std::fegetround() != FE_TONEAREST)
    std::cerr << "hpg::initialize() WARNING:"
              << " Host rounding mode not set to FE_TONEAREST " << std::endl
              << "  To avoid potential inconsistency in gridding on "
              << "  host vs gridding on device,"
              << "  set rounding mode to FE_TONEAREST" << std::endl;
#endif
  hpg_impl_initialized = result;
  return result;
}

/** implementation finalization function */
void
finalize() {
  K::finalize();
}

/** implementation initialization state */
bool
is_initialized() noexcept {
  return hpg_impl_initialized;
}

/** type trait associating Kokkos device with hpg Device */
template <Device D>
struct DeviceT {
  using kokkos_device = void;

  static constexpr unsigned active_task_limit = 0;

  using stream_type = void;

  static constexpr const char* const name = "";
};

#ifdef HPG_ENABLE_SERIAL
template <>
struct DeviceT<Device::Serial> {
  using kokkos_device = K::Serial;

  static constexpr unsigned active_task_limit = 1;

  using stream_type = void;

  static constexpr const char* const name = "Serial";
};
#endif // HPG_ENABLE_SERIAL

#ifdef HPG_ENABLE_OPENMP
template <>
struct DeviceT<Device::OpenMP> {
  using kokkos_device = K::OpenMP;

  static constexpr unsigned active_task_limit = 1;

  using stream_type = void;

  static constexpr const char* const name = "OpenMP";
};
#endif // HPG_ENABLE_OPENMP

#ifdef HPG_ENABLE_CUDA
template <>
struct DeviceT<Device::Cuda> {
  using kokkos_device = K::Cuda;

  // the maximum number of concurrent kernels for NVIDIA devices depends on
  // compute capability; set a large value here, much larger than any capability
  // through 8.6, and leave it to the user to limit the request
  static constexpr unsigned active_task_limit = 1024;

  using stream_type = cudaStream_t;

  static constexpr const char* const name = "Cuda";

  static bool
  create_stream(stream_type& stream) {
    auto rc = cudaStreamCreate(&stream);
    return rc == cudaSuccess;
  }

  static bool
  destroy_stream(stream_type& stream) {
    bool result = true;
    if (stream) {
      auto rc = cudaStreamDestroy(stream);
      result = rc == cudaSuccess;
      stream = NULL;
    }
    return result;
  }
};
#endif // HPG_ENABLE_CUDA

#ifdef HPG_ENABLE_HPX
template <>
struct DeviceT<Device::HPX> {
  // FIXME

  using kokkos_device = K::HPX;

  static unsigned constexpr active_task_limit = 1024;

  using stream_type = void;

  static constexpr const char* const name = "HPX";
};
#endif // HPG_ENABLE_HPX

/** View type for grid values */
template <typename Layout, typename memory_space>
using grid_view = K::View<gv_t****, Layout, memory_space>;

template <typename Layout, typename memory_space>
using const_grid_view = K::View<const gv_t****, Layout, memory_space>;

/** View type for weight values
 *
 * logical axis order: mrow, cube
 */
template <typename Layout, typename memory_space>
using weight_view = K::View<grid_value_fp**, Layout, memory_space>;

template <typename Layout, typename memory_space>
using const_weight_view = K::View<const grid_value_fp**, Layout, memory_space>;

/** View type for CF values */
template <typename Layout, typename memory_space>
using cf_view =
  K::View<cf_t******, Layout, memory_space, K::MemoryTraits<K::Unmanaged>>;

template <typename Layout, typename memory_space>
using const_cf_view =
  K::View<const cf_t******, Layout, memory_space, K::MemoryTraits<K::Unmanaged>>;

/** view type for unmanaged view of vector data on host */
template <typename T>
using vector_view = K::View<T*, K::HostSpace, K::MemoryTraits<K::Unmanaged>>;

/** type for (plain) vector data */
template <typename T>
using vector_data = std::shared_ptr<std::vector<T>>;

/** view for VisData<N> */
template <unsigned N, typename memory_space>
using visdata_view =
  K::View<VisData<N>*, memory_space, K::MemoryTraits<K::Unmanaged>>;

template <unsigned N, typename memory_space>
using const_visdata_view =
  K::View<const VisData<N>*, memory_space, K::MemoryTraits<K::Unmanaged>>;

/** view for backing buffer of visdata_views in ExecSpace */
template <typename memory_space>
using visbuff_view = K::View<VisData<4>*, memory_space>;

/** view for Mueller element index matrix */
template <typename memory_space>
using mindex_view = K::View<int[4][4], memory_space>;

template <typename memory_space>
using const_mindex_view =
  K::View<const int[4][4], memory_space, K::MemoryTraits<K::RandomAccess>>;

/** ordered Grid array axes */
enum class GridAxis {
  x,
  y,
  mrow,
  cube
};

/** axis order for strided grid layout */
static const std::array<int, 4> strided_grid_layout_order{
  static_cast<int>(GridAxis::y),
  static_cast<int>(GridAxis::mrow),
  static_cast<int>(GridAxis::x),
  static_cast<int>(GridAxis::cube)};

/** device-specific grid array layout */
template <Device D>
struct GridLayout {

  /** Kokkos layout type */
  using layout =
    std::conditional_t<
      std::is_same_v<
        typename DeviceT<D>::kokkos_device::array_layout,
        K::LayoutLeft>,
      K::LayoutLeft,
      K::LayoutStride>;

  /** create Kokkos layout using given grid dimensions
   *
   * logical index order matches GridAxis definition
   */
  static layout
  dimensions(const std::array<int, 4>& dims) {
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
static constexpr unsigned cf_layout_version_number = 0;

/** ordered CF array axes */
enum class CFAxis {
  x_major,
  y_major,
  mueller,
  cube,
  x_minor,
  y_minor
};

/** axis order for strided CF array layout */
static const std::array<int, 6> strided_cf_layout_order{
  static_cast<int>(CFAxis::mueller),
  static_cast<int>(CFAxis::y_major),
  static_cast<int>(CFAxis::x_major),
  static_cast<int>(CFAxis::cube),
  static_cast<int>(CFAxis::x_minor),
  static_cast<int>(CFAxis::y_minor)};

/** device-specific constant-support CF array layout */
template <Device D>
struct CFLayout {

  /** Kokkos layout type */
  using layout =
    std::conditional_t<
      std::is_same_v<
        typename DeviceT<D>::kokkos_device::array_layout,
        K::LayoutLeft>,
      K::LayoutLeft,
      K::LayoutStride>;

  /**
   * create Kokkos layout using given CFArray slice
   *
   * logical index order matches CFAxis definition
   *
   * @todo: verify these layouts
   */
  static layout
  dimensions(const CFArrayShape* cf, unsigned grp) {
    auto extents = cf->extents(grp);
    auto os = cf->oversampling();
    std::array<int, 6> dims{
      static_cast<int>((extents[CFArray::Axis::x] + os - 1) / os),
      static_cast<int>((extents[CFArray::Axis::y] + os - 1) / os),
      static_cast<int>(extents[CFArray::Axis::mueller]),
      static_cast<int>(extents[CFArray::Axis::cube]),
      static_cast<int>(os),
      static_cast<int>(os)
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
KOKKOS_FUNCTION std::tuple<int, int, int, int>
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
  // const double fine_origin = g_size / -(2.0 * (fine_scale / oversampling));
  // const long fine =
  //   std::lrint(
  //     std::floor((coord * inv_lambda - fine_origin) * fine_scale)) -
  //   (cf_size * oversampling) / 2;
  // const long major = fine / oversampling;
  // const long major_fine = major * oversampling;
  // int major = std::lrint(major_fine / oversampling) - cf_radius + g_offset;
  // int minor_shift =
  //   ((fine_coord >= nearest_major_fine_coord) ? 0 : oversampling);
  // int minor = fine_coord - (nearest_major_fine_coord - minor_shift);
  // const long minor = fine - major_fine;
  assert(0 <= cf_minor && cf_minor < oversampling);
  return {grid_coord, cf_major, cf_minor, fine_offset};
}

/** portable sincos()
 */
#pragma nv_exec_check_disable
template <typename execution_space, typename T>
KOKKOS_FORCEINLINE_FUNCTION void
sincos(T ph, T* sn, T* cs) {
  *sn = std::sin(ph);
  *cs = std::cos(ph);
}

#ifdef KOKKOS_ENABLE_CUDA
template <>
__device__ __forceinline__ void
sincos<K::Cuda, float>(float ph, float* sn, float* cs) {
  ::sincosf(ph, sn, cs);
}
template <>
__device__ __forceinline__ void
sincos<K::Cuda, double>(double ph, double* sn, double* cs) {
  ::sincos(ph, sn, cs);
}
#endif

/** convert phase to complex value
 */
template <typename execution_space, typename T>
KOKKOS_FORCEINLINE_FUNCTION K::complex<T>
cphase(T ph) {
  K::complex<T> result;
  sincos<execution_space, T>(ph, &result.imag(), &result.real());
  return result;
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION T
mag(const K::complex<T>& v) {
  return std::hypot(v.real(), v.imag());
}

/** helper class for computing visibility value and index metadata
 *
 * Basically exists to encapsulate conversion from a Visibility value to several
 * visibility metadata values needed by gridding kernel
 */
template <unsigned N, typename execution_space>
struct GridVis final {

  int m_grid_coord[2]; /**< grid coordinate */
  int m_cf_minor[2]; /**< CF minor coordinate */
  int m_cf_major[2]; /**< CF major coordinate */
  int m_fine_offset[2]; /**< visibility position - nearest major grid */
  K::Array<vis_t, N> m_values; /**< visibility values */
  K::Array<vis_weight_fp, N> m_weights; /**< visibility weights */
  K::complex<vis_phase_fp> m_phasor;
  int m_grid_cube; /**< grid cube index */
  bool m_pos_w; /**< true iff W coordinate is strictly positive */

  KOKKOS_INLINE_FUNCTION GridVis() {};

  KOKKOS_INLINE_FUNCTION GridVis(
    const VisData<N>& vis,
    const K::Array<int, 2>& grid_size,
    const K::Array<int, 2>& oversampling,
    const K::Array<int, 2>& cf_size,
    const K::Array<grid_scale_fp, 2>& grid_scale)
    : m_values(vis.m_values)
    , m_weights(vis.m_weights)
    , m_phasor(cphase<execution_space>(vis.m_d_phase))
    , m_grid_cube(vis.m_grid_cube) {

    static const vis_frequency_fp c = 299792458.0;
    auto inv_lambda = vis.m_freq / c;
    // can't use std::tie here - CUDA doesn't support it
    auto [g0, maj0, min0, f0] =
      compute_vis_coord(
        grid_size[0],
        oversampling[0],
        cf_size[0] / 2,
        vis.m_uvw[0],
        inv_lambda,
        grid_scale[0]);
    m_grid_coord[0] = g0;
    m_cf_major[0] = maj0;
    m_cf_minor[0] = min0;
    m_fine_offset[0] = f0;
    auto [g1, maj1, min1, f1] =
      compute_vis_coord(
        grid_size[1],
        oversampling[1],
        cf_size[1] / 2,
        vis.m_uvw[1],
        inv_lambda,
        grid_scale[1]);
    m_grid_coord[1] = g1;
    m_cf_major[1] = maj1;
    m_cf_minor[1] = min1;
    m_fine_offset[1] = f1;
    m_pos_w = vis.m_uvw[2] > 0;
  }

  GridVis(GridVis const&) = default;

  GridVis(GridVis&&) = default;

  KOKKOS_INLINE_FUNCTION GridVis& operator=(GridVis const&) = default;

  KOKKOS_INLINE_FUNCTION GridVis& operator=(GridVis&&) = default;
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
KOKKOS_FORCEINLINE_FUNCTION void
pseudo_atomic_add(volatile K::complex<T>& acc, const K::complex<T>& val) {
  K::atomic_add(&acc, val);
}

#ifdef HPG_ENABLE_CUDA
template <>
KOKKOS_FORCEINLINE_FUNCTION void
pseudo_atomic_add<K::Cuda, double>(
  volatile K::complex<double>& acc, const K::complex<double>& val) {

  K::atomic_add(&acc.real(), val.real());
  K::atomic_add(&acc.imag(), val.imag());
}

template <>
KOKKOS_FORCEINLINE_FUNCTION void
pseudo_atomic_add<K::Cuda, float>(
  volatile K::complex<float>& acc, const K::complex<float>& val) {

  K::atomic_add(&acc.real(), val.real());
  K::atomic_add(&acc.imag(), val.imag());
}
#endif // HPG_ENABLE_CUDA
#ifdef HPG_ENABLE_HPX
template <>
KOKKOS_FORCEINLINE_FUNCTION void
pseudo_atomic_add<K::HPX, double>(
  volatile K::complex<double>& acc, const K::complex<double>& val) {

  K::atomic_add(&acc.real(), val.real());
  K::atomic_add(&acc.imag(), val.imag());
}

template <>
KOKKOS_FORCEINLINE_FUNCTION void
pseudo_atomic_add<K::HPX, float>(
  volatile K::complex<float>& acc, const K::complex<float>& val) {

  K::atomic_add(&acc.real(), val.real());
  K::atomic_add(&acc.imag(), val.imag());
}
#endif // HPG_ENABLE_HPX

/** compute kernels
 *
 * A separate namespace, since these kernels may be accessed directly, without
 * the rest of libhpg.
 */
namespace Core {

// we're wrapping each kernel in a class in order to support partial
// specialization of the kernel functions by execution space

/** gridding kernel
 *
 * Note that the default implementation probably is optimal only for many-core
 * devices, probably not OpenMP (although it is correct on all devices).
 */
template <unsigned N, typename execution_space, unsigned version>
struct HPG_EXPORT VisibilityGridder final {

  using member_type = typename K::TeamPolicy<execution_space>::member_type;

  using scratch_phscr_view =
    K::View<
      cf_phase_gradient_fp*,
      typename execution_space::scratch_memory_space>;

  // function for gridding a single visibility
  template <typename cf_layout, typename grid_layout, typename memory_space>
  static KOKKOS_FUNCTION void
  grid_vis(
    const member_type& team_member,
    const GridVis<N, execution_space>& vis,
    const K::Array<int, 2>& oversampling,
    const cf_view<cf_layout, memory_space>& cf,
    const K::Array<int, 2>& cf_size,
    const unsigned cf_cube,
    const K::Array<cf_phase_gradient_fp, 2>& cf_gradient,
    const const_mindex_view<memory_space>& mueller_indexes,
    const const_mindex_view<memory_space>& conjugate_mueller_indexes,
    const const_grid_view<grid_layout, memory_space>& model,
    const grid_view<grid_layout, memory_space>& grid,
    const weight_view<typename execution_space::array_layout, memory_space>&
      weights,
    const scratch_phscr_view& phi_Y) {

    const auto& N_X = cf_size[0];
    const auto& N_Y = cf_size[1];
    const auto N_R = grid.extent_int(static_cast<int>(GridAxis::mrow));

    cf_fp cf_im_factor;
    const_mindex_view<memory_space> gridding_mindex;
    const_mindex_view<memory_space> degridding_mindex;
    if (vis.m_pos_w) {
      cf_im_factor = -1;
      gridding_mindex = mueller_indexes;
      degridding_mindex = conjugate_mueller_indexes;
    } else {
      cf_im_factor = 1;
      gridding_mindex = conjugate_mueller_indexes;
      degridding_mindex = mueller_indexes;
    }

    // phase screen constants at this visibility's location
    const auto phi_X0 =
      -cf_gradient[0]
      * ((cf_size[0] / 2) * oversampling[0] - vis.m_fine_offset[0]);
    const auto dphi_X = cf_gradient[0] * oversampling[0];
    const auto phi_Y0 =
      -cf_gradient[1]
      * ((cf_size[1] / 2) * oversampling[1] - vis.m_fine_offset[1]);
    const auto dphi_Y = cf_gradient[1] * oversampling[1];

    // compute the values of the phase screen along the Y axis now and store the
    // results in scratch memory because gridding on the Y axis accesses the
    // phase screen values for every row of the Mueller matrix column
    K::parallel_for(
      K::TeamVectorRange(team_member, N_Y),
      [=](const int Y) {
        phi_Y(Y) = phi_Y0 + Y * dphi_Y;
      });
    team_member.team_barrier();

    vis_array_type<N> vis_array;

    if (model.is_allocated()) {

      // model degridding

      // serial loop over grid mrow
      for (int R = 0; R < N_R; ++R) {
        vis_array_type<N> va;
        // parallel loop over grid X
        K::parallel_reduce(
          K::TeamVectorRange(team_member, N_X),
          [=](const int X, vis_array_type<N>& vis_array_l) {
            auto phi_X = phi_X0 + X * dphi_X;
            // loop over grid Y
            for (int Y = 0; Y < N_Y; ++Y) {
              auto screen = cphase<execution_space>(phi_X + phi_Y(Y));
              screen.imag() *= -1;
              auto mv =
                model(
                  X + vis.m_grid_coord[0],
                  Y + vis.m_grid_coord[1],
                  R,
                  vis.m_grid_cube)
                * screen;
              // loop over visibility polarizations
              for (int C = 0; C < N; ++C) {
                const auto mindex = degridding_mindex(R, C);
                if (mindex >= 0) {
                  cf_t cfv =
                    cf(
                      X + vis.m_cf_major[0],
                      Y + vis.m_cf_major[1],
                      mindex,
                      cf_cube,
                      vis.m_cf_minor[0],
                      vis.m_cf_minor[1]);
                  cfv.imag() *= -cf_im_factor;
                  vis_array_l.vis[C] += cfv * mv;
                  vis_array_l.wgt[C] += cfv;
                }
              }
            }
          },
          K::Sum<decltype(va)>(va));
        vis_array += va;
      }
    }
    // TODO: keep multiplication by conj_phasor explicit, in anticipation of
    // supporting read-back of residual visibilities
    auto conj_phasor = vis.m_phasor;
    conj_phasor.imag() *= -1;
    for (int C = 0; C < N; ++C)
      vis_array.vis[C] =
        (vis.m_values[C]
         - ((vis_array.vis[C]
             / ((vis_array.wgt[C] != (cf_t)0) ? vis_array.wgt[C] : (cf_t)1))
            * conj_phasor))
        * vis.m_phasor
        * vis.m_weights[C];

    // accumulate to grid, and CF weights per visibility polarization

    // serial loop over grid mrow
    for (int R = 0; R < N_R; ++R) {
      poln_array_type<N> vis_weights;
      // parallel loop over grid X
      K::parallel_reduce(
        K::TeamVectorRange(team_member, N_X),
        [=](const int X, poln_array_type<N>& vis_weights_l) {
          auto phi_X = phi_X0 + X * dphi_X;
          // loop over grid Y
          for (int Y = 0; Y < N_Y; ++Y) {
            auto screen = cphase<execution_space>(phi_X + phi_Y(Y));
            gv_t gv(0);
            // loop over visibility polarizations
            for (int C = 0; C < N; ++C) {
              const auto mindex = gridding_mindex(R, C);
              if (mindex >= 0) {
                cf_t cfv =
                  cf(
                    X + vis.m_cf_major[0],
                    Y + vis.m_cf_major[1],
                    mindex,
                    cf_cube,
                    vis.m_cf_minor[0],
                    vis.m_cf_minor[1]);
                cfv.imag() *= cf_im_factor;
                gv += gv_t(cfv * screen * vis_array.vis[C]);
                vis_weights_l.vals[C] += cfv;
              }
            }
            pseudo_atomic_add<execution_space>(
              grid(
                X + vis.m_grid_coord[0],
                Y + vis.m_grid_coord[1],
                R,
                vis.m_grid_cube),
              gv);
          }
        },
        K::Sum<decltype(vis_weights)>(vis_weights));
      // compute final weight and add it to weights
      K::single(
        K::PerTeam(team_member),
        [&]() {
          grid_value_fp twgt = 0;
          for (int C = 0; C < N; ++C)
            twgt += grid_value_fp(mag(vis_weights.vals[C]) * vis.m_weights[C]);
          K::atomic_add(&weights(R, vis.m_grid_cube), twgt);
        });
    }
  }

  template <
    typename cf_layout,
    typename grid_layout,
    typename memory_space>
  static void
  kernel(
    execution_space exec,
    const K::Array<
      cf_view<cf_layout, memory_space>,
      HPG_MAX_NUM_CF_GROUPS>& cfs,
    const K::Array<K::Array<int, 2>, HPG_MAX_NUM_CF_GROUPS>& cf_radii,
    unsigned max_cf_extent_y,
    const_mindex_view<memory_space> mueller_indexes,
    const_mindex_view<memory_space> conjugate_mueller_indexes,
    int num_visibilities,
    const const_visdata_view<N, memory_space>& visibilities,
    const K::Array<grid_scale_fp, 2>& grid_scale,
    const const_grid_view<grid_layout, memory_space>& model,
    const grid_view<grid_layout, memory_space>& grid,
    const weight_view<typename execution_space::array_layout, memory_space>&
      weights) {

    const K::Array<int, 2>
      grid_size{
        grid.extent_int(static_cast<int>(GridAxis::x)),
        grid.extent_int(static_cast<int>(GridAxis::y))};
    const K::Array<int, 2>
      oversampling{
        cfs[0].extent_int(static_cast<int>(CFAxis::x_minor)),
        cfs[0].extent_int(static_cast<int>(CFAxis::y_minor))};

    auto shmem_size = scratch_phscr_view::shmem_size(max_cf_extent_y);

    K::parallel_for(
      "gridding",
      K::TeamPolicy<execution_space>(exec, num_visibilities, K::AUTO)
      .set_scratch_size(0, K::PerTeam(shmem_size)),
      KOKKOS_LAMBDA(const member_type& team_member) {
        auto i = team_member.league_rank();

        const unsigned& cf_cube = visibilities(i).m_cf_index[0];
        const unsigned& cf_grp = visibilities(i).m_cf_index[1];
        const auto& cf = cfs[cf_grp];
        const K::Array<int, 2>
          cf_size{2 * cf_radii[cf_grp][0] + 1, 2 * cf_radii[cf_grp][1] + 1};
        scratch_phscr_view phi_Y(team_member.team_scratch(0), max_cf_extent_y);
        const auto& cf_gradient = visibilities(i).m_cf_phase_gradient;

        GridVis<N, execution_space>
          vis(visibilities(i), grid_size, oversampling, cf_size, grid_scale);
        // skip this visibility if all of the updated grid points are not
        // within grid bounds
        if ((0 <= vis.m_grid_coord[0])
            && (vis.m_grid_coord[0] + cf_size[0]
                <= grid.extent_int(static_cast<int>(GridAxis::x)))
            && (0 <= vis.m_grid_coord[1])
            && (vis.m_grid_coord[1] + cf_size[1]
                <= grid.extent_int(static_cast<int>(GridAxis::y))))
          grid_vis(
            team_member,
            vis,
            oversampling,
            cf,
            cf_size,
            cf_cube,
            cf_gradient,
            mueller_indexes,
            conjugate_mueller_indexes,
            model,
            grid,
            weights,
            phi_Y);
      });
  }
};

/** grid normalization kernel
 */
template <typename execution_space, unsigned version>
struct HPG_EXPORT GridNormalizer final {

  template <typename grid_layout, typename memory_space>
  static void
  kernel(
    execution_space exec,
    const grid_view<grid_layout, memory_space>& grid,
    const const_weight_view<
      typename execution_space::array_layout, memory_space>& weights,
    const grid_value_fp& wfactor) {

    static_assert(
      static_cast<int>(GridAxis::x) == 0
      && static_cast<int>(GridAxis::y) == 1
      && static_cast<int>(GridAxis::mrow) == 2
      && static_cast<int>(GridAxis::cube) == 3);
    static_assert(
      GridWeightArray::Axis::mrow == 0 && GridWeightArray::Axis::cube == 1);

    K::parallel_for(
      "normalization",
      K::MDRangePolicy<K::Rank<4>, execution_space>(
        exec,
        {0, 0, 0, 0},
        {grid.extent_int(static_cast<int>(GridAxis::x)),
         grid.extent_int(static_cast<int>(GridAxis::y)),
         grid.extent_int(static_cast<int>(GridAxis::mrow)),
         grid.extent_int(static_cast<int>(GridAxis::cube))}),
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
      static_cast<int>(GridAxis::x) == 0
      && static_cast<int>(GridAxis::y) == 1
      && static_cast<int>(GridAxis::mrow) == 2
      && static_cast<int>(GridAxis::cube) == 3);

    grid_value_fp inv_norm = (grid_value_fp)(1.0) / norm;
    K::parallel_for(
      "normalization",
      K::MDRangePolicy<K::Rank<4>, execution_space>(
        exec,
        {0, 0, 0, 0},
        {grid.extent_int(static_cast<int>(GridAxis::x)),
         grid.extent_int(static_cast<int>(GridAxis::y)),
         grid.extent_int(static_cast<int>(GridAxis::mrow)),
         grid.extent_int(static_cast<int>(GridAxis::cube))}),
      KOKKOS_LAMBDA(int x, int y, int mrow, int cube) {
        grid(x, y, mrow, cube) *= inv_norm;
      });
  }
};

/** fftw function class templated on fp precision */
template <typename T>
struct FFTW {

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

template <>
struct FFTW<double> {

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

template <>
struct FFTW<float> {

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
struct HPG_EXPORT FFT final {

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

    {
      [[maybe_unused]] size_t prev_stride = 0;
      for (size_t d = 0; d < 4; ++d) {
        assert(
          prev_stride <= igrid.layout().stride[strided_grid_layout_order[d]]);
        prev_stride = igrid.layout().stride[strided_grid_layout_order[d]];
      }
    }
    // this assumes there is no padding in grid
    assert(igrid.span() ==
           igrid.extent(0) * igrid.extent(1)
           * igrid.extent(2) * igrid.extent(3));
    static_assert(
      static_cast<int>(GridAxis::x) == 0
      && static_cast<int>(GridAxis::y) == 1
      && static_cast<int>(GridAxis::mrow) == 2
      && static_cast<int>(GridAxis::cube) == 3);
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

static Error
cufft_error(const std::string& prefix, cufftResult rc) {
  std::ostringstream oss(prefix);
  oss << ": cufftResult code " << rc;
  return Error(oss.str());
}

/** cufft function class templated on fp precision */
template <typename T>
struct CUFFT {
  //constexpr cufftType type;
  static cufftResult
  exec(cufftHandle, K::complex<T>*, K::complex<T>*, int) {
    assert(false);
    return CUFFT_NOT_SUPPORTED;
  }
};

template <>
struct CUFFT<double> {

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
struct CUFFT<float> {

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
struct HPG_EXPORT FFT<K::Cuda, 0> final {

  template <typename G>
  static std::pair<cufftResult_t, cufftHandle>
  grid_fft_handle(K::Cuda exec, G& grid) {

    using scalar_t = typename G::value_type::value_type;

    // this assumes there is no padding in grid
    assert(grid.span() ==
           grid.extent(0) * grid.extent(1) * grid.extent(2) * grid.extent(3));
    static_assert(
      static_cast<int>(GridAxis::x) == 0
      && static_cast<int>(GridAxis::y) == 1
      && static_cast<int>(GridAxis::mrow) == 2
      && static_cast<int>(GridAxis::cube) == 3);
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
KOKKOS_FORCEINLINE_FUNCTION void
swap_gv(gv_t& a, gv_t&b) {
  std::swap(a, b);
}

#ifdef HPG_ENABLE_CUDA
template <>
KOKKOS_FORCEINLINE_FUNCTION void
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
struct HPG_EXPORT GridShifter final {

  template <typename grid_layout, typename memory_space>
  static void
  kernel(
    execution_space exec,
    const grid_view<grid_layout, memory_space>& grid) {

    using scalar_t =
      typename grid_view<grid_layout, memory_space>::value_type::value_type;

    using member_type = typename K::TeamPolicy<execution_space>::member_type;

    // TODO: is this kernel valid for all possible GridAxis definitions?
    static_assert(
      static_cast<int>(GridAxis::x) == 0
      && static_cast<int>(GridAxis::y) == 1
      && static_cast<int>(GridAxis::mrow) == 2
      && static_cast<int>(GridAxis::cube) == 3);
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

      K::parallel_for(
        "grid_rotation_oo",
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
    } else {
      // two-pass algorithm for the general case

      K::parallel_for(
        "grid_rotation_gen",
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
                  swap_gv<execution_space>(gplane(x, y), gplane(x, y + mid_y));
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
                  swap_gv<execution_space>(gplane(x, y), gplane(x + mid_x, y));
              });
        });
    }
  }
};

} // end namespace Core

/** abstract base class for state implementations */
struct State {

  Device m_device; /**< device type */
  unsigned m_max_active_tasks; /**< maximum number of active tasks */
  size_t m_max_visibility_batch_size; /**< maximum number of visibilities to
                                         sent to gridding kernel at once */
  std::array<unsigned, 4> m_grid_size; /**< grid size */
  K::Array<grid_scale_fp, 2> m_grid_scale; /**< grid scale */
  unsigned m_num_polarizations; /**< number of visibility polarizations */
  std::array<unsigned, 4> m_implementation_versions; /**< impl versions*/

  State(Device device)
    : m_device(device) {}

  State(
    Device device,
    unsigned max_active_tasks,
    size_t max_visibility_batch_size,
    const std::array<unsigned, 4>& grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale,
    unsigned num_polarizations,
    const std::array<unsigned, 4>& implementation_versions)
    : m_device(device)
    , m_max_active_tasks(max_active_tasks)
    , m_max_visibility_batch_size(max_visibility_batch_size)
    , m_grid_size(grid_size)
    , m_grid_scale({grid_scale[0], grid_scale[1]})
    , m_num_polarizations(num_polarizations)
    , m_implementation_versions(implementation_versions) {}

  unsigned
  visibility_gridder_version() const {
    return m_implementation_versions[0];
  }

  unsigned
  grid_normalizer_version() const {
    return m_implementation_versions[1];
  }

  unsigned
  fft_version() const {
    return m_implementation_versions[2];
  }

  unsigned
  grid_shifter_version() const {
    return m_implementation_versions[3];
  }

  virtual size_t
  convolution_function_region_size(const CFArrayShape* shape)
    const noexcept = 0;

  virtual std::optional<Error>
  allocate_convolution_function_region(const CFArrayShape* shape) = 0;

  virtual std::optional<Error>
  set_convolution_function(Device host_device, CFArray&& cf) = 0;

  virtual std::optional<Error>
  set_model(Device host_device, GridValueArray&& gv) = 0;

  virtual std::optional<Error>
  grid_visibilities(Device host_device, VisDataVector&& visibilities) = 0;

  virtual void
  fence() const = 0;

  virtual std::unique_ptr<GridWeightArray>
  grid_weights() const = 0;

  virtual std::unique_ptr<GridValueArray>
  grid_values() const = 0;

  virtual std::unique_ptr<GridValueArray>
  model_values() const = 0;

  virtual void
  reset_grid() = 0;

  virtual void
  reset_model() = 0;

  virtual void
  normalize(grid_value_fp wfactor) = 0;

  virtual std::optional<Error>
  apply_grid_fft(grid_value_fp norm, FFTSign sign, bool in_place) = 0;

  virtual std::optional<Error>
  apply_model_fft(grid_value_fp norm, FFTSign sign, bool in_place) = 0;

  virtual void
  shift_grid() = 0;

  virtual void
  shift_model() = 0;

  virtual ~State() {}
};

/** concrete sub-class of abstract GridValueArray */
template <Device D>
class HPG_EXPORT GridValueViewArray final
  : public GridValueArray {
public:

  using memory_space = typename DeviceT<D>::kokkos_device::memory_space;
  using layout = typename GridLayout<D>::layout;
  using grid_t = typename grid_view<layout, memory_space>::HostMirror;

  grid_t grid;

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
    && GridValueArray::Axis::cube == 3);

  const value_type&
  operator()(unsigned x, unsigned y, unsigned mrow, unsigned cube)
    const override {

    static_assert(
      static_cast<int>(GridAxis::x) == 0
      && static_cast<int>(GridAxis::y) == 1
      && static_cast<int>(GridAxis::mrow) == 2
      && static_cast<int>(GridAxis::cube) == 3);
    return reinterpret_cast<const value_type&>(grid(x, y, mrow, cube));
  }

  value_type&
  operator()(unsigned x, unsigned y, unsigned mrow, unsigned cube) override {

    static_assert(
      static_cast<int>(GridAxis::x) == 0
      && static_cast<int>(GridAxis::y) == 1
      && static_cast<int>(GridAxis::mrow) == 2
      && static_cast<int>(GridAxis::cube) == 3);
    return reinterpret_cast<value_type&>(grid(x, y, mrow, cube));
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
    value_type* src,
    const std::array<unsigned, rank>& extents,
    Layout lyo) {

    std::array<int, rank> iext{
      static_cast<int>(extents[0]),
      static_cast<int>(extents[1]),
      static_cast<int>(extents[2]),
      static_cast<int>(extents[3])};
    grid_t grid(
      K::ViewAllocateWithoutInitializing(name),
      GridLayout<D>::dimensions(iext));

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
          reinterpret_cast<typename grid_t::pointer_type>(src),
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
          reinterpret_cast<typename grid_t::pointer_type>(src),
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
    value_type* src,
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
class HPG_EXPORT GridWeightViewArray final
  : public GridWeightArray {
 public:

  using memory_space = typename DeviceT<D>::kokkos_device::memory_space;
  using layout = typename DeviceT<D>::kokkos_device::array_layout;
  using weight_t = typename weight_view<layout, memory_space>::HostMirror;

  weight_t weight;

  GridWeightViewArray(const weight_t& weight_)
    : weight(weight_) {}

  virtual ~GridWeightViewArray() {}

  unsigned
  extent(unsigned dim) const override {
    return weight.extent(dim);
  }

  const value_type&
  operator()(unsigned mrow, unsigned cube) const override {

    return weight(mrow, cube);
  }

  value_type&
  operator()(unsigned mrow, unsigned cube) override {

    return weight(mrow, cube);
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
        typename weight_t::data_type,
        K::LayoutLeft,
        typename weight_t::memory_space,
        K::MemoryTraits<K::Unmanaged>> dstv(
          reinterpret_cast<typename weight_t::pointer_type>(dst),
          weight.extent(0), weight.extent(1));
      K::deep_copy(espace, dstv, weight);
      espace.fence();
      break;
    }
    case Layout::Right: {
      K::View<
        typename weight_t::data_type,
        K::LayoutRight,
        typename weight_t::memory_space,
        K::MemoryTraits<K::Unmanaged>> dstv(
          reinterpret_cast<typename weight_t::pointer_type>(dst),
          weight.extent(0), weight.extent(1));
      K::deep_copy(espace, dstv, weight);
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
    value_type* src,
    const std::array<unsigned, rank>& extents,
    Layout lyo) {

    weight_t weight(
      K::ViewAllocateWithoutInitializing(name),
      layout(extents[0], extents[1]));

    // we're assuming that a K::LayoutLeft or K::LayoutRight copy has no padding
    // (otherwise, the following is broken, not least because it may result in
    // an out-of-bounds access on dst)

    auto espace = typename DeviceT<H>::kokkos_device::execution_space();

    switch (lyo) {
    case Layout::Left: {
      K::View<
        typename weight_t::data_type,
        K::LayoutLeft,
        typename weight_t::memory_space,
        K::MemoryTraits<K::Unmanaged>> srcv(
          reinterpret_cast<typename weight_t::pointer_type>(src),
          weight.extent(0), weight.extent(1));
      K::deep_copy(espace, weight, srcv);
      espace.fence();
      break;
    }
    case Layout::Right: {
      K::View<
        typename weight_t::data_type,
        K::LayoutRight,
        typename weight_t::memory_space,
        K::MemoryTraits<K::Unmanaged>> srcv(
          reinterpret_cast<typename weight_t::pointer_type>(src),
          weight.extent(0), weight.extent(1));
      K::deep_copy(espace, weight, srcv);
      espace.fence();
      break;
    }
    default:
      assert(false);
      break;
    }
    return std::make_unique<GridWeightViewArray>(weight);
  }

  static std::unique_ptr<GridWeightViewArray>
  copy_from(
    const std::string& name,
    Device host_device,
    value_type* src,
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
class HPG_EXPORT UnallocatedModelValueArray final
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

/** initialize CF array view from CFArray instance */
template <Device D, typename CFH>
static void
init_cf_host(CFH& cf_h, const CFArray& cf, unsigned grp) {
  static_assert(
    K::SpaceAccessibility<
      typename DeviceT<D>::kokkos_device::memory_space,
      K::HostSpace>
    ::accessible);
  static_assert(
    static_cast<int>(CFAxis::x_major) == 0
    && static_cast<int>(CFAxis::y_major) == 1
    && static_cast<int>(CFAxis::mueller) == 2
    && static_cast<int>(CFAxis::cube) == 3
    && static_cast<int>(CFAxis::x_minor) == 4
    && static_cast<int>(CFAxis::y_minor) == 5);
  static_assert(
    CFArray::Axis::x == 0
    && CFArray::Axis::y == 1
    && CFArray::Axis::mueller == 2
    && CFArray::Axis::cube == 3
    && CFArray::Axis::group == 4);

  auto extents = cf.extents(grp);
  auto oversampling = cf.oversampling();
  K::parallel_for(
    "cf_init",
    K::MDRangePolicy<K::Rank<4>, typename DeviceT<D>::kokkos_device>(
      {0, 0, 0, 0},
      {static_cast<int>(extents[0]),
       static_cast<int>(extents[1]),
       static_cast<int>(extents[2]),
       static_cast<int>(extents[3])}),
    [&](int i, int j, int mueller, int cube) {
      auto X = i / oversampling;
      auto x = i % oversampling;
      auto Y = j / oversampling;
      auto y = j % oversampling;
      cf_h(X, Y, mueller, cube, x, y) = cf(i, j, mueller, cube, grp);
    });
}

static std::optional<std::tuple<unsigned, std::optional<Device>>>
parsed_cf_layout_version(const std::string& layout) {
  auto dash = layout.find('-');
  std::optional<int> vn;
  if (dash != std::string::npos) {
    try {
      vn = std::stoi(layout.substr(0, dash));
      if (vn.value() < 0)
        vn.reset();
    } catch (...) {}
  }
  if (vn) {
    std::string dev = layout.substr(dash + 1);
#ifdef HPG_ENABLE_SERIAL
    if (dev == DeviceT<Device::Serial>::name)
      return
        std::make_tuple(
          static_cast<unsigned>(vn.value()),
          std::optional<Device>(Device::Serial));
#endif
#ifdef HPG_ENABLE_OPENMP
    if (dev == DeviceT<Device::OpenMP>::name)
      return
        std::make_tuple(
          static_cast<unsigned>(vn.value()),
          std::optional<Device>(Device::OpenMP));
#endif
#ifdef HPG_ENABLE_CUDA
    if (dev == DeviceT<Device::Cuda>::name)
      return
        std::make_tuple(
          static_cast<unsigned>(vn.value()),
          std::optional<Device>(Device::Cuda));
#endif
#ifdef HPG_ENABLE_HPX
    if (dev == DeviceT<Device::HPX>::name)
      return
        std::make_tuple(
          static_cast<unsigned>(vn.value()),
          std::optional<Device>(Device::HPX));
#endif
    return std::make_tuple(static_cast<unsigned>(vn.value()), std::nullopt);
  }
  return std::nullopt;
}

static std::string
construct_cf_layout_version(unsigned vn, Device device) {
  std::ostringstream oss;
  oss << vn << "-";
  switch (device) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial:
    oss << DeviceT<Device::Serial>::name;
    break;
#endif
#ifdef HPG_ENABLE_OPENMP
  case Device::OpenMP:
    oss << DeviceT<Device::OpenMP>::name;
    break;
#endif
#ifdef HPG_ENABLE_CUDA
  case Device::Cuda:
    oss << DeviceT<Device::Cuda>::name;
    break;
#endif
#ifdef HPG_ENABLE_HPX
  case Device::HPX:
    oss << DeviceT<Device::HPX>::name;
    break;
#endif
  default:
    assert(false);
    break;
  }
  return oss.str();
}

/** device-specific implementation sub-class of hpg::DeviceCFArray class */
template <Device D>
class DeviceCFArray
  : public hpg::DeviceCFArray {
public:

  // notice layout for device D, but in HostSpace
  using cfd_view_h = cf_view<typename CFLayout<D>::layout, K::HostSpace>;

  /** layout version string */
  std::string m_version;
  /** oversampling factor */
  unsigned m_oversampling;
  /** extents by group */
  std::vector<std::array<unsigned, rank - 1>> m_extents;
  /** buffers in host memory with CF values */
  std::vector<std::vector<value_type>> m_arrays;
  /** Views of host memory buffers */
  std::vector<cfd_view_h> m_views;

  DeviceCFArray(
    const std::string& version,
    unsigned oversampling,
    std::vector<
      std::tuple<std::array<unsigned, rank - 1>, std::vector<value_type>>>&&
      arrays)
    : m_version(version)
    , m_oversampling(oversampling) {

    for (auto&& e_v : arrays) {
      m_extents.push_back(std::get<0>(e_v));
      m_arrays.push_back(std::get<1>(std::move(e_v)));
      m_views.emplace_back(
        reinterpret_cast<cf_t*>(m_arrays.back().data()),
        CFLayout<D>::dimensions(this, m_extents.size() - 1));
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
    && CFArray::Axis::cube == 3
    && CFArray::Axis::group == 4);

  std::complex<cf_fp>
  operator()(
    unsigned x,
    unsigned y,
    unsigned mueller,
    unsigned cube,
    unsigned grp)
    const override {

    static_assert(
      static_cast<int>(CFAxis::x_major) == 0
      && static_cast<int>(CFAxis::y_major) == 1
      && static_cast<int>(CFAxis::mueller) == 2
      && static_cast<int>(CFAxis::cube) == 3
      && static_cast<int>(CFAxis::x_minor) == 4
      && static_cast<int>(CFAxis::y_minor) == 5);
    return
      m_views[grp](
        x / m_oversampling,
        y / m_oversampling,
        mueller,
        cube,
        x % m_oversampling,
        y % m_oversampling);
  }

  Device
  device() const override {
    return D;
  }
};

template <Device D>
static void
layout_for_device(
  Device host_device,
  const CFArray& cf,
  unsigned grp,
  CFArray::value_type* dst) {

  auto layout = CFLayout<D>::dimensions(&cf, grp);
  typename DeviceCFArray<D>::cfd_view_h
    cfd(reinterpret_cast<cf_t*>(dst), layout);
  switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial:
    init_cf_host<Device::Serial>(cfd, cf, grp);
    typename DeviceT<Device::Serial>::kokkos_device::execution_space().fence();
    break;
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
  case Device::OpenMP:
    init_cf_host<Device::OpenMP>(cfd, cf, grp);
    typename DeviceT<Device::OpenMP>::kokkos_device::execution_space().fence();
    break;
#endif // HPG_ENABLE_SERIAL
  default:
    assert(false);
    break;
  }
}

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
    static_cast<int>(GridAxis::x) == 0
    && static_cast<int>(GridAxis::y) == 1
    && static_cast<int>(GridAxis::mrow) == 2
    && static_cast<int>(GridAxis::cube) == 3);
  static_assert(
    GridValueArray::Axis::x == 0
    && GridValueArray::Axis::y == 1
    && GridValueArray::Axis::mrow == 2
    && GridValueArray::Axis::cube == 3);

  K::parallel_for(
    "init_model",
    K::MDRangePolicy<K::Rank<4>, typename DeviceT<D>::kokkos_device>(
      {0, 0, 0, 0},
      {static_cast<int>(gv.extent(0)),
       static_cast<int>(gv.extent(1)),
       static_cast<int>(gv.extent(2)),
       static_cast<int>(gv.extent(3))}),
    [&](int x, int y, int mr, int cb) {
      gv_h(x, y, mr, cb) = gv(x, y, mr, cb);
    });
}

/** names for stream states */
enum class StreamPhase {
  COPY,
  COMPUTE
};

std::ostream&
operator<<(std::ostream& str, const StreamPhase& ph) {
  switch (ph) {
  case StreamPhase::COPY:
    str << "COPY";
    break;
  case StreamPhase::COMPUTE:
    str << "COMPUTE";
    break;
  }
  return str;
}

template <Device D>
struct StateT;

template <Device D>
struct CFPool final {

  using kokkos_device = typename DeviceT<D>::kokkos_device;
  using execution_space = typename kokkos_device::execution_space;
  using memory_space = typename execution_space::memory_space;
  using cfd_view = cf_view<typename CFLayout<D>::layout, memory_space>;
  using cfh_view = typename cfd_view::HostMirror;

  StateT<D> *state;
  K::View<cf_t*, memory_space> pool;
  unsigned num_cf_groups;
  unsigned max_cf_extent_y;
  K::Array<cfd_view, HPG_MAX_NUM_CF_GROUPS> cf_d; // unmanaged (in pool)
  std::vector<std::any> cf_h;
  K::Array<K::Array<int, 2>, HPG_MAX_NUM_CF_GROUPS> cf_radii;

  CFPool()
    : state(nullptr)
    , num_cf_groups(0)
    , max_cf_extent_y(0) {

    for (size_t i = 0; i < HPG_MAX_NUM_CF_GROUPS; ++i)
      cf_d[i] = cfd_view();
  }

  CFPool(StateT<D>* st)
    : state(st)
    , num_cf_groups(0)
    , max_cf_extent_y(0) {

    for (size_t i = 0; i < HPG_MAX_NUM_CF_GROUPS; ++i)
      cf_d[i] = cfd_view();
  }

  CFPool(const CFPool& other)
    : num_cf_groups(other.num_cf_groups)
    , max_cf_extent_y(other.max_cf_extent_y)
    , cf_radii(other.cf_radii) {

    if (other.pool.extent(0) > 0) {
      pool =
        decltype(pool)(
          K::ViewAllocateWithoutInitializing("cf"),
          other.pool.extent(0));
      K::deep_copy(
        other.state
        ->m_exec_spaces[other.state->next_exec_space(StreamPhase::COPY)].space,
        pool,
        other.pool);
      other.state->fence();
      for (size_t i = 0; i < num_cf_groups; ++i) {
        // don't need cf_h, since the previous fence ensures that the copy from
        // host has completed
        cf_d[i] =
          cfd_view(
            pool.data() + (other.cf_d[i].data() - other.pool.data()),
            other.cf_d[i].layout());
      }
    }
  }

  CFPool(CFPool&& other) {

    if (other.state)
      other.state->fence();
    std::swap(pool, other.pool);
    std::swap(num_cf_groups, other.num_cf_groups);
    std::swap(max_cf_extent_y, other.max_cf_extent_y);
    std::swap(cf_d, other.cf_d);
    std::swap(cf_h, other.cf_h);
    std::swap(cf_radii, other.cf_radii);
  }

  CFPool&
  operator=(const CFPool& rhs) {

    // TODO: make this exception-safe?
    reset();
    num_cf_groups = rhs.num_cf_groups;
    max_cf_extent_y = rhs.max_cf_extent_y;
    cf_radii = rhs.cf_radii;
    if (rhs.pool.extent(0) > 0) {
      pool =
        decltype(pool)(
          K::ViewAllocateWithoutInitializing("cf"),
          rhs.pool.extent(0));
      // use the rhs execution space for the copy, because otherwise we'd need a
      // fence on that execution space after the copy, and this way we can
      // possibly avoid a fence before the copy
      K::deep_copy(
        rhs.state
        ->m_exec_spaces[rhs.state->next_exec_space(StreamPhase::COPY)].space,
        pool,
        rhs.pool);
      rhs.state->fence();
      for (size_t i = 0; i < num_cf_groups; ++i) {
        // don't need cf_h, since the previous fence ensures that the copy from
        // host has completed
        cf_d[i] =
          cfd_view(
            pool.data() + (rhs.cf_d[i].data() - rhs.pool.data()),
            rhs.cf_d[i].layout());
      }
    }
    return *this;
  }

  CFPool&
  operator=(CFPool&& rhs) {

    if (state)
      state->fence();
    if (rhs.state)
      rhs.state->fence();
    std::swap(pool, rhs.pool);
    std::swap(num_cf_groups, rhs.num_cf_groups);
    std::swap(max_cf_extent_y, rhs.max_cf_extent_y);
    std::swap(cf_d, rhs.cf_d);
    std::swap(cf_h, rhs.cf_h);
    std::swap(cf_radii, rhs.cf_radii);
    return *this;
  }

  virtual ~CFPool() {
    reset();
  }

  static size_t
  cf_size(const CFArrayShape* cf, unsigned grp) {
    auto layout = CFLayout<D>::dimensions(cf, grp);
    // TODO: it would be best to use the following to compute
    // allocation size, but it is not implemented in Kokkos
    // 'auto alloc_sz = cfd_view::required_allocation_size(layout)'
    auto alloc_sz =
      cf_view<typename DeviceT<D>::kokkos_device::array_layout, memory_space>
      ::required_allocation_size(
        layout.dimension[0],
        layout.dimension[1],
        layout.dimension[2],
        layout.dimension[3],
        layout.dimension[4],
        layout.dimension[5]);
    return ((alloc_sz + (sizeof(cf_t) - 1)) / sizeof(cf_t));
  }

  static size_t
  pool_size(const CFArrayShape* cf) {
    size_t result = 0;
    if (cf)
      for (unsigned grp = 0; grp < cf->num_groups(); ++grp)
        result += cf_size(cf, grp);
    return result;
  }

  void
  prepare_pool(const CFArrayShape* cf, bool force = false) {
    auto current_pool_size = pool.extent(0);
    auto min_pool = pool_size(cf);
    reset((min_pool > current_pool_size) || force);
    if ((min_pool > current_pool_size) || (force && min_pool > 0))
      pool = decltype(pool)(K::ViewAllocateWithoutInitializing("cf"), min_pool);
  }

  void
  add_cf_group(
    const std::array<unsigned, 2>& radii,
    cfd_view cfd,
    std::any cfh) {

    assert(num_cf_groups < HPG_MAX_NUM_CF_GROUPS);
    cf_d[num_cf_groups] = cfd;
    cf_h.push_back(cfh);
    cf_radii[num_cf_groups] =
      {static_cast<int>(radii[0]), static_cast<int>(radii[1])};
    ++num_cf_groups;
    max_cf_extent_y =
      std::max(
        max_cf_extent_y,
        static_cast<unsigned>(cfd.extent(static_cast<int>(CFAxis::y_major))));
  }

  void
  add_host_cfs(Device host_device, execution_space espace, CFArray&& cf_array) {
    prepare_pool(&cf_array);
    size_t offset = 0;
    for (unsigned grp = 0; grp < cf_array.num_groups(); ++grp) {
      cfd_view cf_init(
        pool.data() + offset,
        CFLayout<D>::dimensions(&cf_array, grp));
#ifndef NDEBUG
      std::cout << "alloc cf sz " << cf_init.extent(0)
                << " " << cf_init.extent(1)
                << " " << cf_init.extent(2)
                << " " << cf_init.extent(3)
                << " " << cf_init.extent(4)
                << " " << cf_init.extent(5)
                << std::endl;
      std::cout << "alloc cf str " << cf_init.stride(0)
                << " " << cf_init.stride(1)
                << " " << cf_init.stride(2)
                << " " << cf_init.stride(3)
                << " " << cf_init.stride(4)
                << " " << cf_init.stride(5)
                << std::endl;
#endif // NDEBUG

      typename decltype(cf_init)::HostMirror cf_h;
      switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
      case Device::Serial:
        cf_h = K::create_mirror_view(cf_init);
        init_cf_host<Device::Serial>(cf_h, cf_array, grp);
        K::deep_copy(espace, cf_init, cf_h);
        break;
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
      case Device::OpenMP:
        cf_h = K::create_mirror_view(cf_init);
        init_cf_host<Device::OpenMP>(cf_h, cf_array, grp);
        K::deep_copy(espace, cf_init, cf_h);
        break;
#endif // HPG_ENABLE_SERIAL
      default:
        assert(false);
        break;
      }
      offset += cf_size(&cf_array, grp);
      add_cf_group(cf_array.radii(grp), cf_init, cf_h);
    }
  }

  void
  add_device_cfs(execution_space espace, DeviceCFArray<D>&& cf_array) {
    prepare_pool(&cf_array);
    size_t offset = 0;
    for (unsigned grp = 0; grp < cf_array.num_groups(); ++grp) {
      cfd_view cf_init(
        pool.data() + offset,
        CFLayout<D>::dimensions(&cf_array, grp));
#ifndef NDEBUG
      std::cout << "alloc cf sz " << cf_init.extent(0)
                << " " << cf_init.extent(1)
                << " " << cf_init.extent(2)
                << " " << cf_init.extent(3)
                << " " << cf_init.extent(4)
                << " " << cf_init.extent(5)
                << std::endl;
      std::cout << "alloc cf str " << cf_init.stride(0)
                << " " << cf_init.stride(1)
                << " " << cf_init.stride(2)
                << " " << cf_init.stride(3)
                << " " << cf_init.stride(4)
                << " " << cf_init.stride(5)
                << std::endl;
#endif // NDEBUG

      K::deep_copy(espace, cf_init, cf_array.m_views[grp]);
      offset += cf_size(&cf_array, grp);
      add_cf_group(
        cf_array.radii(grp),
        cf_init,
        std::make_tuple(
          std::move(cf_array.m_arrays[grp]),
          cf_array.m_views[grp]));
    }
  }

  void
  reset(bool free_pool = true) {
    if (state && pool.is_allocated()) {
      if (free_pool)
        pool = decltype(pool)();
      cf_h.clear();
      for (size_t i = 0; i < num_cf_groups; ++i)
        cf_d[i] = cfd_view();
      num_cf_groups = 0;
    }
  }
};

template <Device D>
struct ExecSpace final {
  using kokkos_device = typename DeviceT<D>::kokkos_device;
  using execution_space = typename kokkos_device::execution_space;
  using memory_space = typename execution_space::memory_space;

  execution_space space;
  visbuff_view<memory_space> visbuff;
  std::variant<
    visdata_view<1, memory_space>,
    visdata_view<2, memory_space>,
    visdata_view<3, memory_space>,
    visdata_view<4, memory_space>> visibilities;
  std::vector<std::any> copy_state;

  ExecSpace(execution_space sp)
    : space(sp) {
  }

  void
  fence() const {
    space.fence();
  }

  template <unsigned N>
  constexpr const_visdata_view<N, memory_space>
  visdata() const {
    return std::get<visdata_view<N, memory_space>>(visibilities);
  }

  /** copy visibilities to device */
  template <unsigned N>
  size_t
  copy_visibilities(
    size_t offset,
    size_t batch_size,
    const vector_data<::hpg::VisData<N>>& visdata) {

    size_t result = 0;
    if (visdata->size() > 0) {
      copy_state.emplace_back(visdata);
      result = std::min(visdata->size() - offset, batch_size);
      vector_view<VisData<N>> hview(
        reinterpret_cast<VisData<N>*>(visdata->data() + offset),
        result);
      if constexpr (!std::is_same_v<K::HostSpace, memory_space>) {
        visibilities =
          visdata_view<N, memory_space>(
            reinterpret_cast<VisData<N>*>(visbuff.data()),
            result);
        auto dview = std::get<visdata_view<N, memory_space>>(visibilities);
        auto dv = K::subview(dview, std::pair((size_t)0, result));
        K::deep_copy(space, dv, hview);
        copy_state.push_back(hview);
        copy_state.push_back(dv);
      } else {
        visibilities =
          visdata_view<N, memory_space>(
            reinterpret_cast<VisData<N>*>(&hview(0)),
            result);
        copy_state.push_back(hview);
      }
    }
    return result;
  }
};

/** Kokkos state implementation for a device type */
template <Device D>
struct HPG_EXPORT StateT final
  : public State {
public:

  using kokkos_device = typename DeviceT<D>::kokkos_device;
  using execution_space = typename kokkos_device::execution_space;
  using memory_space = typename execution_space::memory_space;
  using stream_type = typename DeviceT<D>::stream_type;

  grid_view<typename GridLayout<D>::layout, memory_space> m_grid;
  weight_view<typename execution_space::array_layout, memory_space> m_weights;
  grid_view<typename GridLayout<D>::layout, memory_space> m_model;
  const_mindex_view<memory_space> m_mueller_indexes;
  const_mindex_view<memory_space> m_conjugate_mueller_indexes;

  // use multiple execution spaces to support overlap of data copying with
  // computation when possible
  std::vector<std::conditional_t<std::is_void_v<stream_type>, int, stream_type>>
    m_streams;
  mutable std::vector<ExecSpace<D>> m_exec_spaces;
  mutable std::vector<std::tuple<CFPool<D>, std::optional<int>>> m_cfs;
  mutable std::deque<int> m_exec_space_indexes;
  mutable std::deque<int> m_cf_indexes;
  mutable StreamPhase m_current = StreamPhase::COPY;

  StateT(
    unsigned max_active_tasks,
    size_t max_visibility_batch_size,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4> grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale,
    const IArrayVector& mueller_indexes,
    const IArrayVector& conjugate_mueller_indexes,
    const std::array<unsigned, 4>& implementation_versions)
    : State(
      D,
      std::min(max_active_tasks, DeviceT<D>::active_task_limit),
      max_visibility_batch_size,
      grid_size,
      grid_scale,
      mueller_indexes.m_npol,
      implementation_versions) {

    init_state(init_cf_shape);
    m_mueller_indexes =
      init_mueller("mueller_indexes", mueller_indexes);
    m_conjugate_mueller_indexes =
      init_mueller("conjugte_mueller_indexes", conjugate_mueller_indexes);
    new_grid(true, true);
  }

  StateT(const StateT& st)
    : State(
      D,
      st.m_max_active_tasks,
      st.m_max_visibility_batch_size,
      st.m_grid_size,
      {st.m_grid_scale[0], st.m_grid_scale[1]},
      st.m_num_polarizations,
      st.m_implementation_versions) {

    st.fence();
    init_state(&st);
    m_mueller_indexes = st.m_mueller_indexes;
    m_conjugate_mueller_indexes = st.m_conjugate_mueller_indexes;
    new_grid(&st, true);
  }

  StateT(StateT&& st)
    : State(D) {

    m_max_active_tasks = std::move(st).m_max_active_tasks;
    m_max_visibility_batch_size = std::move(st).m_max_visibility_batch_size;
    m_grid_size = std::move(st).m_grid_size;
    m_grid_scale = std::move(st).m_grid_scale;
    m_num_polarizations = std::move(st).m_num_polarizations;
    m_implementation_versions = std::move(st).m_implementation_versions;

    m_grid = std::move(st).m_grid;
    m_weights = std::move(st).m_weights;
    m_model = std::move(st).m_model;
    m_mueller_indexes = std::move(st).m_mueller_indexes;
    m_conjugate_mueller_indexes = std::move(st).m_conjugate_mueller_indexes;
    m_streams = std::move(st).m_streams;
    m_exec_spaces = std::move(st).m_exec_spaces;
    m_exec_space_indexes = std::move(st).m_exec_space_indexes;
    m_current = std::move(st).m_current;

    m_cf_indexes = std::move(st).m_cf_indexes;
    m_cfs.resize(m_max_active_tasks);
    for (auto& [cf, last] : m_cfs)
      cf.state = this;
    for (size_t i = 0; i < m_max_active_tasks; ++i) {
      auto tmp_cf = std::move(m_cfs[i]);
      std::get<0>(tmp_cf).state = this;
      m_cfs[i] = std::move(st.m_cfs[i]);
      st.m_cfs[i] = std::move(tmp_cf);
    }
  }

  virtual ~StateT() {
    fence();
    m_grid = decltype(m_grid)();
    m_weights = decltype(m_weights)();
    m_model = decltype(m_model)();
    m_mueller_indexes = decltype(m_mueller_indexes)();
    m_conjugate_mueller_indexes = decltype(m_conjugate_mueller_indexes)();
    for (auto& [cf, last] : m_cfs)
      cf.reset();
    m_exec_spaces.clear();
    if constexpr(!std::is_void_v<stream_type>) {
      for (auto& str : m_streams) {
        auto rc = DeviceT<D>::destroy_stream(str);
        assert(rc);
      }
    }
  }

  StateT&
  operator=(const StateT& st) {
    StateT tmp(st);
    this->swap(tmp);
    return *this;
  }

  StateT&
  operator=(StateT&& st) {
    StateT tmp(std::move(st));
    this->swap(tmp);
    return *this;
  }

  StateT
  copy() const {
    return StateT(*this);
  }

  size_t
  convolution_function_region_size(const CFArrayShape* shape)
    const noexcept override {
    return
      shape
      ? std::get<0>(m_cfs[0]).pool_size(shape)
      : std::get<0>(m_cfs[m_cf_indexes.front()]).pool.extent(0);
  }

  std::optional<Error>
  allocate_convolution_function_region(const CFArrayShape* shape) override {
    fence();
    for (auto& [cf, last] : m_cfs) {
      last = std::nullopt;
      cf.prepare_pool(shape, true);
    }
    return std::nullopt;
  }

  std::optional<Error>
  set_convolution_function(Device host_device, CFArray&& cf_array) override {

    for (unsigned grp = 0; grp < cf_array.num_groups(); ++grp) {
      auto extents = cf_array.extents(grp);
      if ((extents[CFArray::Axis::x] >
           m_grid_size[static_cast<int>(GridAxis::x)]
           * cf_array.oversampling())
          || (extents[CFArray::Axis::y] >
              m_grid_size[static_cast<int>(GridAxis::y)]
              * cf_array.oversampling()))
        return Error("CF support size exceeds grid size");
    }

    switch_cf_pool();
    auto& exec = m_exec_spaces[next_exec_space(StreamPhase::COPY)];
    auto& cf = std::get<0>(m_cfs[m_cf_indexes.front()]);
    try {
      cf.add_device_cfs(
        exec.space,
        std::move(dynamic_cast<DeviceCFArray<D>&&>(cf_array)));
    } catch (const std::bad_cast&) {
      cf.add_host_cfs(host_device, exec.space, std::move(cf_array));
    }
    return std::nullopt;
  }

  std::optional<Error>
  set_model(Device host_device, GridValueArray&& gv) override {
    std::array<unsigned, 4>
      model_sz{gv.extent(0), gv.extent(1), gv.extent(2), gv.extent(3)};
    if (m_grid_size != model_sz)
      return InvalidModelGridSizeError(model_sz, m_grid_size);

    fence();
    auto& exec = m_exec_spaces[next_exec_space(StreamPhase::COPY)];

    if (!m_model.is_allocated()) {
      std::array<int, 4> ig{
        static_cast<int>(m_grid_size[0]),
        static_cast<int>(m_grid_size[1]),
        static_cast<int>(m_grid_size[2]),
        static_cast<int>(m_grid_size[3])};
      m_model =
        decltype(m_model)(
          K::ViewAllocateWithoutInitializing("model"),
          GridLayout<D>::dimensions(ig));
    }

    try {
      GridValueViewArray<D> gvv =
        std::move(dynamic_cast<GridValueViewArray<D>&&>(gv));
      K::deep_copy(exec.space, m_model, gvv.grid);
    } catch (const std::bad_cast&) {
      auto model_h = K::create_mirror_view(m_model);
      switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
      case Device::Serial:
        init_model<Device::Serial>(model_h, gv);
        break;
#endif
#ifdef HPG_ENABLE_OPENMP
      case Device::OpenMP:
        init_model<Device::OpenMP>(model_h, gv);
        break;
#endif
      default:
        return DisabledHostDeviceError();
        break;
      }
      K::deep_copy(exec.space, m_model, model_h);
    }
    return std::nullopt;
  }

  template <unsigned N>
  void
  default_grid_visibilities(
    Device /*host_device*/,
    size_t offset,
    const vector_data<::hpg::VisData<N>>& visibilities) {

    auto& exec_copy = m_exec_spaces[next_exec_space(StreamPhase::COPY)];
    auto len =
      exec_copy.copy_visibilities(
        offset,
        m_max_visibility_batch_size,
        visibilities);

    auto& exec_compute = m_exec_spaces[next_exec_space(StreamPhase::COMPUTE)];
    auto& cf = std::get<0>(m_cfs[m_cf_indexes.front()]);
    const_grid_view<typename GridLayout<D>::layout, memory_space> model
      = m_model;
    Core::VisibilityGridder<N, execution_space, 0>::kernel(
      exec_compute.space,
      cf.cf_d,
      cf.cf_radii,
      cf.max_cf_extent_y,
      m_mueller_indexes,
      m_conjugate_mueller_indexes,
      len,
      exec_compute.template visdata<N>(),
      m_grid_scale,
      model,
      m_grid,
      m_weights);
  }

  template <unsigned N>
  std::optional<Error>
    grid_visibilities(
    Device host_device,
    std::vector<::hpg::VisData<N>>&& visibilities) {

    const auto vis =
      std::make_shared<std::vector<::hpg::VisData<N>>>(
        std::move(visibilities));
// #ifndef NDEBUG
//     for (auto& [cube, supp] : *cf_indexes) {
//       auto& cfpool = std::get<0>(m_cfs[m_cf_indexes.front()]);
//       if ((supp >= cfpool.num_cf_groups)
//           || (cube >= cfpool.cf_d[supp].extent_int(5)))
//         return OutOfBoundsCFIndexError({cube, supp});
//     }
// #endif // NDEBUG

    size_t num_visibilities = vis->size();
    switch (visibility_gridder_version()) {
    case 0:
      for (size_t i = 0; i < num_visibilities; i += m_max_visibility_batch_size)
        default_grid_visibilities(host_device, i, vis);
      break;
    }
    return std::nullopt;
  }

  std::optional<Error>
  grid_visibilities(Device host_device, VisDataVector&& visibilities)
    override {

    switch (visibilities.m_npol) {
    case 1:
      return
        grid_visibilities(host_device, std::move(*visibilities.m_v1));
        break;
    case 2:
      return
        grid_visibilities(host_device, std::move(*visibilities.m_v2));
      break;
    case 3:
      return
        grid_visibilities(host_device, std::move(*visibilities.m_v3));
      break;
    case 4:
      return
        grid_visibilities(host_device, std::move(*visibilities.m_v4));
      break;
    default:
      assert(false);
      return Error("Assertion violation");
      break;
    }
  }

  void
  fence() const override {
    for (auto& i : m_exec_space_indexes) {
      auto& exec = m_exec_spaces[i];
      exec.fence();
    }
    m_current = StreamPhase::COPY;
  }

  std::unique_ptr<GridWeightArray>
  grid_weights() const override {
    fence();
    auto& exec = m_exec_spaces[next_exec_space(StreamPhase::COPY)];
    auto wgts_h = K::create_mirror(m_weights);
    K::deep_copy(exec.space, wgts_h, m_weights);
    exec.fence();
    return std::make_unique<GridWeightViewArray<D>>(wgts_h);
  }

  std::unique_ptr<GridValueArray>
  grid_values() const override {
    fence();
    auto& exec = m_exec_spaces[next_exec_space(StreamPhase::COPY)];
    auto grid_h = K::create_mirror(m_grid);
    K::deep_copy(exec.space, grid_h, m_grid);
    exec.fence();
    return std::make_unique<GridValueViewArray<D>>(grid_h);
  }

  std::unique_ptr<GridValueArray>
  model_values() const override {
    fence();
    auto& exec = m_exec_spaces[next_exec_space(StreamPhase::COPY)];
    if (m_model.is_allocated()) {
      auto model_h = K::create_mirror(m_model);
      K::deep_copy(exec.space, model_h, m_model);
      exec.fence();
      return std::make_unique<GridValueViewArray<D>>(model_h);
    } else {
      std::array<unsigned, 4> ex{
        static_cast<unsigned>(m_grid.extent(0)),
        static_cast<unsigned>(m_grid.extent(1)),
        static_cast<unsigned>(m_grid.extent(2)),
        static_cast<unsigned>(m_grid.extent(3))};
      return std::make_unique<UnallocatedModelValueArray>(ex);
    }
  }

  void
  reset_grid() override {
    fence();
    new_grid(true, true);
  }

  void
  reset_model() override {
    fence();
    m_model = decltype(m_model)();
  }

  void
  normalize(grid_value_fp wfactor) override {
    const_weight_view<typename execution_space::array_layout, memory_space>
      cweights = m_weights;
    switch (grid_normalizer_version()) {
    case 0:
      Core::GridNormalizer<execution_space, 0>::kernel(
        m_exec_spaces[next_exec_space(StreamPhase::COMPUTE)].space,
        m_grid,
        cweights,
        wfactor);
      break;
    default:
      assert(false);
      break;
    }
  }

  std::optional<Error>
  apply_grid_fft(grid_value_fp norm, FFTSign sign, bool in_place)
    override {

    std::optional<Error> err;
    if (in_place) {
      switch (fft_version()) {
      case 0:
        err =
          Core::FFT<execution_space, 0>
          ::in_place_kernel(
            m_exec_spaces[next_exec_space(StreamPhase::COMPUTE)].space,
            sign,
            m_grid);
        break;
      default:
        assert(false);
        break;
      }
    } else {
      const_grid_view<typename GridLayout<D>::layout, memory_space> pre_grid
        = m_grid;
      new_grid(false, false);
      switch (fft_version()) {
      case 0:
        err =
          Core::FFT<execution_space, 0>::out_of_place_kernel(
            m_exec_spaces[next_exec_space(StreamPhase::COMPUTE)].space,
            sign,
            pre_grid,
            m_grid);
        break;
      default:
        assert(false);
        break;
      }
    }
    // apply normalization
    if (norm != 1)
      switch (grid_normalizer_version()) {
      case 0:
        Core::GridNormalizer<execution_space, 0>::kernel(
          m_exec_spaces[next_exec_space(StreamPhase::COMPUTE)].space,
          m_grid,
          norm);
        break;
      default:
        assert(false);
        break;
      }
    return err;
  }

  std::optional<Error>
  apply_model_fft(grid_value_fp norm, FFTSign sign, bool in_place)
    override {

    std::optional<Error> err;
    if (m_model.is_allocated()){
      if (in_place) {
        switch (fft_version()) {
        case 0:
          err =
            Core::FFT<execution_space, 0>
            ::in_place_kernel(
              m_exec_spaces[next_exec_space(StreamPhase::COMPUTE)].space,
              sign,
              m_model);
          break;
        default:
          assert(false);
          break;
        }
      } else {
        const_grid_view<typename GridLayout<D>::layout, memory_space> pre_model
          = m_model;
        std::array<int, 4> ig{
          static_cast<int>(m_grid_size[0]),
          static_cast<int>(m_grid_size[1]),
          static_cast<int>(m_grid_size[2]),
          static_cast<int>(m_grid_size[3])};
        m_model =
          decltype(m_model)(
            K::ViewAllocateWithoutInitializing("grid"),
            GridLayout<D>::dimensions(ig));
        switch (fft_version()) {
        case 0:
          err =
            Core::FFT<execution_space, 0>::out_of_place_kernel(
              m_exec_spaces[next_exec_space(StreamPhase::COMPUTE)].space,
              sign,
              pre_model,
              m_model);
          break;
        default:
          assert(false);
          break;
        }
      }
      // apply normalization
      if (norm != 1)
        switch (grid_normalizer_version()) {
        case 0:
          Core::GridNormalizer<execution_space, 0>::kernel(
            m_exec_spaces[next_exec_space(StreamPhase::COMPUTE)].space,
            m_model,
            norm);
          break;
        default:
          assert(false);
          break;
        }
    }
    return err;
  }

  void
  shift_grid() override {
    switch (grid_shifter_version()) {
    case 0:
      Core::GridShifter<execution_space, 0>::kernel(
        m_exec_spaces[next_exec_space(StreamPhase::COMPUTE)].space,
        m_grid);
      break;
    default:
      assert(false);
      break;
    }
  }

  void
  shift_model() override {
    switch (grid_shifter_version()) {
    case 0:
      Core::GridShifter<execution_space, 0>::kernel(
        m_exec_spaces[next_exec_space(StreamPhase::COMPUTE)].space,
        m_model);
      break;
    default:
      assert(false);
      break;
    }
  }

private:

  void
  swap(StateT& other) {
    assert(m_max_active_tasks == other.m_max_active_tasks);
    std::swap(m_max_visibility_batch_size, other.m_max_visibility_batch_size);
    std::swap(m_grid_size, other.m_grid_size);
    std::swap(m_grid_scale, other.m_grid_scale);
    std::swap(m_implementation_versions, other.m_implementation_versions);

    std::swap(m_grid, other.m_grid);
    std::swap(m_weights, other.m_weights);
    std::swap(m_model, other.m_model);
    std::swap(m_mueller_indexes, other.m_mueller_indexes);
    std::swap(m_conjugate_mueller_indexes, other.m_conjugate_mueller_indexes);
    std::swap(m_streams, other.m_streams);
    std::swap(m_exec_spaces, other.m_exec_spaces);
    std::swap(m_exec_space_indexes, other.m_exec_space_indexes);
    std::swap(m_current, other.m_current);

    for (size_t i = 0; i < m_max_active_tasks; ++i) {
      auto tmp_cf = std::move(m_cfs[i]);
      std::get<0>(tmp_cf).state = this;
      m_cfs[i] = std::move(other.m_cfs[i]);
      other.m_cfs[i] = std::move(tmp_cf);
    }
    std::swap(m_cf_indexes, other.m_cf_indexes);
  }

  void
  init_state(const std::variant<const CFArrayShape*, const StateT*>& init) {
    m_streams.resize(m_max_active_tasks);
    m_exec_spaces.reserve(m_max_active_tasks);
    m_cfs.resize(m_max_active_tasks);
    for (unsigned i = 0; i < m_max_active_tasks; ++i) {
      if constexpr (!std::is_void_v<stream_type>) {
        auto rc = DeviceT<D>::create_stream(m_streams[i]);
        assert(rc);
        m_exec_spaces.emplace_back(execution_space(m_streams[i]));
        if (std::holds_alternative<const CFArrayShape*>(init)) {
          m_exec_space_indexes.push_back(i);
          m_cf_indexes.push_back(i);
        }
      } else {
        m_exec_spaces.emplace_back(execution_space());
        if (std::holds_alternative<const CFArrayShape*>(init)) {
          m_exec_space_indexes.push_back(i);
          m_cf_indexes.push_back(i);
        }
      }
      auto& esp = m_exec_spaces.back();
      if constexpr (!std::is_same_v<K::HostSpace, memory_space>)
        esp.visbuff =
          decltype(esp.visbuff)(
            K::ViewAllocateWithoutInitializing("visibility_buffer"),
            m_max_visibility_batch_size);
    }

    if (std::holds_alternative<const CFArrayShape*>(init)) {
      const CFArrayShape* init_cf_shape = std::get<const CFArrayShape*>(init);
      for (auto& [cf, last] : m_cfs) {
        cf.state = this;
        cf.prepare_pool(init_cf_shape, true);
        last.reset();
      }
    } else {
      const StateT* ost = std::get<const StateT*>(init);
      for (auto& i : ost->m_exec_space_indexes) {
        auto& esp = m_exec_spaces[i];
        m_exec_space_indexes.push_back(i);
        auto& st_esp = ost->m_exec_spaces[i];
        K::deep_copy(esp.space, esp.visbuff, st_esp.visbuff);
        if constexpr (std::is_same_v<K::HostSpace, memory_space>) {
          std::visit(
            overloaded {
              [&esp](const visdata_view<1, memory_space>& v) {
                esp.visibilities = v;
              },
              [&esp](const visdata_view<2, memory_space>& v) {
                esp.visibilities = v;
              },
              [&esp](const visdata_view<3, memory_space>& v) {
                esp.visibilities = v;
              },
              [&esp](const visdata_view<4, memory_space>& v) {
                esp.visibilities = v;
              }
            },
            st_esp.visibilities);
        } else {
          std::visit(
            overloaded {
              [&esp](const visdata_view<1, memory_space>& v) {
                esp.visibilities =
                  visdata_view<1, memory_space>(
                    reinterpret_cast<VisData<1>*>(esp.visbuff.data()),
                    v.extent(0));
              },
              [&esp](const visdata_view<2, memory_space>& v) {
                esp.visibilities =
                  visdata_view<2, memory_space>(
                    reinterpret_cast<VisData<2>*>(esp.visbuff.data()),
                    v.extent(0));
              },
              [&esp](const visdata_view<3, memory_space>& v) {
                esp.visibilities =
                  visdata_view<3, memory_space>(
                    reinterpret_cast<VisData<3>*>(esp.visbuff.data()),
                    v.extent(0));
              },
              [&esp](const visdata_view<4, memory_space>& v) {
                esp.visibilities =
                  visdata_view<4, memory_space>(
                    reinterpret_cast<VisData<4>*>(esp.visbuff.data()),
                    v.extent(0));
              }
            },
            st_esp.visibilities);
        }
        esp.copy_state = st_esp.copy_state;
      }
      for (auto& i : ost->m_cf_indexes) {
        auto& [cf, last] = m_cfs[i];
        m_cf_indexes.push_back(i);
        cf = std::get<0>(ost->m_cfs[i]);
        cf.state = this;
      }
      if (ost->m_model.is_allocated()) {
        std::array<int, 4> ig{
          static_cast<int>(m_grid_size[0]),
          static_cast<int>(m_grid_size[1]),
          static_cast<int>(m_grid_size[2]),
          static_cast<int>(m_grid_size[3])};
        m_model =
          decltype(m_model)(
            K::ViewAllocateWithoutInitializing("model"),
            GridLayout<D>::dimensions(ig));
        K::deep_copy(m_exec_spaces[0].space, m_model, ost->m_model);
      }
    }
    m_current = StreamPhase::COPY;
  }

  /** copy Mueller indexes to device */
  template <size_t N>
  mindex_view<memory_space>
  copy_mueller_indexes(
    const std::string& name,
    const std::vector<iarray<N>>& mindexes) {

    auto esp = m_exec_spaces[next_exec_space(StreamPhase::COMPUTE)];
    mindex_view<memory_space> result(name);
    auto mueller_indexes_h = K::create_mirror_view(result);
    size_t mr = 0;
    for (; mr < mindexes.size(); ++mr) {
      auto& mi_row = mindexes[mr];
      size_t mc = 0;
      for (; mc < N; ++mc)
        mueller_indexes_h(mr, mc) = static_cast<int>(mi_row[mc]);
      for (; mc < result.extent(1); ++mc)
        mueller_indexes_h(mr, mc) = -1;
    }
    for (; mr < result.extent(0); ++mr)
      for (size_t mc = 0; mc < result.extent(1); ++mc)
        mueller_indexes_h(mr, mc) = -1;
    K::deep_copy(esp.space, result, mueller_indexes_h);
    esp.fence();
    return result;
  }

  mindex_view<memory_space>
  init_mueller(const std::string& name, const IArrayVector& mueller_indexes) {

    switch (mueller_indexes.m_npol) {
    case 1:
      return copy_mueller_indexes(name, *mueller_indexes.m_v1);
      break;
    case 2:
      return copy_mueller_indexes(name, *mueller_indexes.m_v2);
      break;
    case 3:
      return copy_mueller_indexes(name, *mueller_indexes.m_v3);
      break;
    case 4:
      return copy_mueller_indexes(name, *mueller_indexes.m_v4);
      break;
    default:
      assert(false);
      return mindex_view<memory_space>(name);
      break;
    }
  }

protected:

  friend class CFPool<D>;

  int
  next_exec_space(StreamPhase next) const {
    int old_idx = m_exec_space_indexes.front();
    int new_idx = old_idx;
    if (m_current == StreamPhase::COMPUTE && next == StreamPhase::COPY) {
      if (m_max_active_tasks > 1) {
        m_exec_space_indexes.push_back(old_idx);
        m_exec_space_indexes.pop_front();
        new_idx = m_exec_space_indexes.front();
        // Although there is no need to fence on the new ExecSpace explicitly
        // for correctness, we use this opportunity to exert back-pressure on
        // the caller to limit the caller's rate of task submissions
        m_exec_spaces[new_idx].fence();
      }
      m_exec_spaces[new_idx].copy_state.clear();
      std::get<1>(m_cfs[m_cf_indexes.front()]) = new_idx;
    }
#ifndef NDEBUG
    std::cout << m_current << "(" << old_idx << ")->"
              << next << "(" << new_idx << ")"
              << std::endl;
#endif // NDEBUG
    m_current = next;
    return new_idx;
  }

  void
  switch_cf_pool() {
    auto esp_idx = next_exec_space(StreamPhase::COPY);
    m_cf_indexes.push_back(m_cf_indexes.front());
    m_cf_indexes.pop_front();
    auto& [cf, last] = m_cfs[m_cf_indexes.front()];
    if (last.value_or(esp_idx) != esp_idx)
      m_exec_spaces[last.value()].fence();
    last = esp_idx;
  }

private:

  void
  new_grid(std::variant<const StateT*, bool> source, bool also_weights) {

    const bool create_without_init =
      std::holds_alternative<const StateT*>(source) || !std::get<bool>(source);

    // in the following, we don't call next_exec_space() except when a stream is
    // required, as there are code paths that never use a stream, and thus we
    // can avoid unnecessary stream switches
    std::array<int, 4> ig{
      static_cast<int>(m_grid_size[0]),
      static_cast<int>(m_grid_size[1]),
      static_cast<int>(m_grid_size[2]),
      static_cast<int>(m_grid_size[3])};
    if (create_without_init)
      m_grid =
        decltype(m_grid)(
          K::ViewAllocateWithoutInitializing("grid"),
          GridLayout<D>::dimensions(ig));
    else
      m_grid =
        decltype(m_grid)(
          K::view_alloc(
            "grid",
            m_exec_spaces[next_exec_space(StreamPhase::COPY)].space),
          GridLayout<D>::dimensions(ig));
#ifndef NDEBUG
    std::cout << "alloc grid sz " << m_grid.extent(0)
              << " " << m_grid.extent(1)
              << " " << m_grid.extent(2)
              << " " << m_grid.extent(3)
              << std::endl;
    std::cout << "alloc grid str " << m_grid.stride(0)
              << " " << m_grid.stride(1)
              << " " << m_grid.stride(2)
              << " " << m_grid.stride(3)
              << std::endl;
#endif // NDEBUG

    static_assert(
      GridWeightArray::Axis::mrow == 0 && GridWeightArray::Axis::cube == 1);
    if (also_weights) {
      if (create_without_init)
        m_weights =
          decltype(m_weights)(
            K::ViewAllocateWithoutInitializing("weights"),
            static_cast<int>(m_grid_size[static_cast<int>(GridAxis::mrow)]),
            static_cast<int>(m_grid_size[static_cast<int>(GridAxis::cube)]));
      else
        m_weights =
          decltype(m_weights)(
            K::view_alloc(
              "weights",
              m_exec_spaces[next_exec_space(StreamPhase::COPY)].space),
            static_cast<int>(m_grid_size[static_cast<int>(GridAxis::mrow)]),
            static_cast<int>(m_grid_size[static_cast<int>(GridAxis::cube)]));
    }
    if (std::holds_alternative<const StateT*>(source)) {
      auto st = std::get<const StateT*>(source);
      auto& exec = m_exec_spaces[next_exec_space(StreamPhase::COPY)];
      K::deep_copy(exec.space, m_grid, st->m_grid);
      if (also_weights)
        K::deep_copy(exec.space, m_weights, st->m_weights);
    }
  }
};

} // end namespace Impl

std::unique_ptr<GridValueArray>
GridValueArray::copy_from(
  const std::string& name,
  Device target_device,
  Device host_device,
  value_type* src,
  const std::array<unsigned, GridValueArray::rank>& extents,
  Layout layout) {

  static_assert(
    static_cast<int>(Impl::GridAxis::x) == GridValueArray::Axis::x
    && static_cast<int>(Impl::GridAxis::y) == GridValueArray::Axis::y
    && static_cast<int>(Impl::GridAxis::mrow) == GridValueArray::Axis::mrow
    && static_cast<int>(Impl::GridAxis::cube) == GridValueArray::Axis::cube);

  switch (target_device) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial:
    return
      Impl::GridValueViewArray<Device::Serial>::copy_from(
        name,
        host_device,
        src,
        extents,
        layout);
    break;
#endif
#ifdef HPG_ENABLE_OPENMP
  case Device::OpenMP:
    return
      Impl::GridValueViewArray<Device::OpenMP>::copy_from(
        name,
        host_device,
        src,
        extents,
        layout);
    break;
#endif
#ifdef HPG_ENABLE_CUDA
  case Device::Cuda:
    return
      Impl::GridValueViewArray<Device::Cuda>::copy_from(
        name,
        host_device,
        src,
        extents,
        layout);
    break;
#endif
#ifdef HPG_ENABLE_HPX
  case Device::HPX:
    return
      Impl::GridValueViewArray<Device::HPX>::copy_from(
        name,
        host_device,
        src,
        extents,
        layout);
    break;
#endif
  default:
    assert(false);
    return nullptr;
    break;
  }
}

std::unique_ptr<GridWeightArray>
GridWeightArray::copy_from(
  const std::string& name,
  Device target_device,
  Device host_device,
  value_type* src,
  const std::array<unsigned, GridWeightArray::rank>& extents,
  Layout layout) {

  switch (target_device) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial:
    return
      Impl::GridWeightViewArray<Device::Serial>::copy_from(
        name,
        host_device,
        src,
        extents,
        layout);
    break;
#endif
#ifdef HPG_ENABLE_OPENMP
  case Device::OpenMP:
    return
      Impl::GridWeightViewArray<Device::OpenMP>::copy_from(
        name,
        host_device,
        src,
        extents,
        layout);
    break;
#endif
#ifdef HPG_ENABLE_CUDA
  case Device::Cuda:
    return
      Impl::GridWeightViewArray<Device::Cuda>::copy_from(
        name,
        host_device,
        src,
        extents,
        layout);
    break;
#endif
#ifdef HPG_ENABLE_HPX
  case Device::HPX:
    return
      Impl::GridWeightViewArray<Device::HPX>::copy_from(
        name,
        host_device,
        src,
        extents,
        layout);
    break;
#endif
  default:
    assert(false);
    return nullptr;
    break;
  }
}

} // end namespace hpg

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
