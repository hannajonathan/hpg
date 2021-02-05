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

#include <iostream> // FIXME: remove

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

/** @file hpg_impl.hpp
 *
 * HPG implementation header file
 */

namespace hpg {

struct OutOfBoundsCFIndexError
  : public Error {

  OutOfBoundsCFIndexError(const vis_cf_index_t& idx)
    : Error(
      "vis_cf_index_t value (" + std::to_string(std::get<0>(idx))
      + "," + std::to_string(std::get<1>(idx))
      + ") is out of bounds for current CFArray",
      ErrorType::OutOfBoundsCFIndex) {}
};

namespace Impl {

/** visibility value type */
using vis_t = K::complex<visibility_fp>;

/** convolution function value type */
using cf_t = K::complex<cf_fp>;

/** gridded value type */
using gv_t = K::complex<grid_value_fp>;

/** portable CF index type */
using cf_index_t = K::pair<unsigned, unsigned>;

/** portable UVW coordinates type */
using uvw_t = std::array<vis_uvw_fp, 3>;

/** portable CF phase screen type */
using cf_ps_t = std::array<cf_phase_screen_fp, 2>;

/** visibility data plus metadata for gridding */
struct Visibility {

  KOKKOS_INLINE_FUNCTION Visibility() {};

  Visibility(
    const vis_t& value_, /**< visibility value */
    unsigned grid_cube_, /**< grid cube index */
    vis_weight_fp weight_, /**< visibility weight */
    vis_frequency_fp freq_, /**< frequency */
    vis_phase_fp d_phase_, /**< phase angle */
    const uvw_t& uvw_ /** < uvw coordinates */)
    : value(value_)
    , grid_cube(grid_cube_)
    , weight(weight_)
    , freq(freq_)
    , d_phase(d_phase_)
    , uvw(uvw_) {
  }

  Visibility(Visibility const&) = default;

  Visibility(Visibility&&) = default;

  KOKKOS_INLINE_FUNCTION ~Visibility() = default;

  KOKKOS_INLINE_FUNCTION Visibility& operator=(Visibility const&) = default;

  KOKKOS_INLINE_FUNCTION Visibility& operator=(Visibility&&) = default;

  vis_t value;
  int grid_cube;
  vis_weight_fp weight;
  vis_frequency_fp freq;
  vis_phase_fp d_phase;
  uvw_t uvw;
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

/** View type for Visibility values */
template <typename memory_space>
using visibility_view = K::View<Visibility*, memory_space>;

template <typename memory_space>
using const_visibility_view = K::View<const Visibility*, memory_space>;

/** view type for unmanaged view of vector data on host */
template <typename T>
using vector_view = K::View<T*, K::HostSpace, K::MemoryTraits<K::Unmanaged>>;

/** type for (plain) vector data */
template <typename T>
using vector_data = std::shared_ptr<std::vector<T>>;

/** device-specific grid layout */
static const std::array<int, 4> strided_grid_layout_order{1, 2, 0, 3};

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
   * logical index order: X, Y, mrow, cube
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
   * logical index order: X, Y, mrow, x, y, cube
   */
  static layout
  dimensions(const CFArrayShape* cf, unsigned grp) {
    auto extents = cf->extents(grp);
    auto os = cf->oversampling();
    std::array<int, 6> dims{
      static_cast<int>((extents[0] + os - 1) / os),
      static_cast<int>((extents[1] + os - 1) / os),
      static_cast<int>(extents[2]),
      static_cast<int>(os),
      static_cast<int>(os),
      static_cast<int>(extents[3])
    };
    if constexpr (std::is_same_v<layout, K::LayoutLeft>) {
      return
        K::LayoutLeft(dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]);
    } else {
      static const std::array<int, 6> order{1, 2, 0, 4, 3, 5};
      return K::LayoutStride::order_dimensions(6, order.data(), dims.data());
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
 * - leftmost major grid coordinate of (visibility-centered) CF support
 * - leftmost major CF coordinate within CF support
 * - offset of visibility (on fine grid) from nearest-left major
 *   grid point (the "minor" coordinate of the CF, always non-negative)
 * - offset of visibility (on fine grid) from nearest major grid
 *   point (positive or negative)
 *
 * For negative grid_scale values, in the above description, change "left" to
 * "right"
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
  long grid_coord = std::lrint(position); // loc
  const long fine_offset = std::lrint((position - grid_coord) * oversampling); // off
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

/** helper class for computing visibility value and index metadata
 *
 * Basically exists to encapsulate conversion from a Visibility value to several
 * visibility metadata values needed by gridding kernel
 */
template <typename execution_space>
struct GridVis final {

  int grid_coord[2]; /**< grid coordinate */
  int cf_minor[2]; /**< CF minor coordinate */
  int cf_major[2]; /**< CF major coordinate */
  int fine_offset[2]; /**< visibility position - nearest major grid */
  vis_t value; /**< visibility value */
  int grid_cube; /**< grid cube index */
  vis_weight_fp weight; /**< visibility weight */
  cf_fp cf_im_factor; /**< weight conjugation factor */

  KOKKOS_INLINE_FUNCTION GridVis() {};

  KOKKOS_INLINE_FUNCTION GridVis(
    const Visibility& vis,
    const K::Array<int, 2>& grid_size,
    const K::Array<int, 2>& oversampling,
    const K::Array<int, 2>& cf_size,
    const K::Array<grid_scale_fp, 2>& grid_scale)
    : grid_cube(vis.grid_cube)
    , weight(vis.weight) {

    static const vis_frequency_fp c = 299792458.0;
    K::complex<vis_phase_fp> phasor = cphase<execution_space>(vis.d_phase);
    value = vis.value * phasor * vis.weight;
    auto inv_lambda = vis.freq / c;
    // can't use std::tie here - CUDA doesn't support it
    auto [g0, maj0, min0, f0] =
      compute_vis_coord(
        grid_size[0],
        oversampling[0],
        cf_size[0] / 2,
        vis.uvw[0],
        inv_lambda,
        grid_scale[0]);
    grid_coord[0] = g0;
    cf_major[0] = maj0;
    cf_minor[0] = min0;
    fine_offset[0] = f0;
    auto [g1, maj1, min1, f1] =
      compute_vis_coord(
        grid_size[1],
        oversampling[1],
        cf_size[1] / 2,
        vis.uvw[1],
        inv_lambda,
        grid_scale[1]);
    grid_coord[1] = g1;
    cf_major[1] = maj1;
    cf_minor[1] = min1;
    fine_offset[1] = f1;
    cf_im_factor = (vis.uvw[2] > 0) ? -1 : 1;
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

/** cf weight array type for reduction by gridding kernel */
struct HPG_EXPORT cf_wgt_array final {
  static constexpr int n_mrow = 4;

  cf_t wgts[n_mrow];

  KOKKOS_INLINE_FUNCTION cf_wgt_array() {
     init();
  }

  KOKKOS_INLINE_FUNCTION cf_wgt_array(const cf_wgt_array& rhs) {
    for (int i = 0; i < n_mrow; ++i)
      wgts[i] = rhs.wgts[i];
  }


  KOKKOS_INLINE_FUNCTION void
  init() {
    for (int i = 0; i < n_mrow; ++i)
      wgts[i] = 0;
  }

  KOKKOS_INLINE_FUNCTION cf_wgt_array&
  operator +=(const cf_wgt_array& src) {
    for (int i = 0; i < n_mrow; ++i)
      wgts[i] += src.wgts[i];
    return *this;
  }

  KOKKOS_INLINE_FUNCTION void
  operator +=(const volatile cf_wgt_array& src) volatile {
    for (int i = 0; i < n_mrow; ++i)
      wgts[i] += src.wgts[i];
  }
};

/** cf weight reducer class
 *
 * for use with Kokkos::parallel_reduce()
 */
template <typename space>
struct HPG_EXPORT SumCFWgts final {
public:

  typedef SumCFWgts reducer;
  typedef cf_wgt_array value_type;
  typedef
    Kokkos::View<value_type*, space, K::MemoryUnmanaged> result_view_type;

private:
  value_type & value;

public:

  KOKKOS_INLINE_FUNCTION
  SumCFWgts(value_type& value_): value(value_) {}

  KOKKOS_INLINE_FUNCTION void
  join(value_type& dest, const value_type& src)  const {
    dest += src;
  }

  KOKKOS_INLINE_FUNCTION void
  join(volatile value_type& dest, const volatile value_type& src) const {
    dest += src;
  }

  KOKKOS_INLINE_FUNCTION void
  init( value_type& val)  const {
    val.init();
  }

  KOKKOS_INLINE_FUNCTION value_type&
  reference() const {
    return value;
  }

  KOKKOS_INLINE_FUNCTION result_view_type
  view() const {
    return result_view_type(&value);
  }

  KOKKOS_INLINE_FUNCTION bool
  references_scalar() const {
    return true;
  }
};

// we're wrapping each kernel in a class in order to support partial
// specialization of the kernel functions by execution space

/** gridding kernel
 *
 * Note that the default implementation probably is optimal only for many-core
 * devices, probably not OpenMP (although it is correct on all devices).
 */
template <typename execution_space, unsigned version>
struct HPG_EXPORT VisibilityGridder final {

  using member_type = typename K::TeamPolicy<execution_space>::member_type;

  using scratch_wgts_view =
    K::View<cf_wgt_array*, typename execution_space::scratch_memory_space>;

  using scratch_phscr_view =
    K::View<
      cf_phase_screen_fp*,
      typename execution_space::scratch_memory_space>;

  // function for gridding a single visibility, without CF phase screen
  template <typename cf_layout, typename grid_layout, typename memory_space>
  static KOKKOS_FUNCTION void
  grid_vis(
    const member_type& team_member,
    const GridVis<execution_space>& vis,
    const K::Array<int, 2>& oversampling,
    const cf_view<cf_layout, memory_space>& cf,
    const K::Array<int, 2>& cf_size,
    const unsigned cf_cube,
    const grid_view<grid_layout, memory_space>& grid,
    const weight_view<typename execution_space::array_layout, memory_space>&
      weights) {

    const int N_X = cf_size[0];
    const int N_Y = cf_size[1];
    const int N_R = cf.extent_int(2);

    // accumulate weights in scratch memory for this visibility
    scratch_wgts_view cfw(team_member.team_scratch(0), 1);
    K::parallel_for(
      K::TeamVectorRange(team_member, N_R),
      [=](const int R) {
        cfw(0).wgts[R] = 0;
      });
    team_member.team_barrier();

    // do the multiplication of visibility by CF over all grid points in the
    // support of the CF centered on the visibility, and add products to grid
    K::parallel_reduce(
      K::TeamVectorRange(team_member, N_X),
      /* loop over majorX */
      [=](const int X, cf_wgt_array& cfw_l) {
        /* loop over elements (rows) of Mueller matrix column  */
        for (int R = 0; R < N_R; ++R) {
          /* loop over majorY */
          for (int Y = 0; Y < N_Y; ++Y) {
            cf_t cfv =
              cf(
                X + vis.cf_major[0],
                Y + vis.cf_major[1],
                R,
                vis.cf_minor[0],
                vis.cf_minor[1],
                cf_cube);
            cfv.imag() *= vis.cf_im_factor;
            pseudo_atomic_add<execution_space>(
              grid(
                X + vis.grid_coord[0],
                Y + vis.grid_coord[1],
                R,
                vis.grid_cube),
              gv_t(cfv * vis.value));
            cfw_l.wgts[R] += cfv;
          }
        }
      },
      SumCFWgts<execution_space>(cfw(0)));
    // by Kokkos reduction semantics the following barrier should not be
    // needed, but recent Slack discussion indicates a possible bug, so
    // we use it here until the issue is resolved
    team_member.team_barrier();
    // update weights array
    K::parallel_for(
      K::TeamVectorRange(team_member, N_R),
      [=](const int R) {
        K::atomic_add(
          &weights(R, vis.grid_cube),
          grid_value_fp(
            std::hypot(cfw(0).wgts[R].real(), cfw(0).wgts[R].imag())
            * vis.weight));
      });
  }

  // function for gridding a single visibility, with CF phase screen
  template <typename cf_layout, typename grid_layout, typename memory_space>
  static KOKKOS_FUNCTION void
  grid_vis(
    const member_type& team_member,
    const GridVis<execution_space>& vis,
    const K::Array<int, 2>& oversampling,
    const cf_view<cf_layout, memory_space>& cf,
    const K::Array<int, 2>& cf_size,
    const unsigned cf_cube,
    const K::Array<cf_phase_screen_fp, 2>& cf_gradient,
    const grid_view<grid_layout, memory_space>& grid,
    const weight_view<typename execution_space::array_layout, memory_space>&
      weights) {

    const int N_X = cf_size[0];
    const int N_Y = cf_size[1];
    const int N_R = cf.extent_int(2);

    // accumulate weights in scratch memory for this visibility
    scratch_wgts_view cfw(team_member.team_scratch(0), 1);
    K::parallel_for(
      K::TeamVectorRange(team_member, N_R),
      [=](const int R) {
        cfw(0).wgts[R] = 0;
      });

    // phase screen constants at this visibility's location
    const auto phi_X0 =
      -cf_gradient[0]
      * ((cf_size[0] * oversampling[0]) / 2 + vis.fine_offset[0]);
    const auto dphi_X = cf_gradient[0] * oversampling[0];
    const auto phi_Y0 =
      -cf_gradient[1]
      * ((cf_size[1] * oversampling[1]) / 2 + vis.fine_offset[1]);
    const auto dphi_Y = cf_gradient[1] * oversampling[1];

    // compute the values of the phase screen along the Y axis now and store the
    // results in scratch memory because gridding on the Y axis accesses the
    // phase screen values for every row of the Mueller matrix column
    scratch_phscr_view phi_Y(team_member.team_scratch(0), N_Y);
    K::parallel_for(
      K::TeamVectorRange(team_member, N_Y),
      [=](const int Y) {
        phi_Y(Y) = phi_Y0 + Y * dphi_Y;
      });
    team_member.team_barrier();

    // do the multiplication of visibility by corrected CF over all grid points
    // in the support of the CF centered on the visibility, and add products to
    // grid
    K::parallel_reduce(
      K::TeamVectorRange(team_member, N_X),
      /* loop over majorX */
      [=](const int X, cf_wgt_array& cfw_l) {
        auto phi_X = phi_X0 + X * dphi_X;
        /* loop over elements (rows) of Mueller matrix column  */
        for (int R = 0; R < N_R; ++R) {
          /* loop over majorY */
          for (int Y = 0; Y < N_Y; ++Y) {
            cf_t cfv =
              cf(
                X + vis.cf_major[0],
                Y + vis.cf_major[1],
                R,
                vis.cf_minor[0],
                vis.cf_minor[1],
                cf_cube);
            cfv.imag() *= vis.cf_im_factor;
            auto screen = cphase<execution_space>(phi_X + phi_Y(Y));
            pseudo_atomic_add<execution_space>(
              grid(
                X + vis.grid_coord[0],
                Y + vis.grid_coord[1],
                R,
                vis.grid_cube),
              gv_t(cfv * screen * vis.value));
            cfw_l.wgts[R] += cfv;
          }
        }
      },
      SumCFWgts<execution_space>(cfw(0)));
    // by Kokkos reduction semantics the following barrier should not be
    // needed, but recent Slack discussion indicates a possible bug, so
    // we use it here until the issue is resolved
    team_member.team_barrier();
    // update weights array
    K::parallel_for(
      K::TeamVectorRange(team_member, N_R),
      [=](const int R) {
        K::atomic_add(
          &weights(R, vis.grid_cube),
          grid_value_fp(
            std::hypot(cfw(0).wgts[R].real(), cfw(0).wgts[R].imag())
            * vis.weight));
      });
  }

  template <typename cf_layout, typename grid_layout, typename memory_space>
  static void
  kernel(
    execution_space exec,
    const K::Array<
      cf_view<cf_layout, memory_space>,
      HPG_MAX_NUM_CF_GROUPS>& cfs,
    const K::Array<K::Array<int, 2>, HPG_MAX_NUM_CF_GROUPS>& cf_radii,
    unsigned max_cf_extent_y,
    int num_visibilities,
    const K::View<const vis_t*, memory_space>& visibilities,
    const K::View<const unsigned*, memory_space>& grid_cubes,
    const K::View<const cf_index_t*, memory_space>& cf_indexes,
    const K::View<const vis_weight_fp*, memory_space>& vis_weights,
    const K::View<const vis_frequency_fp*, memory_space>& frequencies,
    const K::View<const vis_phase_fp*, memory_space>& phases,
    const K::View<const uvw_t*, memory_space>& coordinates,
    const K::View<const cf_ps_t*, memory_space>& cf_phase_screens,
    const K::Array<grid_scale_fp, 2>& grid_scale,
    const grid_view<grid_layout, memory_space>& grid,
    const weight_view<typename execution_space::array_layout, memory_space>&
      weights) {

    const K::Array<int, 2>
      grid_size{grid.extent_int(0), grid.extent_int(1)};
    const K::Array<int, 2>
      oversampling{cfs[0].extent_int(3), cfs[0].extent_int(4)};

    if (cf_phase_screens.extent(0) == 0) {
      // without CF phase screen
      auto shmem_size = scratch_wgts_view::shmem_size(1);

      K::parallel_for(
        "gridding",
        K::TeamPolicy<execution_space>(exec, num_visibilities, K::AUTO)
        .set_scratch_size(0, K::PerTeam(shmem_size)),
        KOKKOS_LAMBDA(const member_type& team_member) {
          auto i = team_member.league_rank();

          const unsigned& cf_cube = cf_indexes(i).first;
          const unsigned& cf_grp = cf_indexes(i).second;
          const auto& cf = cfs[cf_grp];
          K::Array<int, 2>
            cf_size{2 * cf_radii[cf_grp][0] + 1, 2 * cf_radii[cf_grp][1] + 1};

          GridVis<execution_space> vis(
            Visibility(
              visibilities(i),
              grid_cubes(i),
              vis_weights(i),
              frequencies(i),
              phases(i),
              coordinates(i)),
            grid_size,
            oversampling,
            cf_size,
            grid_scale);
          // skip this visibility if all of the updated grid points are not
          // within grid bounds
          if (0 <= vis.grid_coord[0]
              && vis.grid_coord[0] + cf_size[0] <= grid.extent_int(0)
              && 0 <= vis.grid_coord[1]
              && vis.grid_coord[1] + cf_size[1] <= grid.extent_int(1))
            grid_vis(
              team_member,
              vis,
              oversampling,
              cf,
              cf_size,
              cf_cube,
              grid,
              weights);
        });
    } else {
      // with CF phase screen
      auto shmem_size =
        scratch_wgts_view::shmem_size(1)
        + scratch_phscr_view::shmem_size(max_cf_extent_y);

      K::parallel_for(
        "gridding",
        K::TeamPolicy<execution_space>(exec, num_visibilities, K::AUTO)
        .set_scratch_size(0, K::PerTeam(shmem_size)),
        KOKKOS_LAMBDA(const member_type& team_member) {
          auto i = team_member.league_rank();

          const unsigned& cf_cube = cf_indexes(i).first;
          const unsigned& cf_grp = cf_indexes(i).second;
          const auto& cf = cfs[cf_grp];
          K::Array<int, 2>
            cf_size{2 * cf_radii[cf_grp][0] + 1, 2 * cf_radii[cf_grp][1] + 1};

          const K::Array<cf_phase_screen_fp, 2>
            cf_gradient{cf_phase_screens(i)[0], cf_phase_screens(i)[1]};

          GridVis<execution_space> vis(
            Visibility(
              visibilities(i),
              grid_cubes(i),
              vis_weights(i),
              frequencies(i),
              phases(i),
              coordinates(i)),
            grid_size,
            oversampling,
            cf_size,
            grid_scale);
          // skip this visibility if all of the updated grid points are not
          // within grid bounds
          if (0 <= vis.grid_coord[0]
              && vis.grid_coord[0] + cf_size[0] <= grid.extent_int(0)
              && 0 <= vis.grid_coord[1]
              && vis.grid_coord[1] + cf_size[1] <= grid.extent_int(1))
            grid_vis(
              team_member,
              vis,
              oversampling,
              cf,
              cf_size,
              cf_cube,
              cf_gradient,
              grid,
              weights);
        });
    }
  }
};

#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
template <typename execution_space>
struct HPG_EXPORT VisibilityGridder<execution_space, 1> final {

  using member_type = typename K::TeamPolicy<execution_space>::member_type;

  using scratch_wgts_view =
    K::View<cf_wgt_array*, typename execution_space::scratch_memory_space>;

  template <typename cf_layout, typename grid_layout, typename memory_space>
  static void
  kernel(
    execution_space exec,
    const K::Array<
      cf_view<cf_layout, memory_space>,
      HPG_MAX_NUM_CF_GROUPS>& cfs,
    const K::Array<K::Array<int, 2>, HPG_MAX_NUM_CF_GROUPS>& cf_radii,
    unsigned max_cf_extent_y,
    int num_visibilities,
    const K::View<const vis_t*, memory_space>& visibilities,
    const K::View<const unsigned*, memory_space>& grid_cubes,
    const K::View<const cf_index_t*, memory_space>& cf_indexes,
    const K::View<const vis_weight_fp*, memory_space>& vis_weights,
    const K::View<const vis_frequency_fp*, memory_space>& frequencies,
    const K::View<const vis_phase_fp*, memory_space>& phases,
    const K::View<const uvw_t*, memory_space>& coordinates,
    const K::View<const cf_ps_t*, memory_space>& cf_phase_screens,
    const K::Array<grid_scale_fp, 2>& grid_scale,
    const grid_view<grid_layout, memory_space>& grid,
    const weight_view<typename execution_space::array_layout, memory_space>&
    weights) {

    const K::Array<int, 2>
      grid_size{grid.extent_int(0), grid.extent_int(1)};
    const K::Array<int, 2>
      oversampling{cfs[0].extent_int(3), cfs[0].extent_int(4)};

    if (cf_phase_screens.extent(0) == 0) {
      // without CF phase screen
      K::parallel_for(
        "gridding",
        K::TeamPolicy<execution_space>(exec, num_visibilities, K::AUTO)
        .set_scratch_size(0, K::PerTeam(scratch_wgts_view::shmem_size(1))),
        KOKKOS_LAMBDA(const member_type& team_member) {
          auto i = team_member.league_rank();

          const unsigned& cf_cube = cf_indexes(i).first;
          const unsigned& cf_grp = cf_indexes(i).second;
          const auto& cf = cfs[cf_grp];
          K::Array<int, 2>
            cf_size{2 * cf_radii[cf_grp][0] + 1, 2 * cf_radii[cf_grp][1] + 1};

          GridVis<execution_space> vis(
            Visibility(
              visibilities(i),
              grid_cubes(i),
              vis_weights(i),
              frequencies(i),
              phases(i),
              coordinates(i)),
            grid_size,
            oversampling,
            cf_size,
            grid_scale);
          VisibilityGridder<execution_space, 0>::grid_vis(
            team_member,
            vis,
            oversampling,
            cf,
            cf_size,
            cf_cube,
            grid,
            weights);
        });
    } else {
      // with CF phase screen
      K::parallel_for(
        "gridding",
        K::TeamPolicy<execution_space>(exec, num_visibilities, K::AUTO)
        .set_scratch_size(0, K::PerTeam(scratch_wgts_view::shmem_size(1))),
        KOKKOS_LAMBDA(const member_type& team_member) {
          auto i = team_member.league_rank();

          const unsigned& cf_cube = cf_indexes(i).first;
          const unsigned& cf_grp = cf_indexes(i).second;
          const auto& cf = cfs[cf_grp];
          K::Array<int, 2>
            cf_size{2 * cf_radii[cf_grp][0] + 1, 2 * cf_radii[cf_grp][1] + 1};

          const K::Array<cf_phase_screen_fp, 2>
            cf_gradient{cf_phase_screens(i)[0], cf_phase_screens(i)[1]};

          GridVis<execution_space> vis(
            Visibility(
              visibilities(i),
              grid_cubes(i),
              vis_weights(i),
              frequencies(i),
              phases(i),
              coordinates(i)),
            grid_size,
            oversampling,
            cf_size,
            grid_scale);
          VisibilityGridder<execution_space, 0>::grid_vis(
            team_member,
            vis,
            oversampling,
            cf,
            cf_size,
            cf_cube,
            cf_gradient,
            grid,
            weights);
        });
    }
  }
};
#endif //HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS

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

    K::parallel_for(
      "normalization",
      K::MDRangePolicy<K::Rank<4>, execution_space>(
        exec,
        {0, 0, 0, 0},
        {grid.extent_int(0),
         grid.extent_int(1),
         grid.extent_int(2),
         grid.extent_int(3)}),
      KOKKOS_LAMBDA(int x, int y, int mrow, int cube) {
        grid(x, y, mrow, cube) /= (wfactor * weights(mrow, cube));
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
  std::array<unsigned, 4> m_implementation_versions; /**< impl versions*/

  State(Device device)
    : m_device(device) {}

  State(
    Device device,
    unsigned max_active_tasks,
    size_t max_visibility_batch_size,
    const std::array<unsigned, 4>& grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale,
    const std::array<unsigned, 4>& implementation_versions)
    : m_device(device)
    , m_max_active_tasks(max_active_tasks)
    , m_max_visibility_batch_size(max_visibility_batch_size)
    , m_grid_size(grid_size)
    , m_grid_scale({grid_scale[0], grid_scale[1]})
    , m_implementation_versions(implementation_versions) {}

  static size_t
  visibility_batch_allocation(size_t batch_size) {
    return
      batch_size
      * (sizeof(Kokkos::complex<visibility_fp>) // visibilities
         + sizeof(unsigned)                     // grid_cubes
         + sizeof(vis_cf_index_t)               // cf_indexes
         + sizeof(vis_weight_fp)                // weights
         + sizeof(vis_frequency_fp)             // frequencies
         + sizeof(vis_phase_fp)                 // phases
         + sizeof(uvw_t));                  // coordinates
  }

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
  grid_visibilities(
    Device host_device,
    std::vector<std::complex<visibility_fp>>&& visibilities,
    std::vector<unsigned>&& grid_cubes,
    std::vector<vis_cf_index_t>&& cf_indexes,
    std::vector<vis_weight_fp>&& weights,
    std::vector<vis_frequency_fp>&& frequencies,
    std::vector<vis_phase_fp>&& phases,
    std::vector<vis_uvw_t>&& coordinates,
    std::optional<std::vector<cf_phase_screen_t>>&& cf_phase_screens) = 0;

  virtual void
  fence() const = 0;

  virtual std::unique_ptr<GridWeightArray>
  grid_weights() const = 0;

  virtual std::unique_ptr<GridValueArray>
  grid_values() const = 0;

  virtual void
  reset_grid() = 0;

  virtual void
  normalize(grid_value_fp wfactor) = 0;

  virtual std::optional<Error>
  apply_fft(FFTSign sign, bool in_place) = 0;

  virtual void
  shift_grid() = 0;

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

  const value_type&
  operator()(unsigned x, unsigned y, unsigned mrow, unsigned cube)
    const override {

    return reinterpret_cast<const value_type&>(grid(x, y, mrow, cube));
  }

  value_type&
  operator()(unsigned x, unsigned y, unsigned mrow, unsigned cube) override {

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

    // std::array<int, rank> iext{
    //   static_cast<int>(extents[0]),
    //   static_cast<int>(extents[1]),
    //   static_cast<int>(extents[2]),
    //   static_cast<int>(extents[3])};
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

/** initialize CF array view from CFArray instance */
template <Device D, typename CFH>
static void
init_cf_host(CFH& cf_h, const CFArray& cf, unsigned grp) {
  static_assert(
    K::SpaceAccessibility<
      typename DeviceT<D>::kokkos_device::memory_space,
      K::HostSpace>
    ::accessible);

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
    [&](int i, int j, int mrow, int cube) {
      auto X = i / oversampling;
      auto x = i % oversampling;
      auto Y = j / oversampling;
      auto y = j % oversampling;
      cf_h(X, Y, mrow, x, y, cube) = cf(i, j, mrow, cube, grp);
    });
}

static std::optional<std::tuple<unsigned, std::optional<Device>>>
parsed_cf_layout_version(const std::string& layout) {
  auto dash = layout.find('-');
  std::optional<unsigned> vn;
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
      return std::make_tuple(vn.value(), std::optional<Device>(Device::Serial));
#endif
#ifdef HPG_ENABLE_OPENMP
    if (dev == DeviceT<Device::OpenMP>::name)
      return std::make_tuple(vn.value(), std::optional<Device>(Device::OpenMP));
#endif
#ifdef HPG_ENABLE_CUDA
    if (dev == DeviceT<Device::Cuda>::name)
      return std::make_tuple(vn.value(), std::optional<Device>(Device::Cuda));
#endif
#ifdef HPG_ENABLE_HPX
    if (dev == DeviceT<Device::HPX>::name)
      return std::make_tuple(vn.value(), std::optional<Device>(Device::HPX));
#endif
    return std::make_tuple(vn.value(), std::nullopt);
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
  std::vector<std::array<unsigned, 4>> m_extents;
  /** buffers in host memory with CF values */
  std::vector<std::vector<value_type>> m_arrays;
  /** Views of host memory buffers */
  std::vector<cfd_view_h> m_views;

  static std::vector<std::vector<value_type>>
  layout_for_device(Device host_device, const CFArray& cf) {

    std::vector<std::vector<value_type>> result;

    for (unsigned grp = 0; grp < cf.num_groups(); ++grp) {
      auto layout = CFLayout<D>::dimensions(&cf, grp);
      // TODO: it would be best to use the following to compute
      // allocation size, but it is not implemented in Kokkos
      // 'auto alloc_sz = cfd_view_h::required_allocation_size(layout)'
      auto alloc_sz =
        cf_view<typename DeviceT<D>::kokkos_device::array_layout, K::HostSpace>
        ::required_allocation_size(
          layout.dimension[0],
          layout.dimension[1],
          layout.dimension[2],
          layout.dimension[3],
          layout.dimension[4],
          layout.dimension[5]);
      result.emplace_back(((alloc_sz + (sizeof(cf_t) - 1)) / sizeof(cf_t)));
      cfd_view_h cfd(reinterpret_cast<cf_t*>(result.back().data()), layout);
      switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
      case Device::Serial:
        init_cf_host<Device::Serial>(cfd, cf, grp);
        break;
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
      case Device::OpenMP:
        init_cf_host<Device::OpenMP>(cfd, cf, grp);
        break;
#endif // HPG_ENABLE_SERIAL
      default:
        assert(false);
        break;
      }
    }
    return result;
  }

  DeviceCFArray(
    const std::string& version,
    unsigned oversampling,
    std::vector<std::tuple<std::array<unsigned, 4>, std::vector<value_type>>>&&
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

  std::array<unsigned, 4>
  extents(unsigned grp) const override {
    return m_extents[grp];
  }

  const char*
  layout() const override {
    return m_version.c_str();
  }

  std::complex<cf_fp>
  operator()(
    unsigned x,
    unsigned y,
    unsigned mrow,
    unsigned cube,
    unsigned grp) const override {
    return
      m_views[grp](
        x / m_oversampling,
        y / m_oversampling,
        mrow,
        x % m_oversampling,
        y % m_oversampling,
        cube);
  }

  Device
  device() const override {
    return D;
  }
};

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
      std::max(max_cf_extent_y, static_cast<unsigned>(cfd.extent(1)));
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
  K::View<vis_t*, memory_space> visibilities;
  K::View<unsigned*, memory_space> grid_cubes;
  K::View<cf_index_t*, memory_space> cf_indexes;
  K::View<vis_weight_fp*, memory_space> weights;
  K::View<vis_frequency_fp*, memory_space> frequencies;
  K::View<vis_phase_fp*, memory_space> phases;
  K::View<uvw_t*, memory_space> coordinates;
  K::View<cf_ps_t*, memory_space> cf_phase_screens;
  std::vector<std::any> vis_state;

  ExecSpace(execution_space sp)
    : space(sp) {
  }

  void
  fence() const {
    space.fence();
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
    const std::array<unsigned, 4>& implementation_versions)
    : State(
      D,
      std::min(max_active_tasks, DeviceT<D>::active_task_limit),
      max_visibility_batch_size,
      grid_size,
      grid_scale,
      implementation_versions) {

    init_state(init_cf_shape);
    new_grid(true, true);
  }

  StateT(const StateT& st)
    : State(
      D,
      st.m_max_active_tasks,
      st.m_max_visibility_batch_size,
      st.m_grid_size,
      {st.m_grid_scale[0], st.m_grid_scale[1]},
      st.m_implementation_versions) {

    st.fence();
    init_state(&st);
    new_grid(&st, true);
  }

  StateT(StateT&& st)
    : State(D) {

    m_max_active_tasks = std::move(st).m_max_active_tasks;
    m_max_visibility_batch_size = std::move(st).m_max_visibility_batch_size;
    m_grid_size = std::move(st).m_grid_size;
    m_grid_scale = std::move(st).m_grid_scale;
    m_implementation_versions = std::move(st).m_implementation_versions;

    m_grid = std::move(st).m_grid;
    m_weights = std::move(st).m_weights;
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
      if (extents[2] != m_grid_size[2])
        return
          Error("Unequal size of Mueller row dimension in grid and CF");
      if (extents[0] > m_grid_size[0] * cf_array.oversampling()
          || extents[1] > m_grid_size[1] * cf_array.oversampling())
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

  void
  default_grid_visibilities(
    Device /*host_device*/,
    size_t offset,
    const vector_data<std::complex<visibility_fp>>& visibilities,
    const vector_data<unsigned>& vis_grid_cubes,
    const vector_data<vis_cf_index_t>& vis_cf_indexes,
    const vector_data<vis_weight_fp>& vis_weights,
    const vector_data<vis_frequency_fp>& vis_frequencies,
    const vector_data<vis_phase_fp>& vis_phases,
    const vector_data<vis_uvw_t>& vis_coordinates,
    const vector_data<cf_phase_screen_t>& vis_cf_phase_screens) {

    auto& exec_copy = m_exec_spaces[next_exec_space(StreamPhase::COPY)];

    auto len =
      std::min(visibilities->size() - offset, m_max_visibility_batch_size);

    exec_copy.vis_state.emplace_back(visibilities);
    auto vis_views =
      StateT<D>::copy_to_device_view<vis_t>(
        exec_copy.visibilities,
        visibilities,
        offset,
        len,
        exec_copy.space);
    exec_copy.vis_state.push_back(vis_views);

    exec_copy.vis_state.emplace_back(vis_grid_cubes);
    auto grid_cubes_views =
      StateT<D>::copy_to_device_view<unsigned>(
        exec_copy.grid_cubes,
        vis_grid_cubes,
        offset,
        len,
        exec_copy.space);
    exec_copy.vis_state.push_back(grid_cubes_views);

    exec_copy.vis_state.emplace_back(vis_cf_indexes);
    auto cf_indexes_views =
      StateT<D>::copy_to_device_view<cf_index_t>(
        exec_copy.cf_indexes,
        vis_cf_indexes,
        offset,
        len,
        exec_copy.space);
    exec_copy.vis_state.push_back(cf_indexes_views);

    exec_copy.vis_state.emplace_back(vis_weights);
    auto weights_views =
      StateT<D>::copy_to_device_view<vis_weight_fp>(
        exec_copy.weights,
        vis_weights,
        offset,
        len,
        exec_copy.space);
    exec_copy.vis_state.push_back(weights_views);

    exec_copy.vis_state.emplace_back(vis_frequencies);
    auto frequencies_views =
      StateT<D>::copy_to_device_view<vis_frequency_fp>(
        exec_copy.frequencies,
        vis_frequencies,
        offset,
        len,
        exec_copy.space);
    exec_copy.vis_state.push_back(frequencies_views);

    exec_copy.vis_state.emplace_back(vis_phases);
    auto phases_views =
      StateT<D>::copy_to_device_view<vis_phase_fp>(
        exec_copy.phases,
        vis_phases,
        offset,
        len,
        exec_copy.space);
    exec_copy.vis_state.push_back(phases_views);

    exec_copy.vis_state.emplace_back(vis_coordinates);
    auto coordinates_views =
      StateT<D>::copy_to_device_view<vis_uvw_t>(
        exec_copy.coordinates,
        vis_coordinates,
        offset,
        len,
        exec_copy.space);
    exec_copy.vis_state.push_back(coordinates_views);

    exec_copy.vis_state.emplace_back(vis_cf_phase_screens);
    auto cf_phase_screens_views =
      StateT<D>::copy_to_device_view<cf_ps_t>(
        exec_copy.cf_phase_screens,
        vis_cf_phase_screens,
        offset,
        len,
        exec_copy.space);
    exec_copy.vis_state.push_back(cf_phase_screens_views);

    auto& exec_compute = m_exec_spaces[next_exec_space(StreamPhase::COMPUTE)];
    auto& cf = std::get<0>(m_cfs[m_cf_indexes.front()]);
    Core::VisibilityGridder<execution_space, 0>::kernel(
      exec_compute.space,
      cf.cf_d,
      cf.cf_radii,
      cf.max_cf_extent_y,
      len,
      std::get<1>(vis_views),
      std::get<1>(grid_cubes_views),
      std::get<1>(cf_indexes_views),
      std::get<1>(weights_views),
      std::get<1>(frequencies_views),
      std::get<1>(phases_views),
      std::get<1>(coordinates_views),
      std::get<1>(cf_phase_screens_views),
      m_grid_scale,
      m_grid,
      m_weights);
  }

#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  void
  alt1_grid_visibilities(
    Device /*host_device*/,
    size_t offset,
    const vector_data<std::complex<visibility_fp>>& visibilities,
    const vector_data<unsigned>& vis_grid_cubes,
    const vector_data<vis_cf_index_t>& vis_cf_indexes,
    const vector_data<vis_weight_fp>& vis_weights,
    const vector_data<vis_frequency_fp>& vis_frequencies,
    const vector_data<vis_phase_fp>& vis_phases,
    const vector_data<vis_uvw_t>& vis_coordinates,
    const vector_data<cf_phase_screen_t>& vis_cf_phase_screens) {

    auto& exec_copy = m_exec_spaces[next_exec_space(StreamPhase::COPY)];

    auto len =
      std::min(visibilities->size() - offset, m_max_visibility_batch_size);

    exec_copy.vis_state.emplace_back(visibilities);
    auto vis_views =
      StateT<D>::copy_to_device_view<vis_t>(
        exec_copy.visibilities,
        visibilities,
        offset,
        len,
        exec_copy.space);
    exec_copy.vis_state.push_back(vis_views);

    exec_copy.vis_state.emplace_back(vis_grid_cubes);
    auto grid_cubes_views =
      StateT<D>::copy_to_device_view<unsigned>(
        exec_copy.grid_cubes,
        vis_grid_cubes,
        offset,
        len,
        exec_copy.space);
    exec_copy.vis_state.push_back(grid_cubes_views);

    exec_copy.vis_state.emplace_back(vis_cf_indexes);
    auto cf_indexes_views =
      StateT<D>::copy_to_device_view<cf_index_t>(
        exec_copy.cf_indexes,
        vis_cf_indexes,
        offset,
        len,
        exec_copy.space);
    exec_copy.vis_state.push_back(cf_indexes_views);

    exec_copy.vis_state.emplace_back(vis_weights);
    auto weights_views =
      StateT<D>::copy_to_device_view<vis_weight_fp>(
        exec_copy.weights,
        vis_weights,
        offset,
        len,
        exec_copy.space);
    exec_copy.vis_state.push_back(weights_views);

    exec_copy.vis_state.emplace_back(vis_frequencies);
    auto frequencies_views =
      StateT<D>::copy_to_device_view<vis_frequency_fp>(
        exec_copy.frequencies,
        vis_frequencies,
        offset,
        len,
        exec_copy.space);
    exec_copy.vis_state.push_back(frequencies_views);

    exec_copy.vis_state.emplace_back(vis_phases);
    auto phases_views =
      StateT<D>::copy_to_device_view<vis_phase_fp>(
        exec_copy.phases,
        vis_phases,
        offset,
        len,
        exec_copy.space);
    exec_copy.vis_state.push_back(phases_views);

    exec_copy.vis_state.emplace_back(vis_coordinates);
    auto coordinates_views =
      StateT<D>::copy_to_device_view<vis_uvw_t>(
        exec_copy.coordinates,
        vis_coordinates,
        offset,
        len,
        exec_copy.space);
    exec_copy.vis_state.push_back(coordinates_views);

    exec_copy.vis_state.emplace_back(vis_cf_phase_screens);
    auto cf_phase_screens_views =
      StateT<D>::copy_to_device_view<cf_ps_t>(
        exec_copy.cf_phase_screens,
        vis_cf_phase_screens,
        offset,
        len,
        exec_copy.space);
    exec_copy.vis_state.push_back(cf_phase_screens_views);

    auto& exec_compute = m_exec_spaces[next_exec_space(StreamPhase::COMPUTE)];
    auto& cf = std::get<0>(m_cfs[m_cf_indexes.front()]);
    Core::VisibilityGridder<execution_space, 1>::kernel(
      exec_compute.space,
      cf.cf_d,
      cf.cf_radii,
      cf.max_cf_extent_y,
      len,
      std::get<1>(vis_views),
      std::get<1>(grid_cubes_views),
      std::get<1>(cf_indexes_views),
      std::get<1>(weights_views),
      std::get<1>(frequencies_views),
      std::get<1>(phases_views),
      std::get<1>(coordinates_views),
      std::get<1>(cf_phase_screens_views),
      m_grid_scale,
      m_grid,
      m_weights);
  }
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS

  std::optional<Error>
  grid_visibilities(
    Device host_device,
    std::vector<std::complex<visibility_fp>>&& visibilities,
    std::vector<unsigned>&& vis_grid_cubes,
    std::vector<vis_cf_index_t>&& vis_cf_indexes,
    std::vector<vis_weight_fp>&& vis_weights,
    std::vector<vis_frequency_fp>&& vis_frequencies,
    std::vector<vis_phase_fp>&& vis_phases,
    std::vector<vis_uvw_t>&& vis_coordinates,
    std::optional<std::vector<cf_phase_screen_t>>&& vis_cf_phase_screens)
    override {

    const auto vis =
      std::make_shared<std::vector<std::complex<visibility_fp>>>(
        std::move(visibilities));
    const auto grid_cubes =
      std::make_shared<std::vector<unsigned>>(
        std::move(vis_grid_cubes));
    const auto cf_indexes =
      std::make_shared<std::vector<vis_cf_index_t>>(
        std::move(vis_cf_indexes));
#ifndef NDEBUG
    for (auto& [cube, supp] : *cf_indexes) {
      auto& cfpool = std::get<0>(m_cfs[m_cf_indexes.front()]);
      if ((supp >= cfpool.num_cf_groups)
          || (cube >= cfpool.cf_d[supp].extent_int(5)))
        return OutOfBoundsCFIndexError({cube, supp});
    }
#endif // NDEBUG
    const auto weights =
      std::make_shared<std::vector<vis_weight_fp>>(
        std::move(vis_weights));
    const auto frequencies =
      std::make_shared<std::vector<vis_frequency_fp>>(
        std::move(vis_frequencies));
    const auto phases =
      std::make_shared<std::vector<vis_phase_fp>>(
        std::move(vis_phases));
    const auto coordinates =
      std::make_shared<std::vector<vis_uvw_t>>(
        std::move(vis_coordinates));
    std::shared_ptr<std::vector<cf_phase_screen_t>> cf_phase_screens;
    if (vis_cf_phase_screens)
      cf_phase_screens =
        std::make_shared<std::vector<cf_phase_screen_t>>(
          std::move(vis_cf_phase_screens).value());
    else
      cf_phase_screens =
        std::make_shared<std::vector<cf_phase_screen_t>>();

    size_t num_visibilities = vis->size();
    switch (visibility_gridder_version()) {
    case 0:
      for (size_t i = 0; i < num_visibilities; i += m_max_visibility_batch_size)
        default_grid_visibilities(
          host_device,
          i,
          vis,
          grid_cubes,
          cf_indexes,
          weights,
          frequencies,
          phases,
          coordinates,
          cf_phase_screens);
      break;
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    case 1:
      for (size_t i = 0; i < num_visibilities; i += m_max_visibility_batch_size)
        alt1_grid_visibilities(
          host_device,
          i,
          vis,
          grid_cubes,
          cf_indexes,
          weights,
          frequencies,
          phases,
          coordinates,
          cf_phase_screens);
      break;
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    }
    return std::nullopt;
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

  void
  reset_grid() override {
    fence();
    new_grid(true, true);
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
  apply_fft(FFTSign sign, bool in_place) override {
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
      if constexpr (!std::is_same_v<K::HostSpace, memory_space>) {
        auto& esp = m_exec_spaces.back();
        esp.visibilities =
          decltype(esp.visibilities)(
            K::ViewAllocateWithoutInitializing("visibilities"),
            m_max_visibility_batch_size);
        esp.grid_cubes =
          decltype(esp.grid_cubes)(
            K::ViewAllocateWithoutInitializing("grid_cubes"),
            m_max_visibility_batch_size);
        esp.cf_indexes =
          decltype(esp.cf_indexes)(
            K::ViewAllocateWithoutInitializing("cf_indexes"),
            m_max_visibility_batch_size);
        esp.weights =
          decltype(esp.weights)(
            K::ViewAllocateWithoutInitializing("weights"),
            m_max_visibility_batch_size);
        esp.frequencies =
          decltype(esp.frequencies)(
            K::ViewAllocateWithoutInitializing("frequencies"),
            m_max_visibility_batch_size);
        esp.phases =
          decltype(esp.phases)(
            K::ViewAllocateWithoutInitializing("phases"),
            m_max_visibility_batch_size);
        esp.coordinates =
          decltype(esp.coordinates)(
            K::ViewAllocateWithoutInitializing("coordinates"),
            m_max_visibility_batch_size);
        esp.cf_phase_screens =
          decltype(esp.cf_phase_screens)(
            K::ViewAllocateWithoutInitializing("cf_phase_screens"),
            m_max_visibility_batch_size);
      }
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
        K::deep_copy(esp.space, esp.visibilities, st_esp.visibilities);
        K::deep_copy(esp.space, esp.grid_cubes, st_esp.grid_cubes);
        K::deep_copy(esp.space, esp.cf_indexes, st_esp.cf_indexes);
        K::deep_copy(esp.space, esp.weights, st_esp.weights);
        K::deep_copy(esp.space, esp.frequencies, st_esp.frequencies);
        K::deep_copy(esp.space, esp.phases, st_esp.phases);
        K::deep_copy(esp.space, esp.coordinates, st_esp.coordinates);
        K::deep_copy(esp.space, esp.cf_phase_screens, st_esp.cf_phase_screens);
        esp.vis_state = st_esp.vis_state;
      }
      for (auto& i : ost->m_cf_indexes) {
        auto& [cf, last] = m_cfs[i];
        m_cf_indexes.push_back(i);
        cf = std::get<0>(ost->m_cfs[i]);
        cf.state = this;
      }
    }
    m_current = StreamPhase::COPY;
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
      m_exec_spaces[new_idx].vis_state.clear();
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

    if (also_weights) {
      if (create_without_init)
        m_weights =
          decltype(m_weights)(
            K::ViewAllocateWithoutInitializing("weights"),
            static_cast<int>(m_grid_size[2]),
            static_cast<int>(m_grid_size[3]));
      else
        m_weights =
          decltype(m_weights)(
            K::view_alloc(
              "weights",
              m_exec_spaces[next_exec_space(StreamPhase::COPY)].space),
            static_cast<int>(m_grid_size[2]),
            static_cast<int>(m_grid_size[3]));
    }
    if (std::holds_alternative<const StateT*>(source)) {
      auto st = std::get<const StateT*>(source);
      auto& exec = m_exec_spaces[next_exec_space(StreamPhase::COPY)];
      K::deep_copy(exec.space, m_grid, st->m_grid);
      if (also_weights)
        K::deep_copy(exec.space, m_weights, st->m_weights);
    }
  }

  template <typename DT, typename ST>
  std::tuple<vector_view<const DT>, K::View<const DT*, memory_space>>
  copy_to_device_view(
    const K::View<DT*, memory_space>& dview,
    const vector_data<ST>& vect,
    size_t offset,
    size_t len,
    execution_space& exec) {

    if (vect->size() > 0) {
      vector_view<const DT>
        hview(reinterpret_cast<const DT*>(vect->data() + offset), len);
      if constexpr (!std::is_same_v<K::HostSpace, memory_space>) {
          auto dv = K::subview(dview, std::pair((size_t)0, len));
          K::deep_copy(exec, dv, hview);
          return {hview, dv};
      } else {
        return {hview, hview};
      }
#ifdef WORKAROUND_NVCC_IF_CONSTEXPR_BUG
      return
        std::tuple<vector_view<const DT>, K::View<const DT*, memory_space>>();
#endif
    } else {
      return {vector_view<const DT>(), K::View<const DT*, memory_space>()};
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
