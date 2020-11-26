#pragma once

#include "hpg.hpp"

#include <algorithm>
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

namespace K = Kokkos;
namespace KExp = Kokkos::Experimental;

namespace hpg {
namespace Impl {

/** visibility value type */
using vis_t = K::complex<visibility_fp>;

/** convolution function value type */
using cf_t = K::complex<cf_fp>;

/** gridded value type */
using gv_t = K::complex<grid_value_fp>;

/** visibility data plus metadata for gridding */
struct Visibility {

  KOKKOS_INLINE_FUNCTION Visibility() {};

  Visibility(
    const vis_t& value_, /**< visibility value */
    unsigned grid_cube_, /**< grid cube index */
    unsigned cf_cube_, /**< cf cube index */
    vis_weight_fp weight_, /**< visibility weight */
    vis_frequency_fp freq_, /**< frequency */
    vis_phase_fp d_phase_, /**< phase angle */
    const vis_uvw_t& uvw_ /** < uvw coordinates */)
    : value(value_)
    , grid_cube(grid_cube_)
    , cf_cube(cf_cube_)
    , weight(weight_)
    , freq(freq_)
    , d_phase(d_phase_) {
    uvw[0] = std::get<0>(uvw_);
    uvw[1] = std::get<1>(uvw_);
    uvw[2] = std::get<2>(uvw_);
  }

  Visibility(Visibility const&) = default;

  Visibility(Visibility&&) = default;

  KOKKOS_INLINE_FUNCTION ~Visibility() = default;

  KOKKOS_INLINE_FUNCTION Visibility& operator=(Visibility const&) = default;

  KOKKOS_INLINE_FUNCTION Visibility& operator=(Visibility&&) = default;

  vis_t value;
  int grid_cube;
  int cf_cube;
  vis_weight_fp weight;
  vis_frequency_fp freq;
  vis_phase_fp d_phase;
  K::Array<vis_uvw_fp, 3> uvw;
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

  static unsigned constexpr active_task_limit = 0;

  using stream_type = void;
};

#ifdef HPG_ENABLE_SERIAL
template <>
struct DeviceT<Device::Serial> {
  using kokkos_device = K::Serial;

  static unsigned constexpr active_task_limit = 1;

  using stream_type = void;
};
#endif // HPG_ENABLE_SERIAL

#ifdef HPG_ENABLE_OPENMP
template <>
struct DeviceT<Device::OpenMP> {
  using kokkos_device = K::OpenMP;

  static unsigned constexpr active_task_limit = 1;

  using stream_type = void;
};
#endif // HPG_ENABLE_OPENMP

#ifdef HPG_ENABLE_CUDA
template <>
struct DeviceT<Device::Cuda> {
  using kokkos_device = K::Cuda;

  static unsigned constexpr active_task_limit = 4;

  using stream_type = cudaStream_t;

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

  static unsigned constexpr active_task_limit = 1;

  using stream_type = void;
};
#endif // HPG_ENABLE_HPX

/** View type for grid values */
template <typename Layout, typename memory_space>
using grid_view = K::View<gv_t****, Layout, memory_space>;

template <typename Layout, typename memory_space>
using const_grid_view = K::View<const gv_t****, Layout, memory_space>;

/** View type for weight values
 *
 * logical axis order: stokes, cube
 */
template <typename Layout, typename memory_space>
using weight_view = K::View<grid_value_fp**, Layout, memory_space>;

template <typename Layout, typename memory_space>
using const_weight_view = K::View<const grid_value_fp**, Layout, memory_space>;

/** View type for CF values */
template <typename Layout, typename memory_space>
using cf_view = K::View<cf_t******, Layout, memory_space>;

template <typename Layout, typename memory_space>
using const_cf_view = K::View<const cf_t******, Layout, memory_space>;

/** View type for Visibility values */
template <typename memory_space>
using visibility_view = K::View<Visibility*, memory_space>;

template <typename memory_space>
using const_visibility_view = K::View<const Visibility*, memory_space>;

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
   * logical index order: X, Y, stokes, cube
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
  }
};

/** device-specific CF array layout */
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
   * create Kokkos layout using given CFArray
   *
   * logical index order: X, Y, stokes, x, y, cube
   */
  static layout
  dimensions(const CFArray& cf) {
    std::array<int, 6> dims{
      static_cast<int>(cf.extent(0) / cf.oversampling()),
      static_cast<int>(cf.extent(1) / cf.oversampling()),
      static_cast<int>(cf.extent(2)),
      static_cast<int>(cf.oversampling()),
      static_cast<int>(cf.oversampling()),
      static_cast<int>(cf.extent(3))
    };
    if constexpr (std::is_same_v<layout, K::LayoutLeft>) {
      return
        K::LayoutLeft(dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]);
    } else {
      static const std::array<int, 6> order{1, 2, 0, 4, 3, 5};
      return K::LayoutStride::order_dimensions(6, order.data(), dims.data());
    }
  }
};

/** convert UV coordinate to major and minor grid coordinates
 */
KOKKOS_FUNCTION std::tuple<int, int>
compute_vis_coord(
  int g_offset,
  int oversampling,
  int cf_radius,
  vis_uvw_fp coord,
  vis_frequency_fp inv_lambda,
  grid_scale_fp fine_scale) {

  long fine_coord = std::lrint((double(coord) * inv_lambda) * fine_scale);
  long nearest_major_fine_coord =
    std::lrint(double(fine_coord) / oversampling) * oversampling;
  int major = nearest_major_fine_coord / oversampling - cf_radius + g_offset;
  int minor;
  if (fine_coord >= nearest_major_fine_coord)
    minor = fine_coord - nearest_major_fine_coord;
  else
    minor = fine_coord - (nearest_major_fine_coord - oversampling);
  assert(0 <= minor && minor < oversampling);
  return {major, minor};
}

/** portable sincos()
 */
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

/** helper class for computing visibility value and index metadata
 *
 * Basically exists to encapsulate conversion from a Visibility value to several
 * visibility metadata values needed by gridding kernel
 */
template <typename execution_space>
struct GridVis final {

  int major[2]; /**< major grid coordinate */
  int minor[2]; /**< minor grid coordinate */
  vis_t value; /**< visibility value */
  int grid_cube; /**< grid cube index */
  int cf_cube; /**< cf cube index */
  vis_weight_fp weight; /**< visibility weight */
  cf_fp cf_im_factor; /**< weight conjugation factor */

  KOKKOS_INLINE_FUNCTION GridVis() {};

  KOKKOS_INLINE_FUNCTION GridVis(
    const Visibility& vis,
    const K::Array<int, 2>& grid_radius,
    const K::Array<int, 2>& oversampling,
    const K::Array<int, 2>& cf_radius,
    const K::Array<grid_scale_fp, 2>& fine_scale)
    : grid_cube(vis.grid_cube)
    , cf_cube(vis.cf_cube)
    , weight(vis.weight) {

    static const vis_frequency_fp c = 299792458.0;
    K::complex<vis_phase_fp> phasor;
    sincos<execution_space>(vis.d_phase, &phasor.imag(), &phasor.real());
    value = vis.value * phasor * vis.weight;
    auto inv_lambda = vis.freq / c;
    // can't use std::tie here - CUDA doesn't support it
    auto [c0, f0] =
      compute_vis_coord(
        grid_radius[0],
        oversampling[0],
        cf_radius[0],
        vis.uvw[0],
        inv_lambda,
        fine_scale[0]);
    major[0] = c0;
    minor[0] = f0;
    auto [c1, f1] =
      compute_vis_coord(
        grid_radius[1],
        oversampling[1],
        cf_radius[1],
        vis.uvw[1],
        inv_lambda,
        fine_scale[1]);
    major[1] = c1;
    minor[1] = f1;
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
  static constexpr int n_sto = 4;

  cf_t wgts[n_sto];

  KOKKOS_INLINE_FUNCTION cf_wgt_array() {
     init();
  }

  KOKKOS_INLINE_FUNCTION cf_wgt_array(const cf_wgt_array& rhs) {
    for (int i = 0; i < n_sto; ++i)
      wgts[i] = rhs.wgts[i];
  }


  KOKKOS_INLINE_FUNCTION void
  init() {
    for (int i = 0; i < n_sto; ++i)
      wgts[i] = 0;
  }

  KOKKOS_INLINE_FUNCTION cf_wgt_array&
  operator +=(const cf_wgt_array& src) {
    for (int i = 0; i < n_sto; ++i)
      wgts[i] += src.wgts[i];
    return *this;
  }

  KOKKOS_INLINE_FUNCTION void
  operator +=(const volatile cf_wgt_array& src) volatile {
    for (int i = 0; i < n_sto; ++i)
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
    Kokkos::View<value_type[1], space, K::MemoryUnmanaged> result_view_type;

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

  template <typename cf_layout, typename grid_layout, typename memory_space>
  static void
  kernel(
    execution_space exec,
    const const_cf_view<cf_layout, memory_space>& cf,
    const const_visibility_view<memory_space>& visibilities,
    const std::array<grid_scale_fp, 2>& grid_scale,
    const grid_view<grid_layout, memory_space>& grid,
    const weight_view<typename execution_space::array_layout, memory_space>&
    weights) {

    const K::Array<int, 2>
      cf_radius{cf.extent_int(0) / 2, cf.extent_int(1) / 2};
    const K::Array<int, 2>
      grid_radius{grid.extent_int(0) / 2, grid.extent_int(1) / 2};
    const K::Array<int, 2>
      oversampling{cf.extent_int(3), cf.extent_int(4)};
    const K::Array<grid_scale_fp, 2> fine_scale{
      grid_scale[0] * oversampling[0],
      grid_scale[1] * oversampling[1]};

    using member_type = typename K::TeamPolicy<execution_space>::member_type;

    using scratch_wgts_view =
      K::View<cf_wgt_array[1], typename execution_space::scratch_memory_space>;

    K::parallel_for(
      "gridding",
      K::TeamPolicy<execution_space>(
        exec,
        static_cast<int>(visibilities.size()),
        K::AUTO)
      .set_scratch_size(0, K::PerTeam(scratch_wgts_view::shmem_size())),
      KOKKOS_LAMBDA(const member_type& team_member) {
        GridVis<execution_space> vis(
          visibilities(team_member.league_rank()),
          grid_radius,
          oversampling,
          cf_radius,
          fine_scale);
        // convenience variables
        const int N_X = cf.extent_int(0);
        const int N_Y = cf.extent_int(1);
        const int N_S = cf.extent_int(2);
        // accumulate weights in scratch memory for this visibility
        scratch_wgts_view cfw(team_member.team_scratch(0));
        K::parallel_for(
          K::TeamVectorRange(team_member, cf.extent_int(2)),
          [=](const int S) {
            cfw(0).wgts[S] = 0;
          });
        team_member.team_barrier();
        /* loop over majorX */
        K::parallel_reduce(
          K::TeamVectorRange(team_member, N_X),
          [=](const int X, cf_wgt_array& cfw_l) {
            /* loop over elements (rows) of Mueller matrix column  */
            for (int S = 0; S < N_S; ++S){
              /* loop over majorY */
              for (int Y = 0; Y < N_Y; ++Y) {
                cf_t cfv = cf(X, Y, S, vis.minor[0], vis.minor[1], vis.cf_cube);
                cfv.imag() *= vis.cf_im_factor;
                pseudo_atomic_add<execution_space>(
                  grid(vis.major[0] + X, vis.major[1] + Y, S, vis.grid_cube),
                  gv_t(cfv * vis.value));
                cfw_l.wgts[S] += cfv;
              }
            }
          },
          SumCFWgts<execution_space>(cfw(0)));
        // by Kokkos reduction semantics the following barrier should not be
        // needed, but recent Slack discussion indicates a possible bug, so we
        // use it here until the issue is resolved
        team_member.team_barrier();
        // update weights array
        K::parallel_for(
          K::TeamVectorRange(team_member, N_S),
          [=](const int S) {
            K::atomic_add(
              &weights(S, vis.grid_cube),
              grid_value_fp(
                std::hypot(cfw(0).wgts[S].real(), cfw(0).wgts[S].imag())
                * vis.weight));
          });
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

    K::parallel_for(
      "normalization",
      K::MDRangePolicy<K::Rank<4>, execution_space>(
        exec,
        {0, 0, 0, 0},
        {grid.extent_int(0),
         grid.extent_int(1),
         grid.extent_int(2),
         grid.extent_int(3)}),
      KOKKOS_LAMBDA(int x, int y, int stokes, int cube) {
        grid(x, y, stokes, cube) /= (wfactor * weights(stokes, cube));
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
         reinterpret_cast<complex_t*>(in + 1), inembed, istride, idist,
         reinterpret_cast<complex_t*>(out + 1), onembed, ostride, odist,
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
      size_t prev_stride = 0;
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
    int stride = igrid.extent_int(2);
    int dist = igrid.extent_int(0) * igrid.extent_int(1) * igrid.extent_int(2);
    auto result =
      FFTW<scalar_t>::plan_many(
        2, n, igrid.extent_int(3),
        const_cast<K::complex<scalar_t>*>(&igrid(0, 0, 0, 0)),
        NULL, stride, dist,
        &ogrid(0, 0, 0, 0), NULL, stride, dist,
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
      for (int sto = 0; sto < grid.extent_int(2); ++sto) {
        FFTW<scalar_t>::exec(h0, &grid(0, 0, sto, 0), &grid(0, 0, sto, 0));
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
      for (int sto = 0; sto < pre_grid.extent_int(2); ++sto) {
        FFTW<scalar_t>::exec(
          h0,
          const_cast<K::complex<scalar_t>*>(&pre_grid(0, 0, sto, 0)),
          &post_grid(0, 0, sto, 0));
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
    int n_sto = grid.extent_int(2);
    int n_cube = grid.extent_int(3);

    int mid_x = n_x / 2;
    int mid_y = n_y / 2;

    if (n_x % 2 == 0 && n_y % 2 == 0) {
      // simpler (faster?) algorithm when both grid side lengths are even

      K::parallel_for(
        "grid_shift_ee",
        K::TeamPolicy<execution_space>(exec, n_sto * n_cube, K::AUTO),
        KOKKOS_LAMBDA(const member_type& team_member) {
          auto gplane =
            K::subview(
              grid,
              K::ALL,
              K::ALL,
              team_member.league_rank() % n_sto,
              team_member.league_rank() / n_sto);
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
        K::TeamPolicy<execution_space>(exec, n_sto * n_cube, K::AUTO),
        KOKKOS_LAMBDA(const member_type& team_member) {
          auto gplane =
            K::subview(
              grid,
              K::ALL,
              K::ALL,
              team_member.league_rank() % n_sto,
              team_member.league_rank() / n_sto);
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
        K::TeamPolicy<execution_space>(exec, n_sto * n_cube, K::AUTO),
        KOKKOS_LAMBDA(const member_type& team_member) {
          auto gplane =
            K::subview(
              grid,
              K::ALL,
              K::ALL,
              team_member.league_rank() % n_sto,
              team_member.league_rank() / n_sto);

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

  Device device; /**< device type */
  unsigned max_active_tasks; /**< maximum number of active tasks */
  std::array<unsigned, 4> grid_size; /**< grid size */
  std::array<grid_scale_fp, 2> grid_scale; /**< grid scale */
  std::array<unsigned, 4> implementation_versions; /**< impl versions*/

  State(Device device_)
    : device(device_) {}

  State(
    Device device_,
    unsigned max_active_tasks_,
    const std::array<unsigned, 4>& grid_size_,
    const std::array<grid_scale_fp, 2>& grid_scale_,
    const std::array<unsigned, 4>& implementation_versions_)
    : device(device_)
    , max_active_tasks(max_active_tasks_)
    , grid_size(grid_size_)
    , grid_scale(grid_scale_)
    , implementation_versions(implementation_versions_) {}

  unsigned
  visibility_gridder_version() const {
    return implementation_versions[0];
  }

  unsigned
  grid_normalizer_version() const {
    return implementation_versions[1];
  }

  unsigned
  fft_version() const {
    return implementation_versions[2];
  }

  unsigned
  grid_shifter_version() const {
    return implementation_versions[3];
  }

  virtual std::optional<Error>
  set_convolution_function(Device host_device, const CFArray& cf) = 0;

  virtual void
  grid_visibilities(
    Device host_device,
    const std::vector<std::complex<visibility_fp>>& visibilities,
    const std::vector<unsigned> visibility_grid_cubes,
    const std::vector<unsigned> visibility_cf_cubes,
    const std::vector<vis_weight_fp>& visibility_weights,
    const std::vector<vis_frequency_fp>& visibility_frequencies,
    const std::vector<vis_phase_fp>& visibility_phases,
    const std::vector<vis_uvw_t>& visibility_coordinates) = 0;

  virtual void
  fence() const volatile = 0;

  virtual std::unique_ptr<GridWeightArray>
  grid_weights() const volatile = 0;

  virtual std::unique_ptr<GridValueArray>
  grid_values() const volatile = 0;

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
  using grid_t =
    typename const_grid_view<layout, memory_space>::host_mirror_type;

  grid_t grid;

  GridValueViewArray(const grid_t& grid_)
    : grid(grid_) {}

  ~GridValueViewArray() {}

  unsigned
  extent(unsigned dim) const override {
    return grid.extent(dim);
  }

  std::complex<grid_value_fp>
  operator()(unsigned x, unsigned y, unsigned stokes, unsigned cube)
    const override {

    return grid(x, y, stokes, cube);
  }
};

/** concrete sub-class of abstract GridWeightArray */
template <Device D>
class HPG_EXPORT GridWeightViewArray final
  : public GridWeightArray {
 public:

  using memory_space = typename DeviceT<D>::kokkos_device::memory_space;
  using layout = typename DeviceT<D>::kokkos_device::array_layout;
  using weight_t =
    typename const_weight_view<layout, memory_space>::host_mirror_type;

  weight_t weight;

  GridWeightViewArray(const weight_t& weight_)
    : weight(weight_) {}

  ~GridWeightViewArray() {}

  unsigned
  extent(unsigned dim) const override {
    return weight.extent(dim);
  }

  grid_value_fp
  operator()(unsigned stokes, unsigned cube) const override {

    return weight(stokes, cube);
  }
};

/** initialize CF array view from CFArray instance */
template <Device D, typename CFH>
static void
init_cf_host(CFH& cf_h, const CFArray& cf) {
  static_assert(
    K::SpaceAccessibility<
      typename DeviceT<D>::kokkos_device::memory_space,
      K::HostSpace>
    ::accessible);

  K::parallel_for(
    "cf_init",
    K::MDRangePolicy<K::Rank<4>, typename DeviceT<D>::kokkos_device>(
      {0, 0, 0, 0},
      {static_cast<int>(cf.extent(0)),
       static_cast<int>(cf.extent(1)),
       static_cast<int>(cf.extent(2)),
       static_cast<int>(cf.extent(3))}),
    [&](int i, int j, int poln, int cube) {
      auto X = i / cf.oversampling();
      auto x = i % cf.oversampling();
      auto Y = j / cf.oversampling();
      auto y = j % cf.oversampling();
      cf_h(X, Y, poln, x, y, cube) = cf(i, j, poln, cube);
    });
}

/** initialize visibility array view from visibility values and metadata */
template <Device D, typename VisH>
static void
init_vis(
  VisH& vis_h,
  const std::vector<std::complex<visibility_fp>>& visibilities,
  const std::vector<unsigned> visibility_grid_cubes,
  const std::vector<unsigned> visibility_cf_cubes,
  const std::vector<vis_weight_fp>& visibility_weights,
  const std::vector<vis_frequency_fp>& visibility_frequencies,
  const std::vector<vis_phase_fp>& visibility_phases,
  const std::vector<vis_uvw_t>& visibility_coordinates) {

  static_assert(
    K::SpaceAccessibility<
    typename DeviceT<D>::kokkos_device::memory_space,
    K::HostSpace>
    ::accessible);
  K::parallel_for(
    "vis_init",
    K::RangePolicy<typename DeviceT<D>::kokkos_device>(
      0,
      static_cast<int>(visibilities.size())),
    KOKKOS_LAMBDA(int i) {
      vis_h(i) =
        Visibility(
          visibilities[i],
          visibility_grid_cubes[i],
          visibility_cf_cubes[i],
          visibility_weights[i],
          visibility_frequencies[i],
          visibility_phases[i],
          visibility_coordinates[i]);
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

/** Kokkos state implementation for a device type */
template <Device D>
struct HPG_EXPORT StateT final
  : public State {
public:

  using kokkos_device = typename DeviceT<D>::kokkos_device;
  using execution_space = typename kokkos_device::execution_space;
  using memory_space = typename execution_space::memory_space;
  using stream_type = typename DeviceT<D>::stream_type;

  grid_view<typename GridLayout<D>::layout, memory_space> grid;
  const_cf_view<typename CFLayout<D>::layout, memory_space> cf;
  weight_view<typename execution_space::array_layout, memory_space> weights;

  // use multiple execution spaces to support overlap of data copying with
  // computation when possible
  std::vector<
    std::conditional_t<std::is_void_v<stream_type>, int, stream_type>> streams;
  std::vector<execution_space> exec_spaces;
  mutable std::deque<int> exec_space_indexes;
  mutable StreamPhase current;

  StateT(
    unsigned max_active_tasks,
    const std::array<unsigned, 4> grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale,
    const std::array<unsigned, 4>& implementation_versions)
    : State(
      D,
      std::min(max_active_tasks, DeviceT<D>::active_task_limit),
      grid_size,
      grid_scale,
      implementation_versions) {

    init_exec_spaces();
    new_grid(true, true);
  }

  StateT(const volatile StateT& st)
    : State(
      D,
      const_cast<const StateT&>(st).max_active_tasks,
      const_cast<const StateT&>(st).grid_size,
      const_cast<const StateT&>(st).grid_scale,
      const_cast<const StateT&>(st).implementation_versions) {

    init_exec_spaces();
    new_grid(const_cast<const StateT*>(&st), true);
    cf = const_cast<const StateT*>(&st)->cf;
  }

  StateT(StateT&& st)
    : State(D) {
    swap(st);
  }

  virtual ~StateT() {
    fence();
    grid = decltype(grid)();
    weights = decltype(weights)();
    cf = decltype(cf)();
    if constexpr(!std::is_void_v<stream_type>) {
      for (auto& str : streams) {
        auto rc = DeviceT<D>::destroy_stream(str);
        assert(rc);
      }
    }
  }

  StateT&
  operator=(const volatile StateT& st) {
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
  copy() const volatile {
    return StateT(*this);
  }

  std::optional<Error>
  set_convolution_function(Device host_device, const CFArray& cf_array)
    override {

    if (cf_array.extent(2) != grid_size[2])
      return Error("Unequal size of Stokes dimension in grid and CF");
    if (cf_array.extent(0) > grid_size[0] * cf_array.oversampling()
        || cf_array.extent(1) > grid_size[1] * cf_array.oversampling())
      return Error("CF support size exceeds grid size");

    cf_view<typename CFLayout<D>::layout, memory_space> cf_init(
      K::ViewAllocateWithoutInitializing("cf"),
      CFLayout<D>::dimensions(cf_array));
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

    auto exec = next_exec_space(StreamPhase::COPY);

    switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
    case Device::Serial: {
      auto cf_h = K::create_mirror_view(cf_init);
      init_cf_host<Device::Serial>(cf_h, cf_array);
      K::deep_copy(exec, cf_init, cf_h);
      break;
    }
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
    case Device::OpenMP: {
      auto cf_h = K::create_mirror_view(cf_init);
      init_cf_host<Device::OpenMP>(cf_h, cf_array);
      K::deep_copy(exec, cf_init, cf_h);
      break;
    }
#endif // HPG_ENABLE_SERIAL
    default:
      assert(false);
      break;
    }
    cf = cf_init;
    return std::nullopt;
  }

  void
  grid_visibilities(
    Device host_device,
    const std::vector<std::complex<visibility_fp>>& visibilities,
    const std::vector<unsigned> visibility_grid_cubes,
    const std::vector<unsigned> visibility_cf_cubes,
    const std::vector<vis_weight_fp>& visibility_weights,
    const std::vector<vis_frequency_fp>& visibility_frequencies,
    const std::vector<vis_phase_fp>& visibility_phases,
    const std::vector<vis_uvw_t>& visibility_coordinates) override {

    auto exec_copy = next_exec_space(StreamPhase::COPY);

    visibility_view<memory_space> vis(
      K::ViewAllocateWithoutInitializing("visibilities"),
      visibilities.size());

    switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
    case Device::Serial: {
      auto vis_h = K::create_mirror_view(vis);
      init_vis<Device::Serial>(
        vis_h,
        visibilities,
        visibility_grid_cubes,
        visibility_cf_cubes,
        visibility_weights,
        visibility_frequencies,
        visibility_phases,
        visibility_coordinates);
      K::deep_copy(exec_copy, vis, vis_h);
      break;
    }
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
    case Device::OpenMP: {
      auto vis_h = K::create_mirror_view(vis);
      init_vis<Device::OpenMP>(
        vis_h,
        visibilities,
        visibility_grid_cubes,
        visibility_cf_cubes,
        visibility_weights,
        visibility_frequencies,
        visibility_phases,
        visibility_coordinates);
      K::deep_copy(exec_copy, vis, vis_h);
      break;
    }
#endif // HPG_ENABLE_SERIAL
    default:
      assert(false);
      break;
    }

    const_visibility_view<memory_space> cvis = vis;
    switch (visibility_gridder_version()) {
    case 0:
      Core::VisibilityGridder<execution_space, 0>::kernel(
        next_exec_space(StreamPhase::COMPUTE),
        cf,
        cvis,
        grid_scale,
        grid,
        weights);
      break;
    default:
      assert(false);
      break;
    }
  }

  void
  fence() const volatile override {
    auto st = const_cast<StateT*>(this);
    for (unsigned i = 0; i < st->exec_space_indexes.size(); ++i) {
      st->exec_spaces[st->exec_space_indexes.front()].fence();
      st->exec_space_indexes.push_back(st->exec_space_indexes.front());
      st->exec_space_indexes.pop_front();
    }
    current = StreamPhase::COPY;
  }

  std::unique_ptr<GridWeightArray>
  grid_weights() const volatile override {
    auto st = const_cast<StateT*>(this);
    auto exec = st->next_exec_space(StreamPhase::COPY);
    auto wgts_h = K::create_mirror(st->weights);
    K::deep_copy(exec, wgts_h, st->weights);
    exec.fence();
    return std::make_unique<GridWeightViewArray<D>>(wgts_h);
  }

  std::unique_ptr<GridValueArray>
  grid_values() const volatile override {
    auto st = const_cast<StateT*>(this);
    auto exec = st->next_exec_space(StreamPhase::COPY);
    auto grid_h = K::create_mirror(st->grid);
    K::deep_copy(exec, grid_h, st->grid);
    exec.fence();
    return std::make_unique<GridValueViewArray<D>>(grid_h);
  }

  void
  reset_grid() override {
    new_grid(true, true);
  }

  void
  normalize(grid_value_fp wfactor) override {
    const_weight_view<typename execution_space::array_layout, memory_space>
      cweights = weights;
    switch (grid_normalizer_version()) {
    case 0:
      Core::GridNormalizer<execution_space, 0>::kernel(
        next_exec_space(StreamPhase::COMPUTE),
        grid,
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
          ::in_place_kernel(next_exec_space(StreamPhase::COMPUTE), sign, grid);
        break;
      default:
        assert(false);
        break;
      }
    } else {
      const_grid_view<typename GridLayout<D>::layout, memory_space> pre_grid
        = grid;
      new_grid(false, false);
      switch (fft_version()) {
      case 0:
        err =
          Core::FFT<execution_space, 0>::out_of_place_kernel(
            next_exec_space(StreamPhase::COMPUTE),
            sign,
            pre_grid,
            grid);
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
        next_exec_space(StreamPhase::COMPUTE),
        grid);
      break;
    default:
      assert(false);
      break;
    }
  }

private:
  void
  swap(StateT& other) {
    std::swap(streams, other.streams);
    std::swap(exec_spaces, other.exec_spaces);
    std::swap(exec_space_indexes, other.exec_space_indexes);
    std::swap(max_active_tasks, other.max_active_tasks);
    std::swap(grid_size, other.grid_size);
    std::swap(grid_scale, other.grid_scale);
    std::swap(grid, other.grid);
    std::swap(cf, other.cf);
    std::swap(weights, other.weights);
    std::swap(implementation_versions, other.implementation_versions);
  }

  void
  init_exec_spaces() {
    streams.resize(max_active_tasks);
    exec_spaces.reserve(max_active_tasks);
    for (unsigned i = 0; i < max_active_tasks; ++i) {
      if constexpr (!std::is_void_v<stream_type>) {
        auto rc = DeviceT<D>::create_stream(streams[i]);
        assert(rc);
        exec_spaces.push_back(execution_space(streams[i]));
        exec_space_indexes.push_back(i);
      } else {
        exec_spaces.push_back(execution_space());
        exec_space_indexes.push_back(i);
      }
    }
    current = StreamPhase::COPY;
  }

  execution_space&
  next_exec_space(StreamPhase next) {
    if (max_active_tasks > 1) {
      if (current == StreamPhase::COMPUTE && next == StreamPhase::COPY) {
        exec_space_indexes.push_back(exec_space_indexes.front());
        exec_space_indexes.pop_front();
        exec_spaces[exec_space_indexes.front()].fence();
      } else if (current == StreamPhase::COPY && next == StreamPhase::COMPUTE) {
        exec_spaces[exec_space_indexes[1]].fence();
      }
    }
#ifndef NDEBUG
    std::cout << current << "->"
              << next << ": "
              << exec_space_indexes.front() << std::endl;
#endif // NDEBUG
    current = next;
    return exec_spaces[exec_space_indexes.front()];
  }

  void
  new_grid(std::variant<const StateT*, bool> source, bool also_weights) {

    const bool create_without_init =
      std::holds_alternative<const StateT*>(source) || !std::get<bool>(source);

    // in the following, we don't call next_exec_space() except when a stream is
    // required, as there are code paths that never use a stream, and thus we
    // can avoid unnecessary stream switches
    std::array<int, 4> ig{
      static_cast<int>(grid_size[0]),
      static_cast<int>(grid_size[1]),
      static_cast<int>(grid_size[2]),
      static_cast<int>(grid_size[3])};
    if (create_without_init)
      grid =
        decltype(grid)(
          K::ViewAllocateWithoutInitializing("grid"),
          GridLayout<D>::dimensions(ig));
    else
      grid =
        decltype(grid)(
          K::view_alloc("grid", next_exec_space(StreamPhase::COPY)),
          GridLayout<D>::dimensions(ig));
#ifndef NDEBUG
    std::cout << "alloc grid sz " << grid.extent(0)
              << " " << grid.extent(1)
              << " " << grid.extent(2)
              << " " << grid.extent(3)
              << std::endl;
    std::cout << "alloc grid str " << grid.stride(0)
              << " " << grid.stride(1)
              << " " << grid.stride(2)
              << " " << grid.stride(3)
              << std::endl;
#endif // NDEBUG

    if (also_weights) {
      if (create_without_init)
        weights =
          decltype(weights)(
            K::ViewAllocateWithoutInitializing("weights"),
            static_cast<int>(grid_size[2]),
            static_cast<int>(grid_size[3]));
      else
        weights =
          decltype(weights)(
            K::view_alloc("weights", next_exec_space(StreamPhase::COPY)),
            static_cast<int>(grid_size[2]),
            static_cast<int>(grid_size[3]));
    }
    if (std::holds_alternative<const StateT*>(source)) {
      auto exec = next_exec_space(StreamPhase::COPY);
      auto st = std::get<const StateT*>(source);
      K::deep_copy(exec, grid, st->grid);
      if (also_weights)
        K::deep_copy(exec, weights, st->weights);
    }
  }
};

} // end namespace Impl
} // end namespace hpg

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
