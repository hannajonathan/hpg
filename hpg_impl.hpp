#include "hpg.hpp"

#include <algorithm>
#include <cassert>
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

struct Visibility {

  KOKKOS_INLINE_FUNCTION Visibility() {};

  Visibility(
    const vis_t& value_,
    const grid_plane_t& grid_plane,
    unsigned cf_cube_,
    vis_weight_fp weight_,
    vis_frequency_fp freq_,
    vis_phase_fp d_phase_,
    const vis_uvw_t& uvw_)
    : value(value_)
    , grid_stokes(std::get<0>(grid_plane))
    , grid_cube(std::get<1>(grid_plane))
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
  int grid_stokes;
  int grid_cube;
  int cf_cube;
  vis_weight_fp weight;
  vis_frequency_fp freq;
  vis_phase_fp d_phase;
  K::Array<vis_uvw_fp, 3> uvw;
};

static bool hpg_impl_initialized = false;

/** implementation initialization function */
void
initialize() {
  K::initialize();
#ifdef HPG_ENABLE_OPENMP
  auto rc = fftw_init_threads();
  assert(rc != 0);
#endif
  hpg_impl_initialized = true;
}

/** implementation finalization function */
void
finalize() {
  K::finalize();
}

/** */
bool
is_initialized() {
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
static const std::array<int, 4> strided_grid_layout_order{2, 1, 0, 3};

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

/** device-specific CF layout */
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
   * logical index order: X, Y, polarization, x, y, cube
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
      static const std::array<int, 6> order{2, 1, 0, 4, 3, 5};
      return K::LayoutStride::order_dimensions(6, order.data(), dims.data());
    }
  }
};

/**
 * convert a UV coordinate to coarse and fine grid coordinates
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
  long nearest_coarse_fine_coord =
    std::lrint(double(fine_coord) / oversampling) * oversampling;
  int coarse = nearest_coarse_fine_coord / oversampling - cf_radius + g_offset;
  int fine;
  if (fine_coord >= nearest_coarse_fine_coord)
    fine = fine_coord - nearest_coarse_fine_coord;
  else
    fine = fine_coord - (nearest_coarse_fine_coord - oversampling);
  assert(0 <= fine && fine < oversampling);
  return {coarse, fine};
}

/**
 * portable sincos()
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
sincos<K::CudaSpace, float>(float ph, float* sn, float* cs) {
  ::sincosf(ph, sn, cs);
}
template <>
__device__ __forceinline__ void
sincos<K::CudaSpace, double>(double ph, double* sn, double* cs) {
  ::sincos(ph, sn, cs);
}
#endif

/**
 * helper class for computing visibility value and index metadata
 */
template <typename execution_space>
struct GridVis final {

  int coarse[2];
  int fine[2];
  vis_t value;
  int grid_stokes;
  int grid_cube;
  int cf_cube;
  vis_weight_fp weight;
  cf_fp cf_im_factor;

  KOKKOS_INLINE_FUNCTION GridVis() {};

  KOKKOS_INLINE_FUNCTION GridVis(
    const Visibility& vis,
    const K::Array<int, 2>& grid_radius,
    const K::Array<int, 2>& oversampling,
    const K::Array<int, 2>& cf_radius,
    const K::Array<grid_scale_fp, 2>& fine_scale)
    : grid_stokes(vis.grid_stokes)
    , grid_cube(vis.grid_cube)
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
    coarse[0] = c0;
    fine[0] = f0;
    auto [c1, f1] =
      compute_vis_coord(
        grid_radius[1],
        oversampling[1],
        cf_radius[1],
        vis.uvw[1],
        inv_lambda,
        fine_scale[1]);
    coarse[1] = c1;
    fine[1] = f1;
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
pseudo_atomic_add(K::complex<T>& acc, const K::complex<T>& val) {
  K::atomic_add(&acc, val);
}

#ifdef HPG_ENABLE_CUDA
template <>
KOKKOS_FORCEINLINE_FUNCTION void
pseudo_atomic_add<K::Cuda, double>(
  K::complex<double>& acc, const K::complex<double>& val) {

  K::atomic_add(&acc.real(), val.real());
  K::atomic_add(&acc.imag(), val.imag());
}

template <>
KOKKOS_FORCEINLINE_FUNCTION void
pseudo_atomic_add<K::Cuda, float>(
  K::complex<float>& acc, const K::complex<float>& val) {

  K::atomic_add(&acc.real(), val.real());
  K::atomic_add(&acc.imag(), val.imag());
}
#endif // HPG_ENABLE_CUDA
#ifdef HPG_ENABLE_HPX
template <>
KOKKOS_FORCEINLINE_FUNCTION void
pseudo_atomic_add<K::HPX, double>(
  K::complex<double>& acc, const K::complex<double>& val) {

  K::atomic_add(&acc.real(), val.real());
  K::atomic_add(&acc.imag(), val.imag());
}

template <>
KOKKOS_FORCEINLINE_FUNCTION void
pseudo_atomic_add<K::HPX, float>(
  K::complex<float>& acc, const K::complex<float>& val) {

  K::atomic_add(&acc.real(), val.real());
  K::atomic_add(&acc.imag(), val.imag());
}
#endif // HPG_ENABLE_HPX

namespace Core {

// we're wrapping each kernel in a class in order to support partial
// specialization of the kernel functions by execution space

template <typename execution_space>
struct HPG_EXPORT VisibilityGridder final {

  /** gridding kernel
   *
   * Note that the default implementation probably is optimal only for many-core
   * devices, probably not for OpenMP.
   */
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

    K::parallel_for(
      "gridding",
      K::TeamPolicy<execution_space>(
        exec,
        static_cast<int>(visibilities.size()),
        K::AUTO),
      KOKKOS_LAMBDA(const member_type& team_member) {
        GridVis<execution_space> vis(
          visibilities(team_member.league_rank()),
          grid_radius,
          oversampling,
          cf_radius,
          fine_scale);
        auto wgt = &weights(vis.grid_stokes, vis.grid_cube);
        /* loop over coarseX */
        K::parallel_for(
          K::TeamVectorRange(team_member, cf.extent_int(0)),
          [=](const int X) {
            /* loop over coarseY */
            gv_t n;
            for (int Y = 0; Y < cf.extent_int(1); ++Y) {
              /* loop over elements of Mueller matrix column  */
              for (int P = 0; P < cf.extent_int(2); ++P) {
                gv_t cfv = cf(X, Y, P, vis.fine[0], vis.fine[1], vis.cf_cube);
                cfv.imag() *= vis.cf_im_factor;
                pseudo_atomic_add<execution_space>(
                  grid(
                    vis.coarse[0] + X,
                    vis.coarse[1] + Y,
                    vis.grid_stokes,
                    vis.grid_cube),
                  cfv * vis.value);
                n += cfv;
              }
            }
            K::atomic_add(wgt, std::hypot(n.real(), n.imag()) * vis.weight);
          });
      });
  }
};

template <typename execution_space>
struct HPG_EXPORT GridNormalizer final {

  /** grid normalization kernel
   */
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

template <typename execution_space>
struct HPG_EXPORT FFT final {

  // default implementation assumes FFTW3

  template <typename IG, typename OG>
  static auto
  grid_fft_handle(execution_space exec, IG& igrid, OG& ogrid) {

    using scalar_t = typename OG::value_type::value_type;

#ifdef HPG_ENABLE_OPENMP
    if constexpr (std::is_same_v<execution_space, K::Serial>)
      FFTW<scalar_t>::plan_with_nthreads(1);
    else
      FFTW<scalar_t>::plan_with_nthreads(omp_get_max_threads());
#endif // HPG_ENABLE_OPENMP

    {
      std::set<std::tuple<size_t, int>> order;
      for (size_t d = 0; d < 4; ++d)
        order.emplace(igrid.layout().stride[d], d);
      assert(
        std::equal(
          order.begin(),
          order.end(),
          strided_grid_layout_order.begin(),
          [](const auto& s_d, const auto& i) {
            return std::get<1>(s_d) == i;
          }));
    }
    // this assumes there is no padding in grid
    assert(igrid.span() ==
           igrid.extent(0) * igrid.extent(1)
           * igrid.extent(2) * igrid.extent(3));
    int n[2]{igrid.extent_int(0), igrid.extent_int(1)};
    int stride = igrid.extent_int(2);
    int dist = igrid.extent_int(0) *igrid.extent_int(1) * igrid.extent_int(3);
    auto result =
      FFTW<scalar_t>::plan_many(
        2, n, igrid.extent_int(3),
        const_cast<K::complex<scalar_t>*>(&igrid(0, 0, 0, 0)),
        NULL, stride, dist,
        &ogrid(0, 0, 0, 0), NULL, stride, dist,
        FFTW_FORWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    return result;
  }
  /** in-place FFT kernel
   */
  template <typename grid_layout, typename memory_space>
  static void
  in_place_kernel(
    execution_space exec,
    const grid_view<grid_layout, memory_space>& grid) {

    using scalar_t =
      typename grid_view<grid_layout, memory_space>::value_type::value_type;

    auto handles = grid_fft_handle(exec, grid, grid);
    auto& [h0, h1] = handles;
    for (int sto = 0; sto < grid.extent_int(2); ++sto) {
      FFTW<scalar_t>::exec(h0, &grid(0, 0, sto, 0), &grid(0, 0, sto, 0));
      std::swap(h0, h1);
    }
    FFTW<scalar_t>::destroy_plan(handles);
  }

  /** out-of-place FFT kernel
   */
  template <typename grid_layout, typename memory_space>
  static void
  out_of_place_kernel(
    execution_space exec,
    const const_grid_view<grid_layout, memory_space>& pre_grid,
    const grid_view<grid_layout, memory_space>& post_grid) {

    using scalar_t =
      typename grid_view<grid_layout, memory_space>::value_type::value_type;

    auto handles = grid_fft_handle(exec, pre_grid, post_grid);
    auto& [h0, h1] = handles;
    for (int sto = 0; sto < pre_grid.extent_int(2); ++sto) {
      FFTW<scalar_t>::exec(
        h0,
        const_cast<K::complex<scalar_t>*>(&pre_grid(0, 0, sto, 0)),
        &post_grid(0, 0, sto, 0));
      std::swap(h0, h1);
    }
    FFTW<scalar_t>::destroy_plan(handles);
  }
};

#ifdef HPG_ENABLE_CUDA

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
    K::complex<double>* idata,
    K::complex<double>* odata,
    int direction) {
    return
      cufftExecZ2Z(
        plan,
        reinterpret_cast<cufftDoubleComplex*>(idata),
        reinterpret_cast<cufftDoubleComplex*>(odata),
        direction);
  }
};

template <>
struct CUFFT<float> {

  static constexpr cufftType type = CUFFT_C2C;

  static cufftResult
  exec(
    cufftHandle plan,
    K::complex<float>* idata,
    K::complex<float>* odata,
    int direction) {
    return
      cufftExecC2C(
        plan,
        reinterpret_cast<cufftComplex*>(idata),
        reinterpret_cast<cufftComplex*>(odata),
        direction);
  }
};

template <>
struct HPG_EXPORT FFT<K::Cuda> final {

  template <typename G>
  static cufftHandle
  grid_fft_handle(K::Cuda exec, G& grid) {

    using scalar_t = typename G::value_type::value_type;

    // this assumes there is no padding in grid
    assert(grid.span() ==
           grid.extent(0) * grid.extent(1) * grid.extent(2) * grid.extent(3));
    int n[2]{grid.extent_int(0), grid.extent_int(1)};
    cufftHandle result;
    auto rc =
      cufftPlanMany(
        &result, 2, n,
        NULL, 1, 1,
        NULL, 1, 1,
        CUFFT<scalar_t>::type,
        grid.extent_int(2) * grid.extent_int(3));
    assert(rc == CUFFT_SUCCESS);
    rc = cufftSetStream(result, exec.cuda_stream());
    assert(rc == CUFFT_SUCCESS);
    return result;
  }

  /** in-place FFT kernel
   */
  template <typename grid_layout, typename memory_space>
  static void
  in_place_kernel(
    K::Cuda exec,
    const grid_view<grid_layout, memory_space>& grid) {

    using scalar_t =
      typename grid_view<grid_layout, memory_space>::value_type::value_type;

    auto handle = grid_fft_handle(exec, grid);
    auto rc =
      CUFFT<scalar_t>::exec(
        handle,
        &grid(0, 0, 0, 0),
        &grid(0, 0, 0, 0),
        CUFFT_FORWARD);
    assert(rc == CUFFT_SUCCESS);
    rc = cufftDestroy(handle);
    assert(rc == CUFFT_SUCCESS);
  }

  /** out-of-place FFT kernel
   */
  template <typename grid_layout, typename memory_space>
  static void
  out_of_place_kernel(
    K::Cuda exec,
    const const_grid_view<grid_layout, memory_space>& pre_grid,
    const grid_view<grid_layout, memory_space>& post_grid) {

    using scalar_t =
      typename grid_view<grid_layout, memory_space>::value_type::value_type;

    auto handle = grid_fft_handle(exec, post_grid);
    auto rc =
      CUFFT<scalar_t>::exec(
        handle,
        const_cast<K::complex<scalar_t>*>(&pre_grid(0, 0, 0, 0)),
        &post_grid(0, 0, 0, 0),
        CUFFT_FORWARD);
    assert(rc == CUFFT_SUCCESS);
    rc = cufftDestroy(handle);
    assert(rc == CUFFT_SUCCESS);
  }
};
#endif // HPG_ENABLE_CUDA

} // end namespace Core

/** abstract base class for state implementations */
struct State {

  Device device; /**< device type */
  unsigned max_active_tasks; /**< maximum number of active tasks */
  std::array<unsigned, 4> grid_size; /**< grid size */
  std::array<grid_scale_fp, 2> grid_scale; /**< grid scale */

  State(Device device_)
    : device(device_) {}

  State(
    Device device_,
    unsigned max_active_tasks_,
    const std::array<unsigned, 4>& grid_size_,
    const std::array<grid_scale_fp, 2>& grid_scale_)
    : device(device_)
    , max_active_tasks(max_active_tasks_)
    , grid_size(grid_size_)
    , grid_scale(grid_scale_) {}

  virtual void
  set_convolution_function(Device host_device, const CFArray& cf) = 0;

  virtual void
  grid_visibilities(
    Device host_device,
    const std::vector<std::complex<visibility_fp>>& visibilities,
    const std::vector<grid_plane_t> visibility_grid_planes,
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

  virtual void
  apply_fft(bool in_place) = 0;

  virtual ~State() {}
};

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

template <Device D, typename CFH>
static void
init_cf_host(CFH& cf_h, const CFArray& cf) {
  static_assert(
    K::SpaceAccessibility<
      typename DeviceT<D>::kokkos_device::memory_space,
      K::HostSpace>
    ::accessible);

  K::parallel_for(
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

template <Device D, typename VisH>
static void
init_vis(
  VisH& vis_h,
  const std::vector<std::complex<visibility_fp>>& visibilities,
  const std::vector<grid_plane_t> visibility_grid_planes,
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
    K::RangePolicy<typename DeviceT<D>::kokkos_device>(
      0,
      static_cast<int>(visibilities.size())),
    KOKKOS_LAMBDA(int i) {
      vis_h(i) =
        Visibility(
          visibilities[i],
          visibility_grid_planes[i],
          visibility_cf_cubes[i],
          visibility_weights[i],
          visibility_frequencies[i],
          visibility_phases[i],
          visibility_coordinates[i]);
    });
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
  std::deque<int> exec_space_indexes;

  StateT(
    unsigned max_active_tasks,
    const std::array<unsigned, 4> grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale)
    : State(
      D,
      std::min(max_active_tasks, DeviceT<D>::active_task_limit),
      grid_size,
      grid_scale) {

    init_exec_spaces();
    new_grid(true, true);
  }

  StateT(const volatile StateT& st)
    : State(
      D,
      const_cast<const StateT&>(st).max_active_tasks,
      const_cast<const StateT&>(st).grid_size,
      const_cast<const StateT&>(st).grid_scale) {

    init_exec_spaces();
    new_grid(const_cast<const StateT*>(&st), true);
  }

  StateT(StateT&& st)
    : State(D) {
    swap(st);
  }

  virtual ~StateT() {
    fence();
    grid = decltype(grid)();
    cf = decltype(cf)();
    weights = decltype(weights)();
    if constexpr(!std::is_void_v<stream_type>) {
      for (unsigned i = 0; i < max_active_tasks; ++i) {
        auto rc = DeviceT<D>::destroy_stream(streams[i]);
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

  void
  set_convolution_function(Device host_device, const CFArray& cf_array)
    override {
    cf_view<typename CFLayout<D>::layout, memory_space> cf_init(
      K::ViewAllocateWithoutInitializing("cf"),
      CFLayout<D>::dimensions(cf_array));
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

    switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
    case Device::Serial: {
      auto cf_h = K::create_mirror_view(cf_init);
      init_cf_host<Device::Serial>(cf_h, cf_array);
      K::deep_copy(current_exec_space(), cf_init, cf_h);
      break;
    }
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
    case Device::OpenMP: {
      auto cf_h = K::create_mirror_view(cf_init);
      init_cf_host<Device::OpenMP>(cf_h, cf_array);
      K::deep_copy(current_exec_space(), cf_init, cf_h);
      break;
    }
#endif // HPG_ENABLE_SERIAL
    default:
      assert(false);
      break;
    }
    cf = cf_init;
  }

  void
  grid_visibilities(
    Device host_device,
    const std::vector<std::complex<visibility_fp>>& visibilities,
    const std::vector<grid_plane_t> visibility_grid_planes,
    const std::vector<unsigned> visibility_cf_cubes,
    const std::vector<vis_weight_fp>& visibility_weights,
    const std::vector<vis_frequency_fp>& visibility_frequencies,
    const std::vector<vis_phase_fp>& visibility_phases,
    const std::vector<vis_uvw_t>& visibility_coordinates) override {

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
        visibility_grid_planes,
        visibility_cf_cubes,
        visibility_weights,
        visibility_frequencies,
        visibility_phases,
        visibility_coordinates);
      K::deep_copy(current_exec_space(), vis, vis_h);
      break;
    }
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
    case Device::OpenMP: {
      auto vis_h = K::create_mirror_view(vis);
      init_vis<Device::OpenMP>(
        vis_h,
        visibilities,
        visibility_grid_planes,
        visibility_cf_cubes,
        visibility_weights,
        visibility_frequencies,
        visibility_phases,
        visibility_coordinates);
      K::deep_copy(current_exec_space(), vis, vis_h);
      break;
    }
#endif // HPG_ENABLE_SERIAL
    default:
      assert(false);
      break;
    }
    const_visibility_view<memory_space> cvis = vis;
    Core::VisibilityGridder<execution_space>::kernel(
      current_exec_space(),
      cf,
      cvis,
      grid_scale,
      grid,
      weights);
    next_exec_space();
  }

  void
  fence() const volatile override {
    auto st = const_cast<StateT*>(this);
    if (st->exec_space_indexes.front() == -1)
      st->exec_space_indexes.pop_front();
    for (unsigned i = 0; i < st->max_active_tasks; ++i) {
      st->exec_spaces[st->exec_space_indexes.front()].fence();
      st->exec_space_indexes.push_back(st->exec_space_indexes.front());
      st->exec_space_indexes.pop_front();
    }
  }

  std::unique_ptr<GridWeightArray>
  grid_weights() const volatile override {
    fence();
    auto st = const_cast<StateT*>(this);
    auto wgts_h = K::create_mirror(st->weights);
    K::deep_copy(st->current_exec_space(), wgts_h, st->weights);
    st->current_exec_space().fence();
    return std::make_unique<GridWeightViewArray<D>>(wgts_h);
  }

  std::unique_ptr<GridValueArray>
  grid_values() const volatile override {
    fence();
    auto st = const_cast<StateT*>(this);
    auto grid_h = K::create_mirror(st->grid);
    K::deep_copy(st->current_exec_space(), grid_h, st->grid);
    st->current_exec_space().fence();
    return std::make_unique<GridValueViewArray<D>>(grid_h);
  }

  void
  reset_grid() override {
    next_exec_space();
    new_grid(true, true);
  }

  void
  normalize(grid_value_fp wfactor) override {
    const_weight_view<typename execution_space::array_layout, memory_space>
      cweights = weights;
    Core::GridNormalizer<execution_space>
      ::kernel(current_exec_space(), grid, cweights, wfactor);
    next_exec_space();
  }

  void
  apply_fft(bool in_place) override {
    if (in_place) {
      Core::FFT<execution_space>
        ::in_place_kernel(current_exec_space(), grid);
    } else {
      const_grid_view<typename GridLayout<D>::layout, memory_space> pre_grid
        = grid;
      new_grid(false, false);
      Core::FFT<execution_space>
        ::out_of_place_kernel(current_exec_space(), pre_grid, grid);
    }
    next_exec_space();
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
    std::swap(weights, other.weights);
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
  }

  execution_space&
  current_exec_space() {
    if (exec_space_indexes.front() == -1) {
      exec_space_indexes.pop_front();
      assert(exec_space_indexes.front() != -1);
      exec_spaces[exec_space_indexes.front()].fence();
    }
    return exec_spaces[exec_space_indexes.front()];
  }

  void
  next_exec_space() {
    if (exec_space_indexes.front() != -1) {
      exec_space_indexes.push_back(exec_space_indexes.front());
      exec_space_indexes.pop_front();
      exec_space_indexes.push_front(-1);
    }
  }

  void
  new_grid(std::variant<const StateT*, bool> source, bool also_weights) {

    const bool create_without_init =
      std::holds_alternative<const StateT*>(source) || !std::get<bool>(source);

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
          K::view_alloc("grid", current_exec_space()),
          GridLayout<D>::dimensions(ig));
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
            K::view_alloc("weights", current_exec_space()),
            static_cast<int>(grid_size[2]),
            static_cast<int>(grid_size[3]));
    }
    if (std::holds_alternative<const StateT*>(source)) {
      auto st = std::get<const StateT*>(source);
      K::deep_copy(current_exec_space(), grid, st->grid);
      if (also_weights)
        K::deep_copy(current_exec_space(), weights, st->weights);
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
