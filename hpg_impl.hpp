#include "hpg.hpp"

#include <algorithm>
#include <cassert>
#include <type_traits>

#include <iostream> // FIXME: remove

#include <Kokkos_Core.hpp>

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
    vis_weight_fp weight_,
    vis_frequency_fp freq_,
    vis_phase_fp d_phase_,
    const vis_uvw_t& uvw_)
    : value(value_)
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
  vis_weight_fp weight;
  vis_frequency_fp freq;
  vis_phase_fp d_phase;
  K::Array<vis_uvw_fp, 3> uvw;
};

/** implementation initialization function */
void
initialize() {
  K::initialize();
}

/** implementation finalization function */
void
finalize() {
  K::finalize();
}

/** type trait associating Kokkos device with hpg Device */
template <Device D>
struct DeviceT {
  //using kokkos_device;
};

#ifdef HPG_ENABLE_SERIAL
template <>
struct DeviceT<Device::Serial> {
  using kokkos_device = K::Serial;
};
#endif // HPG_ENABLE_SERIAL

#ifdef HPG_ENABLE_OPENMP
template <>
struct DeviceT<Device::OpenMP> {
  using kokkos_device = K::OpenMP;
};
#endif // HPG_ENABLE_OPENMP

#ifdef HPG_ENABLE_CUDA
template <>
struct DeviceT<Device::Cuda> {
  using kokkos_device = K::Cuda;
};
#endif // HPG_ENABLE_CUDA

#ifdef HPG_ENABLE_HPX
template <>
struct DeviceT<Device::HPX> {
  using kokkos_device = K::HPX;
};
#endif // HPG_ENABLE_HPX

/** Kokkos::View type for grid values */
template <typename Layout, typename memory_space>
using grid_view = K::View<gv_t***, Layout, memory_space>;

template <typename Layout, typename memory_space>
using const_grid_view = K::View<const gv_t***, Layout, memory_space>;

/** View type for CF2 values */
template <typename Layout, typename memory_space>
using cf2_view = K::View<cf_t****, Layout, memory_space>;

template <typename Layout, typename memory_space>
using const_cf2_view = K::View<const cf_t****, Layout, memory_space>;

/** View type for Visibility values */
template <typename memory_space>
using visibility_view = K::View<Visibility*, memory_space>;

template <typename memory_space>
using const_visibility_view = K::View<const Visibility*, memory_space>;

/** device-specific grid layout */
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
   */
  static layout
  dimensions(const std::array<int, 3>& dims) {
    if constexpr (std::is_same_v<layout, K::LayoutLeft>) {
      return K::LayoutLeft(dims[0], dims[1], dims[2]);
    } else {
      static const std::array<int, 3> order{1, 0, 2};
      return K::LayoutStride::order_dimensions(3, order.data(), dims.data());
    }
  }
};

/** device-specific CF2 layout */
template <Device D>
struct CF2Layout {

  /** Kokkos layout type */
  using layout =
    std::conditional_t<
    std::is_same_v<
      typename DeviceT<D>::kokkos_device::array_layout,
      K::LayoutLeft>,
    K::LayoutLeft,
    K::LayoutStride>;

  /**
   * create Kokkos layout using given CF2
   *
   * logical layout is X, Y, x, y
   */
  static layout
  dimensions(const CF2& cf) {
    std::array<int, 4> dims{
      static_cast<int>(cf.extent[0] / cf.oversampling),
      static_cast<int>(cf.extent[1] / cf.oversampling),
      static_cast<int>(cf.oversampling),
      static_cast<int>(cf.oversampling)
    };
    if constexpr (std::is_same_v<layout, K::LayoutLeft>) {
      return K::LayoutLeft(dims[0], dims[1], dims[2], dims[3]);
    } else {
      static const std::array<int, 4> order{1, 0, 3, 2};
      return K::LayoutStride::order_dimensions(4, order.data(), dims.data());
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
  cf_fp cf_im_factor;

  KOKKOS_INLINE_FUNCTION GridVis() {};

  KOKKOS_INLINE_FUNCTION GridVis(
    const Visibility& vis,
    const K::Array<int, 2>& grid_radius,
    const K::Array<int, 2>& oversampling,
    const K::Array<int, 2>& cf_radius,
    const K::Array<grid_scale_fp, 2>& fine_scale) {

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

/** gridding kernel
 *
 * The function that launches the gridding kernel is wrapped by a class to allow
 * partial specialization by execution space. Note that the default
 * implementation probably is optimal only for many-core devices, probably not
 * for OpenMP. A specialization for the host Serial device is provided, however.
 */
template <typename execution_space>
struct HPG_EXPORT Gridder final {

  template <typename cf_layout, typename grid_layout, typename memory_space>
  static void
  grid_visibilities(
    execution_space exec,
    const const_cf2_view<cf_layout, memory_space>& cf,
    const const_visibility_view<memory_space>& visibilities,
    const std::array<grid_scale_fp, 2>& grid_scale,
    const grid_view<grid_layout, memory_space>& grid,
    const K::View<gv_t, memory_space>& norm) {

    const K::Array<int, 2>
      cf_radius{cf.extent_int(0) / 2, cf.extent_int(1) / 2};
    const K::Array<int, 2>
      grid_radius{grid.extent_int(0) / 2, grid.extent_int(1) / 2};
    const K::Array<int, 2>
      oversampling{cf.extent_int(2), cf.extent_int(3)};
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
        /* loop over coarseX */
        K::parallel_for(
          K::TeamVectorRange(team_member, cf.extent_int(0)),
          [=](const int X) {
            /* loop over coarseY */
            gv_t n;
            for (int Y = 0; Y < cf.extent_int(1); ++Y) {
              gv_t cfv = cf(X, Y, vis.fine[0], vis.fine[1]);
              cfv.imag() *= vis.cf_im_factor;
              pseudo_atomic_add<execution_space>(
                grid(vis.coarse[0] + X, vis.coarse[1] + Y, 0),
                cfv * vis.value);
              n += cfv;
            }
            pseudo_atomic_add<execution_space>(norm(), n);
          });
      });
  }
};

/** grid_visibilities() specialization for K::Serial
 *
 * only difference is that atomic addition to grid isn't used
 */
#ifdef HPG_ENABLE_SERIAL
template <>
struct HPG_EXPORT Gridder<K::Serial> final {

  template <typename cf_layout, typename grid_layout, typename memory_space>
  static void
  grid_visibilities(
    K::Serial /* unused */,
    const const_cf2_view<cf_layout, memory_space>& cf,
    const const_visibility_view<memory_space>& visibilities,
    const std::array<grid_scale_fp, 2>& grid_scale,
    const grid_view<grid_layout, memory_space>& grid,
    const K::View<gv_t, memory_space>& norm) {

    const K::Array<int, 2>
      cf_radius{cf.extent_int(0) / 2, cf.extent_int(1) / 2};
    const K::Array<int, 2>
      grid_radius{grid.extent_int(0) / 2, grid.extent_int(1) / 2};
    const K::Array<int, 2>
      oversampling{cf.extent_int(2), cf.extent_int(3)};
    const K::Array<grid_scale_fp, 2> fine_scale{
      grid_scale[0] * oversampling[0],
      grid_scale[1] * oversampling[1]};

    using member_type = typename K::TeamPolicy<K::Serial>::member_type;

    K::parallel_for(
      "gridding",
      K::TeamPolicy<K::Serial>(static_cast<int>(visibilities.size()), K::AUTO),
      KOKKOS_LAMBDA(const member_type& team_member) {
        GridVis<K::Serial> vis(
          visibilities(team_member.league_rank()),
          grid_radius,
          oversampling,
          cf_radius,
          fine_scale);
        /* loop over coarseX */
        K::parallel_for(
          K::TeamVectorRange(team_member, cf.extent_int(0)),
          [=](const int X) {
            /* loop over coarseY */
            for (int Y = 0; Y < cf.extent_int(1); ++Y) {
              gv_t cfv = cf(X, Y, vis.fine[0], vis.fine[1]);
              cfv.imag() *= vis.cf_im_factor;
              grid(vis.coarse[0] + X, vis.coarse[1] + Y, 0) += cfv * vis.value;
              norm() += cfv;
            }
          });
      });
  }
};
#endif // HPG_ENABLE_SERIAL
} // end namespace Core

/** abstract base class for state implementations */
struct State {

  Device device; /**< device type */
  std::array<unsigned, 3> grid_size; /**< grid size */
  std::array<grid_scale_fp, 2> grid_scale; /**< grid scale */

  State(
    Device device_,
    const std::array<unsigned, 3>& grid_size_,
    const std::array<grid_scale_fp, 2>& grid_scale_)
    : device(device_)
    , grid_size(grid_size_)
    , grid_scale(grid_scale_) {}

  virtual void
  set_convolution_function(Device host_device, const CF2& cf) = 0;

  virtual void
  grid_visibilities(
    Device host_device,
    const std::vector<std::complex<visibility_fp>>& visibilities,
    const std::vector<vis_weight_fp>& visibility_weights,
    const std::vector<vis_frequency_fp>& visibility_frequencies,
    const std::vector<vis_phase_fp>& visibility_phase,
    const std::vector<vis_uvw_t>& visibility_coordinates) = 0;

  virtual void
  fence() const volatile = 0;

  virtual std::complex<grid_value_fp>
  get_normalization() const volatile = 0;

  virtual std::complex<grid_value_fp>
  set_normalization(const std::complex<grid_value_fp>&) = 0;

  virtual ~State() {}
};

template <Device D, typename CFH>
static void
init_cf_host(CFH& cf_h, const CF2& cf2) {
  static_assert(
    K::SpaceAccessibility<
      typename DeviceT<D>::kokkos_device::memory_space,
      K::HostSpace>
    ::accessible);
  K::parallel_for(
    K::MDRangePolicy<
    K::Rank<2>,
    typename DeviceT<D>::kokkos_device>(
      {0, 0},
      {static_cast<int>(cf2.extent[0]), static_cast<int>(cf2.extent[1])}),
    [&](int i, int j) {
      auto X = i / cf2.oversampling;
      auto x = i % cf2.oversampling;
      auto Y = j / cf2.oversampling;
      auto y = j % cf2.oversampling;
      cf_h(X, Y, x, y) = cf2(i, j);
    });
}

template <Device D, typename VisH>
static void
init_vis(
  VisH& vis_h,
  const std::vector<std::complex<visibility_fp>>& visibilities,
  const std::vector<vis_weight_fp>& visibility_weights,
  const std::vector<vis_frequency_fp>& visibility_frequencies,
  const std::vector<vis_phase_fp>& visibility_phase,
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
          visibility_weights[i],
          visibility_frequencies[i],
          visibility_phase[i],
          visibility_coordinates[i]);
    });
}

/** Kokkos state implementation for a device type */
template <Device D>
class HPG_EXPORT StateT final
  : public State {
public:

  using kokkos_device = typename DeviceT<D>::kokkos_device;
  using execution_space = typename kokkos_device::execution_space;
  using memory_space = typename kokkos_device::memory_space;

  execution_space exec_space;

  grid_view<typename GridLayout<D>::layout, memory_space> grid;
  const_cf2_view<typename CF2Layout<D>::layout, memory_space> cf;
  K::View<gv_t, memory_space> norm;

  StateT()
    : State(D) {
  }

  StateT(
    const std::array<unsigned, 3> grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale)
    : State(D, grid_size,  grid_scale) {

    std::array<int, 3> ig{
      static_cast<int>(grid_size[0]),
      static_cast<int>(grid_size[1]),
      static_cast<int>(grid_size[2])};
    // NB: initialize grid here, since there's only one execution space instance
    grid = decltype(grid)("grid", GridLayout<D>::dimensions(ig));
    std::cout << "alloc grid sz " << grid.extent(0)
              << " " << grid.extent(1)
              << " " << grid.extent(2)
              << std::endl;
    std::cout << "alloc grid str " << grid.stride(0)
              << " " << grid.stride(1)
              << " " << grid.stride(2)
              << std::endl;
    norm = decltype(norm)("norm");
  }

  StateT(const volatile StateT& st)
    : State(
      D,
      const_cast<const StateT&>(st).grid_size,
      const_cast<const StateT&>(st).grid_scale) {

    const StateT& cst = const_cast<const StateT&>(st);
    std::array<int, 3> ig{
      static_cast<int>(cst.grid_size[0]),
      static_cast<int>(cst.grid_size[1]),
      static_cast<int>(cst.grid_size[2])};
    grid =
      decltype(grid)(
        K::ViewAllocateWithoutInitializing("grid"),
        GridLayout<D>::dimensions(ig));
    std::cout << "alloc grid sz " << grid.extent(0)
              << " " << grid.extent(1)
              << " " << grid.extent(2)
              << std::endl;
    std::cout << "alloc grid str " << grid.stride(0)
              << " " << grid.stride(1)
              << " " << grid.stride(2)
              << std::endl;
    st.fence();
    K::deep_copy(exec_space, grid, cst.grid);
    norm = decltype(norm)(K::ViewAllocateWithoutInitializing("norm"));
    K::deep_copy(exec_space, norm, cst.norm);
  }

  StateT(StateT&& st)
    : State(D, std::move(st).grid_size, std::move(st).grid_scale) {
    grid = st.grid;
    norm = st.norm;
  }

  StateT&
  operator=(const volatile StateT& st) {
    StateT tmp(st);
    this->swap(tmp);
    return *this;
  }

  StateT&
  operator=(StateT&& st) {
    StateT tmp(st);
    this->swap(tmp);
    return *this;
  }

  StateT
  copy() const volatile {
    return StateT(*this);
  }

  void
  set_convolution_function(Device host_device, const CF2& cf2) override {
    cf2_view<typename CF2Layout<D>::layout, memory_space> cf_init(
      K::ViewAllocateWithoutInitializing("cf2"),
      CF2Layout<D>::dimensions(cf2));
    std::cout << "alloc cf sz " << cf_init.extent(0)
              << " " << cf_init.extent(1)
              << " " << cf_init.extent(2)
              << " " << cf_init.extent(3)
              << std::endl;
    std::cout << "alloc cf str " << cf_init.stride(0)
              << " " << cf_init.stride(1)
              << " " << cf_init.stride(2)
              << " " << cf_init.stride(3)
              << std::endl;
    switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
    case Device::Serial: {
      auto cf_h = K::create_mirror_view(cf_init);
      init_cf_host<Device::Serial>(cf_h, cf2);
      K::deep_copy(exec_space, cf_init, cf_h);
      break;
    }
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
    case Device::OpenMP: {
      auto cf_h = K::create_mirror_view(cf_init);
      init_cf_host<Device::OpenMP>(cf_h, cf2);
      K::deep_copy(exec_space, cf_init, cf_h);
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
    const std::vector<vis_weight_fp>& visibility_weights,
    const std::vector<vis_frequency_fp>& visibility_frequencies,
    const std::vector<vis_phase_fp>& visibility_phase,
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
        visibility_weights,
        visibility_frequencies,
        visibility_phase,
        visibility_coordinates);
      K::deep_copy(exec_space, vis, vis_h);
      break;
    }
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
    case Device::OpenMP: {
      auto vis_h = K::create_mirror_view(vis);
      init_vis<Device::OpenMP>(
        vis_h,
        visibilities,
        visibility_weights,
        visibility_frequencies,
        visibility_phase,
        visibility_coordinates);
      K::deep_copy(exec_space, vis, vis_h);
      break;
    }
#endif // HPG_ENABLE_SERIAL
    default:
      assert(false);
      break;
    }
    const_visibility_view<memory_space> cvis = vis;
    Core::Gridder<execution_space>
      ::grid_visibilities(exec_space, cf, cvis, grid_scale, grid, norm);
  }

  void
  fence() const volatile override {
    const_cast<const execution_space&>(exec_space).fence();
  }

  std::complex<grid_value_fp>
  get_normalization() const volatile override {
    gv_t result;
    auto nvnorm = const_cast<StateT*>(this)->norm;
    K::parallel_reduce(
      K::RangePolicy<execution_space>(0, 1),
      KOKKOS_LAMBDA(int, gv_t& acc) {
        acc += nvnorm();
      },
      result);
    return result;
  }

  std::complex<grid_value_fp>
  set_normalization(const std::complex<grid_value_fp>& val) override {
    gv_t new_val = val;
    gv_t old_val;
    auto nvnorm = const_cast<StateT*>(this)->norm;
    K::parallel_reduce(
      K::RangePolicy<execution_space>(0, 1),
      KOKKOS_LAMBDA(int, gv_t& acc) {
        acc += nvnorm();
        nvnorm() = new_val;
      },
      old_val);
    return old_val;
  }

private:
  void
  swap(StateT& other) {
    std::swap(grid_size, other.grid_size);
    std::swap(grid_scale, other.grid_scale);
    decltype(grid) g = grid;
    grid = other.grid;
    other.grid = g;
    decltype(norm) n = norm;
    norm = other.norm;
    other.norm = n;
  }
};

#ifdef HPG_ENABLE_CUDA
/** Kokkos state implementation for Cuda device */
template <>
struct HPG_EXPORT StateT<Device::Cuda> final
  : public State {
public:

  using execution_space = K::Cuda;
  using memory_space = execution_space::memory_space;

  grid_view<GridLayout<Device::Cuda>::layout, memory_space> grid;
  const_cf2_view<CF2Layout<Device::Cuda>::layout, memory_space> cf;
  K::View<gv_t, memory_space> norm;

  // use two execution spaces to support overlap of data copying with
  // computation using CUDA streams
  std::array<cudaStream_t, 2> streams{NULL, NULL};
  std::array<execution_space, 2> exec_spaces;
  execution_space* exec_copy = &exec_spaces[0];
  execution_space* exec_compute = &exec_spaces[1];

  StateT(
    const std::array<unsigned, 3> grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale)
    : State(Device::Cuda, grid_size, grid_scale) {

    init_exec_spaces();

    std::array<int, 3> ig{
      static_cast<int>(grid_size[0]),
      static_cast<int>(grid_size[1]),
      static_cast<int>(grid_size[2])};
    grid = decltype(grid)(
      K::ViewAllocateWithoutInitializing("grid"),
      GridLayout<Device::Cuda>::dimensions(ig));
    std::cout << "alloc grid sz " << grid.extent(0)
              << " " << grid.extent(1)
              << " " << grid.extent(2)
              << std::endl;
    std::cout << "alloc grid str " << grid.stride(0)
              << " " << grid.stride(1)
              << " " << grid.stride(2)
              << std::endl;
    K::deep_copy(*exec_copy, grid, 0);
    norm = decltype(norm)(K::ViewAllocateWithoutInitializing("norm"));
    K::deep_copy(*exec_copy, norm, 0);
  }

  StateT(const volatile StateT& st)
    : State(
      Device::Cuda,
      const_cast<const StateT&>(st).grid_size,
      const_cast<const StateT&>(st).grid_scale) {

    st.fence();
    init_exec_spaces();

    const StateT& cst = const_cast<const StateT&>(st);
    std::array<int, 3> ig{
      static_cast<int>(cst.grid_size[0]),
      static_cast<int>(cst.grid_size[1]),
      static_cast<int>(cst.grid_size[2])};
    grid =
      decltype(grid)(
        K::ViewAllocateWithoutInitializing("grid"),
        GridLayout<Device::Cuda>::dimensions(ig));
    std::cout << "alloc grid sz " << grid.extent(0)
              << " " << grid.extent(1)
              << " " << grid.extent(2)
              << std::endl;
    std::cout << "alloc grid str " << grid.stride(0)
              << " " << grid.stride(1)
              << " " << grid.stride(2)
              << std::endl;
    K::deep_copy(*exec_copy, grid, cst.grid);
    norm = decltype(norm)(K::ViewAllocateWithoutInitializing("norm"));
    K::deep_copy(*exec_copy, norm, cst.norm);
  }

  StateT(StateT&& st)
    : State(Device::Cuda, std::move(st).grid_size, std::move(st).grid_scale) {
    grid = st.grid;
    norm = st.norm;
    std::swap(streams[0], std::move(st).streams[0]);
    std::swap(streams[1], std::move(st).streams[1]);
    *exec_copy = std::move(*std::move(st).exec_copy);
    *exec_compute = std::move(*std::move(st).exec_compute);
  }

  virtual ~StateT() {
    for (int i = 0; i < static_cast<int>(streams.size()); ++i)
      if (streams[i])
        cudaStreamDestroy(streams[i]);
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
  set_convolution_function(Device host_device, const CF2& cf2) override {
    cf2_view<CF2Layout<Device::Cuda>::layout, memory_space> cf_init(
      K::ViewAllocateWithoutInitializing("cf2"),
      CF2Layout<Device::Cuda>::dimensions(cf2));
    std::cout << "alloc cf sz " << cf_init.extent(0)
              << " " << cf_init.extent(1)
              << " " << cf_init.extent(2)
              << " " << cf_init.extent(3)
              << std::endl;
    std::cout << "alloc cf str " << cf_init.stride(0)
              << " " << cf_init.stride(1)
              << " " << cf_init.stride(2)
              << " " << cf_init.stride(3)
              << std::endl;
    switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
    case Device::Serial: {
      auto cf_h = K::create_mirror_view(cf_init);
      init_cf_host<Device::Serial>(cf_h, cf2);
      K::deep_copy(*exec_copy, cf_init, cf_h);
      break;
    }
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
    case Device::OpenMP: {
      auto cf_h = K::create_mirror_view(cf_init);
      init_cf_host<Device::OpenMP>(cf_h, cf2);
      K::deep_copy(*exec_copy, cf_init, cf_h);
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
    const std::vector<vis_weight_fp>& visibility_weights,
    const std::vector<vis_frequency_fp>& visibility_frequencies,
    const std::vector<vis_phase_fp>& visibility_phase,
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
        visibility_weights,
        visibility_frequencies,
        visibility_phase,
        visibility_coordinates);
      K::deep_copy(*exec_copy, vis, vis_h);
      break;
    }
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
    case Device::OpenMP: {
      auto vis_h = K::create_mirror_view(vis);
      init_vis<Device::OpenMP>(
        vis_h,
        visibilities,
        visibility_weights,
        visibility_frequencies,
        visibility_phase,
        visibility_coordinates);
      K::deep_copy(*exec_copy, vis, vis_h);
      break;
    }
#endif // HPG_ENABLE_SERIAL
    default:
      assert(false);
      break;
    }
    exec_compute->fence();
    std::swap(exec_copy, exec_compute);
    const_visibility_view<memory_space> cvis = vis;
    Core::Gridder<execution_space>
      ::grid_visibilities(*exec_compute, cf, cvis, grid_scale, grid, norm);
  }

  void
  fence() const volatile override {
    const_cast<const execution_space*>(exec_compute)->fence();
  }

  std::complex<grid_value_fp>
  get_normalization() const volatile override {
    gv_t result;
    auto nvnorm = const_cast<StateT*>(this)->norm;
    fence();
    K::parallel_reduce(
      K::RangePolicy<execution_space>(*exec_compute, 0, 1),
      KOKKOS_LAMBDA(int, gv_t& acc) {
        acc += nvnorm();
      },
      result);
    return result;
  }

  std::complex<grid_value_fp>
  set_normalization(const std::complex<grid_value_fp>& val) override {
    gv_t new_val = val;
    gv_t old_val;
    fence();
    auto nvnorm = const_cast<StateT*>(this)->norm;
    K::parallel_reduce(
      K::RangePolicy<execution_space>(0, 1),
      KOKKOS_LAMBDA(int, gv_t& acc) {
        acc += nvnorm();
        nvnorm() = new_val;
      },
      old_val);
    return old_val;
  }

private:
  void
  swap(StateT& other) {
    std::swap(streams, other.streams);
    {
      auto other_copy_off = std::distance(&other.exec_spaces[0], other.exec_copy);
      auto copy_off = std::distance(&exec_spaces[0], exec_copy);
      exec_copy = &exec_spaces[other_copy_off];
      other.exec_copy = &other.exec_spaces[copy_off];
    }
    {
      auto other_compute_off =
        std::distance(&other.exec_spaces[1], other.exec_compute);
      auto compute_off = std::distance(&exec_spaces[1], exec_compute);
      exec_compute = &exec_spaces[other_compute_off];
      other.exec_compute = &other.exec_spaces[compute_off];
    }
    std::swap(grid_size, other.grid_size);
    std::swap(grid_scale, other.grid_scale);
    decltype(grid) g = grid;
    grid = other.grid;
    other.grid = g;
    decltype(norm) n = norm;
    norm = other.norm;
    other.norm = n;
  }

  void
  init_exec_spaces() {
    for (int i = 0; i < static_cast<int>(exec_spaces.size()); ++i) {
      auto rc = cudaStreamCreate(&streams[i]);
      assert(rc == cudaSuccess);
      exec_spaces[i] = execution_space(streams[i]);
    }
  }
  };
#endif // HPG_ENABLE_CUDA

} // end namespace Impl
} // end namespace hpg

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
