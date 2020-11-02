#include "hpg.hpp"

#include <algorithm>
#include <cassert>

#include <iostream> // FIXME: remove

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

namespace K = Kokkos;

namespace hpg {
namespace Impl {

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

/** portable single precision complex type */
using cxf_t = Kokkos::complex<float>;

/** portable double precision complex type */
using cxd_t = Kokkos::complex<double>;

/** visibility value type */
using vis_t = cxf_t;

/** convolution function value type */
using cf_t = cxf_t;

/** gridded value type */
using gv_t = cxd_t;

/** Kokkos::View type for grid values
 */
template <typename Layout, typename memory_space>
using grid_view = K::View<gv_t***, Layout, memory_space>;

template <typename Layout, typename memory_space>
using const_grid_view = K::View<const gv_t***, Layout, memory_space>;

/**
 * View type for CF2 values
 */
template <typename Layout, typename memory_space>
using cf2_view = K::View<cf_t****, Layout, memory_space>;

template <typename Layout, typename memory_space>
using const_cf2_view = K::View<const cf_t****, Layout, memory_space>;

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

  /**
   * create Kokkos layout using given grid dimensions (in logical order X, Y,
   * ch)
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

/** abstract base class for state implementations */
struct State {

  Device device; /**< device type */
  std::array<unsigned, 3> grid_size; /**< grid size */
  std::array<float, 2> grid_scale; /**< grid scale */

  State(
    Device device_,
    const std::array<unsigned, 3>& grid_size_,
    const std::array<float, 2>& grid_scale_)
    : device(device_)
    , grid_size(grid_size_)
    , grid_scale(grid_scale_) {}

  virtual void
  set_convolution_function(Device host_device, const CF2& cf) = 0;

  virtual void
  fence() const volatile = 0;

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


/** Kokkos state implementation for a device type */
template <Device D>
class HPG_EXPORT StateT final
  : public State {
public:

  using execution_space = typename DeviceT<D>::kokkos_device::execution_space;
  using memory_space = typename DeviceT<D>::kokkos_device::memory_space;

  execution_space exec_space;

  grid_view<typename GridLayout<D>::layout, memory_space> grid;
  const_cf2_view<typename CF2Layout<D>::layout, memory_space> cf;

  StateT()
    : State(D) {
  }

  StateT(
    const std::array<unsigned, 3> grid_size,
    const std::array<float, 2>& grid_scale)
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
  }

  StateT(StateT&& st)
    : State(D, std::move(st).grid_size, std::move(st).grid_scale) {
    grid = st.grid;
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
  fence() const volatile override {
    const_cast<const execution_space&>(exec_space).fence();
  }

private:
  void
  swap(StateT& other) {
    std::swap(grid_size, other.grid_size);
    std::swap(grid_scale, other.grid_scale);
    decltype(grid) g = grid;
    grid = other.grid;
    other.grid = g;
  }
};

#ifdef HPG_ENABLE_CUDA
/** Kokkos state implementation for Cuda device */
template <>
struct HPG_EXPORT StateT<Device::Cuda> final
  : public State {
public:

  using execution_space = Kokkos::Cuda;
  using memory_space = execution_space::memory_space;

  grid_view<GridLayout<Device::Cuda>::layout, memory_space> grid;
  const_cf2_view<CF2Layout<Device::Cuda>::layout, memory_space> cf;

  // use two execution spaces to support overlap of data copying with
  // computation using CUDA streams
  std::array<cudaStream_t, 2> streams{NULL, NULL};
  std::array<execution_space, 2> exec_spaces;
  execution_space* exec_copy = &exec_spaces[0];
  execution_space* exec_compute = &exec_spaces[1];

  StateT(
    const std::array<unsigned, 3> grid_size,
    const std::array<float, 2>& grid_scale)
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
  }

  StateT(StateT&& st)
    : State(Device::Cuda, std::move(st).grid_size, std::move(st).grid_scale) {
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
  fence() const volatile override {
    const_cast<const execution_space*>(exec_compute)->fence();
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
