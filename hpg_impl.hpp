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

/** abstract base class for state implementations */
struct State {

  Device device; /**< device type */

  State(Device device_)
    : device(device_) {}

  virtual void
  fence() = 0;

  virtual ~State() {}
};

/** Kokkos state implementation for a device type */
template <Device D>
class StateT
  : public State {
public:

  using execution_space = typename DeviceT<D>::kokkos_device::execution_space;
  using memory_space = typename DeviceT<D>::kokkos_device::memory_space;

  execution_space exec_space;

  std::array<unsigned, 3> grid_size;
  grid_view<typename GridLayout<D>::layout, memory_space> grid;

  StateT()
    : State(D) {
  }

  StateT(const std::array<unsigned, 3> grid_size_)
    : State(D)
    , grid_size(grid_size_) {

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

  StateT(StateT& st)
    : State(D) {
    std::array<int, 3> ig{
      static_cast<int>(st.grid_size[0]),
      static_cast<int>(st.grid_size[1]),
      static_cast<int>(st.grid_size[2])};
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
    K::deep_copy(exec_space, grid, st.grid);
  }

  StateT(StateT&& st)
    : State(D) {
    grid_size = std::move(st).grid_size;
    grid = st.grid;
  }

  StateT&
  operator=(StateT& st) {
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
  copy() {
    return StateT(*this);
  }

  void
  fence() override {
    exec_space.fence();
  }

private:
  void
  swap(StateT& right) {
    std::swap(grid_size, right.grid_size);
    decltype(grid) g = grid;
    grid = right.grid;
    right.grid = g;
  }
};

#ifdef HPG_ENABLE_CUDA
/** Kokkos state implementation for Cuda device */
template <>
struct StateT<Device::Cuda>
  : public State {
public:

  using execution_space = Kokkos::Cuda;
  using memory_space = execution_space::memory_space;

  std::array<unsigned, 3> grid_size;
  grid_view<GridLayout<Device::Cuda>::layout, memory_space> grid;

  // use two execution spaces to support overlap of data copying with
  // computation using CUDA streams
  std::array<cudaStream_t, 2> streams{NULL, NULL};
  std::array<execution_space, 2> exec_spaces;
  execution_space* exec_copy = &exec_spaces[0];
  execution_space* exec_compute = &exec_spaces[1];

  StateT(const std::array<unsigned, 3> grid_size_)
    : State(Device::Cuda)
    , grid_size(grid_size_) {

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

  StateT(StateT& st)
    : State(Device::Cuda) {

    st.fence();
    init_exec_spaces();

    std::array<int, 3> ig{
      static_cast<int>(st.grid_size[0]),
      static_cast<int>(st.grid_size[1]),
      static_cast<int>(st.grid_size[2])};
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
    K::deep_copy(*exec_copy, grid, st.grid);
  }

  StateT(StateT&& st)
    : State(Device::Cuda) {
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
  operator=(StateT& st) {
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
  copy() {
    return StateT(*this);
  }

  void
  fence() override {
    exec_compute->fence();
  }

private:
  void
  swap(StateT& right) {
    std::swap(streams[0], right.streams[0]);
    std::swap(streams[1], right.streams[1]);
    {
      auto r_copy_off = std::distance(&right.exec_spaces[0], right.exec_copy);
      auto copy_off = std::distance(&exec_spaces[0], exec_copy);
      exec_copy = &exec_spaces[r_copy_off];
      right.exec_copy = &right.exec_spaces[copy_off];
    }
    {
      auto r_compute_off =
        std::distance(&right.exec_spaces[1], right.exec_compute);
      auto compute_off = std::distance(&exec_spaces[1], exec_compute);
      exec_compute = &exec_spaces[r_compute_off];
      right.exec_compute = &right.exec_spaces[compute_off];
    }
    std::swap(grid_size, right.grid_size);
    decltype(grid) g = grid;
    grid = right.grid;
    right.grid = g;
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
