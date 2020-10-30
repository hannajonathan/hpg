#include "hpg_impl.hpp"

using namespace hpg;

hpg::GridderState::GridderState() {
  std::cout << "default construct" << std::endl;
}

hpg::GridderState::GridderState(GridderState& h) {
  *this = h;
  std::cout << "copy construct" << std::endl;
}

hpg::GridderState::GridderState(GridderState&& h) {
  *this = std::move(h);
  std::cout << "move construct" << std::endl;
}

GridderState&
hpg::GridderState::operator=(GridderState& rhs) {
  switch (rhs.impl->device) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial:
    impl =
      std::make_shared<Impl::StateT<Device::Serial>>(
        dynamic_cast<Impl::StateT<Device::Serial>*>(rhs.impl.get())
        ->copy());
    break;
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
  case Device::OpenMP:
    impl =
      std::make_shared<Impl::StateT<Device::OpenMP>>(
        dynamic_cast<Impl::StateT<Device::OpenMP>*>(rhs.impl.get())
        ->copy());
    break;
#endif // HPG_ENABLE_OPENMP
#ifdef HPG_ENABLE_CUDA
  case Device::Cuda:
    impl =
      std::make_shared<Impl::StateT<Device::Cuda>>(
        dynamic_cast<Impl::StateT<Device::Cuda>*>(rhs.impl.get())
        ->copy());
    break;
#endif // HPG_ENABLE_CUDA
#ifdef HPG_ENABLE_HPX
  case Device::HPX:
    impl =
      std::make_shared<Impl::StateT<Device::HPX>>(
        dynamic_cast<const Impl::StateT<Device::HPX>*>(rhs.impl.get())
        ->copy());
    break;
#endif // HPG_ENABLE_HPX
  default:
    assert(false);
    break;
  }
  return *this;
}

GridderState&
hpg::GridderState::operator=(GridderState&& rhs) {
  impl = std::move(std::move(rhs).impl);
  return *this;
}

GridderState
Gridder::init(Device device, const std::array<unsigned, 3>& grid_size) {

  GridderState result;

  switch (device) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial:
    result.impl = std::make_shared<Impl::StateT<Device::Serial>>(grid_size);
    break;
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
  case Device::OpenMP:
    result.impl = std::make_shared<Impl::StateT<Device::OpenMP>>(grid_size);
    break;
#endif // HPG_ENABLE_OPENMP
#ifdef HPG_ENABLE_CUDA
  case Device::Cuda:
    result.impl = std::make_shared<Impl::StateT<Device::Cuda>>(grid_size);
    break;
#endif // HPG_ENABLE_CUDA
#ifdef HPG_ENABLE_HPX
  case Device::HPX:
    result.impl = std::make_shared<Impl::StateT<Device::HPX>>(grid_size);
    break;
#endif // HPG_ENABLE_HPX
  default:
    assert(false);
    break;
  }
  return result;
}

GridderState
Gridder::fence(GridderState handle) {
  handle.impl->fence();
  return handle;
}

void
hpg::initialize() {
  Impl::initialize();
}

void
hpg::finalize() {
  Impl::finalize();
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
