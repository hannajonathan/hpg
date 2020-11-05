#include "hpg_impl.hpp"

using namespace hpg;

GridderState::GridderState() {
}

GridderState::GridderState(
  Device device,
  unsigned max_async_tasks,
  const std::array<unsigned, 4>& grid_size,
  const std::array<grid_scale_fp, 2>& grid_scale) {

  const unsigned max_active_tasks = max_async_tasks + 1;

  switch (device) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial:
    impl =
      std::make_shared<Impl::StateT<Device::Serial>>(
        max_active_tasks,
        grid_size,
        grid_scale);
    break;
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
  case Device::OpenMP:
    impl =
      std::make_shared<Impl::StateT<Device::OpenMP>>(
        max_active_tasks,
        grid_size,
        grid_scale);
    break;
#endif // HPG_ENABLE_OPENMP
#ifdef HPG_ENABLE_CUDA
  case Device::Cuda:
    impl =
      std::make_shared<Impl::StateT<Device::Cuda>>(
        max_active_tasks,
        grid_size,
        grid_scale);
    break;
#endif // HPG_ENABLE_CUDA
#ifdef HPG_ENABLE_HPX
  case Device::HPX:
    impl =
      std::make_shared<Impl::StateT<Device::HPX>>(
        max_active_tasks,
        grid_size,
        grid_scale);
    break;
#endif // HPG_ENABLE_HPX
  default:
    assert(false);
    break;
  }
}

GridderState::GridderState(const volatile GridderState& h) {
  *this = h;
}

GridderState::GridderState(GridderState&& h) {
  *this = std::move(h);
}

GridderState&
GridderState::operator=(const volatile GridderState& rhs) {

  const GridderState& crhs = const_cast<const GridderState&>(rhs);
  switch (crhs.impl->device) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial:
    impl =
      std::make_shared<Impl::StateT<Device::Serial>>(
        dynamic_cast<Impl::StateT<Device::Serial>*>(crhs.impl.get())
        ->copy());
    break;
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
  case Device::OpenMP:
    impl =
      std::make_shared<Impl::StateT<Device::OpenMP>>(
        dynamic_cast<Impl::StateT<Device::OpenMP>*>(crhs.impl.get())
        ->copy());
    break;
#endif // HPG_ENABLE_OPENMP
#ifdef HPG_ENABLE_CUDA
  case Device::Cuda:
    impl =
      std::make_shared<Impl::StateT<Device::Cuda>>(
        dynamic_cast<Impl::StateT<Device::Cuda>*>(crhs.impl.get())
        ->copy());
    break;
#endif // HPG_ENABLE_CUDA
#ifdef HPG_ENABLE_HPX
  case Device::HPX:
    impl =
      std::make_shared<Impl::StateT<Device::HPX>>(
        dynamic_cast<const Impl::StateT<Device::HPX>*>(crhs.impl.get())
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
GridderState::operator=(GridderState&& rhs) {

  impl = std::move(std::move(rhs).impl);
  return *this;
}

Device
GridderState::device() const {
  return impl->device;
}

unsigned
GridderState::max_async_tasks() const {
  return impl->max_active_tasks - 1;
}

const std::array<unsigned, 4>&
GridderState::grid_size() const {
  return impl->grid_size;
}

const std::array<grid_scale_fp, 2>&
GridderState::grid_scale() const {
  return impl->grid_scale;
}

GridderState
GridderState::set_convolution_function(
  Device host_device,
  const CFArray& cf) & {

  GridderState result(*this);
  result.impl->set_convolution_function(host_device, cf);
  return result;
}

GridderState
GridderState::set_convolution_function(
  Device host_device,
  const CFArray& cf) && {

  GridderState result(std::move(*this));
  result.impl->set_convolution_function(host_device, cf);
  return result;
}

GridderState
GridderState::grid_visibilities(
  Device host_device,
  const std::vector<std::complex<visibility_fp>>& visibilities,
  const std::vector<grid_plane_t> visibility_grid_planes,
  const std::vector<unsigned> visibility_cf_cubes,
  const std::vector<vis_weight_fp>& visibility_weights,
  const std::vector<vis_frequency_fp>& visibility_frequencies,
  const std::vector<vis_phase_fp>& visibility_phases,
  const std::vector<vis_uvw_t>& visibility_coordinates) & {

  GridderState result(*this);
  result.impl
  ->grid_visibilities(
    host_device,
    visibilities,
    visibility_grid_planes,
    visibility_cf_cubes,
    visibility_weights,
    visibility_frequencies,
    visibility_phases,
    visibility_coordinates);
  return result;
}

GridderState
GridderState::grid_visibilities(
  Device host_device,
  const std::vector<std::complex<visibility_fp>>& visibilities,
  const std::vector<grid_plane_t> visibility_grid_planes,
  const std::vector<unsigned> visibility_cf_cubes,
  const std::vector<vis_weight_fp>& visibility_weights,
  const std::vector<vis_frequency_fp>& visibility_frequencies,
  const std::vector<vis_phase_fp>& visibility_phases,
  const std::vector<vis_uvw_t>& visibility_coordinates) && {

  GridderState result(std::move(*this));
  result.impl
  ->grid_visibilities(
    host_device,
    visibilities,
    visibility_grid_planes,
    visibility_cf_cubes,
    visibility_weights,
    visibility_frequencies,
    visibility_phases,
    visibility_coordinates);
  return result;
}

GridderState
GridderState::fence() const volatile & {

  GridderState result(*this);
  result.impl->fence();
  return result;
}

GridderState
GridderState::fence() && {

  GridderState result(std::move(*this));
  result.impl->fence();
  return result;
}

std::tuple<GridderState, std::shared_ptr<GridWeightArray>>
GridderState::grid_weights() const volatile & {

  GridderState result(*this);
  return {std::move(result), result.impl->grid_weights()};
}

std::tuple<GridderState, std::shared_ptr<GridWeightArray>>
GridderState::grid_weights() && {

  GridderState result(std::move(*this));
  return {std::move(result), result.impl->grid_weights()};
}

std::tuple<GridderState, std::shared_ptr<GridValueArray>>
GridderState::grid_values() const volatile & {

  GridderState result(*this);
  return {std::move(result), result.impl->grid_values()};
}

std::tuple<GridderState, std::shared_ptr<GridValueArray>>
GridderState::grid_values() && {

  GridderState result(std::move(*this));
  return {std::move(result), result.impl->grid_values()};
}

GridderState
GridderState::reset_grid() & {

  GridderState result(*this);
  result.impl->reset_grid();
  return result;
}

GridderState
GridderState::reset_grid() && {

  GridderState result(std::move(*this));
  result.impl->reset_grid();
  return result;
}

void
GridderState::swap(GridderState& other) {
  std::swap(impl, other.impl);
}

Gridder&
Gridder::operator=(const volatile Gridder& rhs) {
  GridderState tmp(rhs.state);
  state.swap(tmp);
  return *this;
}

Gridder&
Gridder::operator=(Gridder&& rhs) {
  GridderState tmp(std::move(rhs).state);
  state.swap(tmp);
  return *this;
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
