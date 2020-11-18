#include "hpg_impl.hpp"

#include <optional>
#include <variant>

using namespace hpg;

struct Impl::GridderState {

  static std::variant<Error, ::hpg::GridderState>
  set_convolution_function(
    const volatile ::hpg::GridderState& st,
    Device host_device,
    const CFArray& cf) {

    ::hpg::GridderState result(st);
    auto error = result.impl->set_convolution_function(host_device, cf);
    if (error)
      return std::move(error.value());
    else
      return std::move(result);
  }

  static std::tuple<std::optional<Error>, ::hpg::GridderState>
  set_convolution_function(
    ::hpg::GridderState&& st,
    Device host_device,
    const CFArray& cf) {

    ::hpg::GridderState result(std::move(st));
    auto error = result.impl->set_convolution_function(host_device, cf);
    return {std::move(error), std::move(result)};
  }

  static std::variant<Error, ::hpg::GridderState>
  apply_fft(
    const volatile ::hpg::GridderState& st,
    FFTSign sign,
    bool in_place) {

    ::hpg::GridderState result(st);
    auto error = result.impl->apply_fft(sign, in_place);
    if (error)
      return std::move(error.value());
    else
      return std::move(result);
  }

  static std::tuple<std::optional<Error>, ::hpg::GridderState>
  apply_fft(::hpg::GridderState&& st, FFTSign sign, bool in_place) {

    ::hpg::GridderState result(std::move(st));
    auto error = result.impl->apply_fft(sign, in_place);
    return {std::move(error), std::move(result)};
  }
};

GridderState::GridderState() {
}

GridderState::GridderState(
  Device device,
  unsigned max_added_tasks,
  const std::array<unsigned, 4>& grid_size,
  const std::array<grid_scale_fp, 2>& grid_scale) {

  const unsigned max_active_tasks = max_added_tasks + 1;

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
GridderState::device() const noexcept {
  return impl->device;
}

unsigned
GridderState::max_added_tasks() const noexcept {
  return impl->max_active_tasks - 1;
}

const std::array<unsigned, 4>&
GridderState::grid_size() const noexcept {
  return impl->grid_size;
}

const std::array<grid_scale_fp, 2>&
GridderState::grid_scale() const noexcept {
  return impl->grid_scale;
}

bool
GridderState::is_null() const noexcept {
  return !bool(impl);
}

#if HPG_API >= 17
std::variant<Error, GridderState>
GridderState::set_convolution_function(
  Device host_device,
  const CFArray& cf) const volatile & {

  return Impl::GridderState::set_convolution_function(*this, host_device, cf);
}
#else
std::tuple<std::unique_ptr<Error>, GridderState>
GridderState::set_convolution_function(
  Device host_device,
  const CFArray& cf) const volatile & {

  auto err_or_gs =
    Impl::GridderState::set_convolution_function(*this, host_device, cf);
  if (std::holds_alternative<Error>(err_or_gs))
    return {std::make_unique<Error>(std::get<Error>(err_or_gs)), *this};
  else
    return {
      std::unique_ptr<Error>(),
      std::move(std::get<GridderState>(err_or_gs))};
}
#endif

#if HPG_API >= 17
std::tuple<std::optional<Error>, GridderState>
GridderState::set_convolution_function(
  Device host_device,
  const CFArray& cf) && {

  return
    Impl::GridderState
    ::set_convolution_function(std::move(*this), host_device, cf);
}
#else
std::tuple<std::unique_ptr<Error>, GridderState>
GridderState::set_convolution_function(
  Device host_device,
  const CFArray& cf) && {

  std::optional<Error> oerr;
  GridderState gs;
  std::tie(oerr, gs) =
    std::move(
      Impl::GridderState
      ::set_convolution_function(std::move(*this), host_device, cf));
  std::unique_ptr<Error> err;
  if (oerr)
    err =  std::make_unique<Error>(oerr.value());
  return {std::move(err), std::move(gs)};
}
#endif

GridderState
GridderState::grid_visibilities(
  Device host_device,
  const std::vector<std::complex<visibility_fp>>& visibilities,
  const std::vector<unsigned> visibility_grid_cubes,
  const std::vector<unsigned> visibility_cf_cubes,
  const std::vector<vis_weight_fp>& visibility_weights,
  const std::vector<vis_frequency_fp>& visibility_frequencies,
  const std::vector<vis_phase_fp>& visibility_phases,
  const std::vector<vis_uvw_t>& visibility_coordinates) const volatile & {

  GridderState result(*this);
  result.impl
  ->grid_visibilities(
    host_device,
    visibilities,
    visibility_grid_cubes,
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
  const std::vector<unsigned> visibility_grid_cubes,
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
    visibility_grid_cubes,
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

std::tuple<GridderState, std::unique_ptr<GridWeightArray>>
GridderState::grid_weights() const volatile & {

  GridderState result(*this);
  return {std::move(result), std::move(result.impl->grid_weights())};
}

std::tuple<GridderState, std::unique_ptr<GridWeightArray>>
GridderState::grid_weights() && {

  GridderState result(std::move(*this));
  return {std::move(result), std::move(result.impl->grid_weights())};
}

std::tuple<GridderState, std::unique_ptr<GridValueArray>>
GridderState::grid_values() const volatile & {

  GridderState result(*this);
  return {std::move(result), std::move(result.impl->grid_values())};
}

std::tuple<GridderState, std::unique_ptr<GridValueArray>>
GridderState::grid_values() && {

  GridderState result(std::move(*this));
  return {std::move(result), std::move(result.impl->grid_values())};
}

GridderState
GridderState::reset_grid() const volatile & {

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

GridderState
GridderState::normalize(grid_value_fp wfactor) const volatile & {

  GridderState result(*this);
  result.impl->normalize(wfactor);
  return result;
}

GridderState
GridderState::normalize(grid_value_fp wfactor) && {

  GridderState result(std::move(*this));
  result.impl->normalize(wfactor);
  return result;
}

#if HPG_API >= 17
std::variant<Error, GridderState>
GridderState::apply_fft(FFTSign sign, bool in_place) const volatile & {

  return Impl::GridderState::apply_fft(*this, sign, in_place);
}
#else
std::tuple<std::unique_ptr<Error>, GridderState>
GridderState::apply_fft(FFTSign sign, bool in_place) const volatile & {
  auto err_or_gs = Impl::GridderState::apply_fft(*this, sign, in_place);
  if (std::holds_alternative<Error>(err_or_gs))
    return {std::make_unique<Error>(std::get<Error>(err_or_gs)), *this};
  else
    return {
      std::unique_ptr<Error>(),
      std::move(std::get<GridderState>(err_or_gs))};
}
#endif

#if HPG_API >= 17
std::tuple<std::optional<Error>, GridderState>
GridderState::apply_fft(FFTSign sign, bool in_place) && {

  return Impl::GridderState::apply_fft(std::move(*this), sign, in_place);
}
#else
std::tuple<std::unique_ptr<Error>, GridderState>
GridderState::apply_fft(FFTSign sign, bool in_place) && {

  std::optional<Error> oerr;
  GridderState gs;
  std::tie(oerr, gs) =
    std::move(Impl::GridderState::apply_fft(std::move(*this), sign, in_place));
  std::unique_ptr<Error> err;
  if (oerr)
    err =  std::make_unique<Error>(oerr.value());
  return {std::move(err), std::move(gs)};
}
#endif

GridderState
GridderState::rotate_grid() const volatile & {

  GridderState result(*this);
  result.impl->rotate_grid();
  return result;
}

GridderState
GridderState::rotate_grid() && {

  GridderState result(std::move(*this));
  result.impl->rotate_grid();
  return result;
}

void
GridderState::swap(GridderState& other) noexcept {
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

bool
hpg::initialize() {
  return Impl::initialize();
}

void
hpg::finalize() {
  Impl::finalize();
}

bool
hpg::is_initialized() {
  return Impl::is_initialized();
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End: