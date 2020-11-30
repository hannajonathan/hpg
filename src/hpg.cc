#include "hpg_impl.hpp"

#include <optional>
#include <variant>

using namespace hpg;

struct DisabledDeviceError
  : public Error {

  DisabledDeviceError()
    : Error("Requested device is not enabled", ErrorType::DisabledDevice) {}
};

struct DisabledHostDeviceError
  : public Error {

  DisabledHostDeviceError()
    : Error(
      "Requested host device is not enabled",
      ErrorType::DisabledHostDevice) {}
};

struct IncompatibleVisVectorLengthError
  : public Error {

  IncompatibleVisVectorLengthError()
    : Error (
      "Incompatible visibility vector lengths",
      ErrorType::IncompatibleVisVectorLengths) {}
};

Error::Error(const std::string& msg, ErrorType err)
  : m_type(err)
  , m_msg(msg) {}

const std::string&
Error::message() const {
  return m_msg;
}

ErrorType
Error::type() const {
  return m_type;
}

Error::~Error() {}

ScopeGuard::ScopeGuard()
  : init(false) {
  if (!is_initialized()) {
    initialize();
    init = true;
  }
}

ScopeGuard::~ScopeGuard() {
  if (is_initialized() && init)
    finalize();
}

struct Impl::GridderState {

  template <typename GS>
  static std::variant<Error, ::hpg::GridderState>
  set_convolution_function(GS&& st, Device host_device, CFArray&& cf) {

    if (host_devices().count(host_device) > 0) {
      ::hpg::GridderState result(std::forward<GS>(st));
      auto error =
        result.impl->set_convolution_function(host_device, std::move(cf));
      if (error)
        return std::move(error.value());
      else
        return std::move(result);
    } else {
      return DisabledHostDeviceError();
    }
  }

  template <typename GS>
  static std::variant<Error, ::hpg::GridderState>
  grid_visibilities(
    GS&& st,
    Device host_device,
    std::vector<std::complex<visibility_fp>>&& visibilities,
    std::vector<unsigned>&& visibility_grid_cubes,
    std::vector<unsigned>&& visibility_cf_cubes,
    std::vector<vis_weight_fp>&& visibility_weights,
    std::vector<vis_frequency_fp>&& visibility_frequencies,
    std::vector<vis_phase_fp>&& visibility_phases,
    std::vector<vis_uvw_t>&& visibility_coordinates) {

    auto len = visibilities.size();
    if (visibility_grid_cubes.size() < len
        || visibility_cf_cubes.size() < len
        || visibility_weights.size() < len
        || visibility_frequencies.size() < len
        || visibility_phases.size() < len
        || visibility_coordinates.size() < len)
      return IncompatibleVisVectorLengthError();

    if (host_devices().count(host_device) > 0) {
      ::hpg::GridderState result(std::forward<GS>(st));
      result.impl->grid_visibilities(
        host_device,
        std::move(visibilities),
        std::move(visibility_grid_cubes),
        std::move(visibility_cf_cubes),
        std::move(visibility_weights),
        std::move(visibility_frequencies),
        std::move(visibility_phases),
        std::move(visibility_coordinates));
      return result;
    } else {
      return DisabledHostDeviceError();
    }
  }

  template <typename GS>
  static std::variant<Error, ::hpg::GridderState>
  apply_fft(GS&& st, FFTSign sign, bool in_place) {

    ::hpg::GridderState result(std::forward<GS>(st));
    auto error = result.impl->apply_fft(sign, in_place);
    if (error)
      return std::move(error.value());
    else
      return std::move(result);
  }
};

template <typename T>
static rval_t<T>
to_rval(std::variant<Error, T>&& t) {
  if (std::holds_alternative<T>(t))
    return rval<T>(std::get<T>(std::move(t)));
  else
    return rval<T>(std::get<Error>(std::move(t)));
}

GridderState::GridderState() {
}

GridderState::GridderState(
  Device device,
  unsigned max_added_tasks,
  const std::array<unsigned, 4>& grid_size,
  const std::array<grid_scale_fp, 2>& grid_scale
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  , const std::array<unsigned, 4>& implementation_versions
#endif
  ) {

#ifndef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  std::array<unsigned, 4> implementation_versions{0, 0, 0, 0};
#endif

  const unsigned max_active_tasks = max_added_tasks + 1;

  switch (device) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial:
    impl =
      std::make_shared<Impl::StateT<Device::Serial>>(
        max_active_tasks,
        grid_size,
        grid_scale,
        implementation_versions);
#else
    assert(false);
#endif // HPG_ENABLE_SERIAL
    break;
#ifdef HPG_ENABLE_OPENMP
  case Device::OpenMP:
    impl =
      std::make_shared<Impl::StateT<Device::OpenMP>>(
        max_active_tasks,
        grid_size,
        grid_scale,
        implementation_versions);
#else
    assert(false);
#endif // HPG_ENABLE_OPENMP
    break;
#ifdef HPG_ENABLE_CUDA
  case Device::Cuda:
    impl =
      std::make_shared<Impl::StateT<Device::Cuda>>(
        max_active_tasks,
        grid_size,
        grid_scale,
        implementation_versions);
#else
    assert(false);
#endif //HPG_ENABLE_CUDA
    break;
#ifdef HPG_ENABLE_HPX
  case Device::HPX:
    impl =
      std::make_shared<Impl::StateT<Device::HPX>>(
        max_active_tasks,
        grid_size,
        grid_scale,
        implementation_versions);
#else
    assert(false);
#endif // HPG_ENABLE_HPX
    break;
  default:
    assert(false);
    break;
  }
}

rval_t<GridderState>
GridderState::create(
  Device device,
  unsigned max_added_tasks,
  const std::array<unsigned, 4>& grid_size,
  const std::array<grid_scale_fp, 2>& grid_scale
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  , const std::array<unsigned, 4>& versions
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  ) noexcept {

  if (devices().count(device) > 0)
    return
      rval<GridderState>(
        GridderState(
          device,
          max_added_tasks,
          grid_size,
          grid_scale
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
          , versions
#endif
          ));
  else
    return
      rval<GridderState>(DisabledDeviceError());

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

  impl = std::move(rhs).impl; // TODO: is this OK, or do I need another move()?
  return *this;
}

GridderState::~GridderState() {}

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

rval_t<GridderState>
GridderState::set_convolution_function(Device host_device, CFArray&& cf)
  const volatile & {

  return
    to_rval(
      Impl::GridderState::set_convolution_function(
        *this,
        host_device,
        std::move(cf)));
}

rval_t<GridderState>
GridderState::set_convolution_function(Device host_device, CFArray&& cf) && {

  return
    to_rval(
      Impl::GridderState
      ::set_convolution_function(
        std::move(*this),
        host_device,
        std::move(cf)));
}

rval_t<GridderState>
GridderState::grid_visibilities(
  Device host_device,
  std::vector<std::complex<visibility_fp>>&& visibilities,
  std::vector<unsigned>&& visibility_grid_cubes,
  std::vector<unsigned>&& visibility_cf_cubes,
  std::vector<vis_weight_fp>&& visibility_weights,
  std::vector<vis_frequency_fp>&& visibility_frequencies,
  std::vector<vis_phase_fp>&& visibility_phases,
  std::vector<vis_uvw_t>&& visibility_coordinates) const volatile & {

  return
    to_rval(
      Impl::GridderState::grid_visibilities(
        *this,
        host_device,
        std::move(visibilities),
        std::move(visibility_grid_cubes),
        std::move(visibility_cf_cubes),
        std::move(visibility_weights),
        std::move(visibility_frequencies),
        std::move(visibility_phases),
        std::move(visibility_coordinates)));
}

rval_t<GridderState>
GridderState::grid_visibilities(
  Device host_device,
  std::vector<std::complex<visibility_fp>>&& visibilities,
  std::vector<unsigned>&& visibility_grid_cubes,
  std::vector<unsigned>&& visibility_cf_cubes,
  std::vector<vis_weight_fp>&& visibility_weights,
  std::vector<vis_frequency_fp>&& visibility_frequencies,
  std::vector<vis_phase_fp>&& visibility_phases,
  std::vector<vis_uvw_t>&& visibility_coordinates) && {

  return
    to_rval(
      Impl::GridderState::grid_visibilities(
        std::move(*this),
        std::move(host_device),
        std::move(visibilities),
        std::move(visibility_grid_cubes),
        std::move(visibility_cf_cubes),
        std::move(visibility_weights),
        std::move(visibility_frequencies),
        std::move(visibility_phases),
        std::move(visibility_coordinates)));

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

rval_t<GridderState>
GridderState::apply_fft(FFTSign sign, bool in_place) const volatile & {

  return to_rval(Impl::GridderState::apply_fft(*this, sign, in_place));
}

rval_t<GridderState>
GridderState::apply_fft(FFTSign sign, bool in_place) && {

  return
    to_rval(Impl::GridderState::apply_fft(std::move(*this), sign, in_place));
}

GridderState
GridderState::shift_grid() const volatile & {

  GridderState result(*this);
  result.impl->shift_grid();
  return result;
}

GridderState
GridderState::shift_grid() && {

  GridderState result(std::move(*this));
  result.impl->shift_grid();
  return result;
}

void
GridderState::swap(GridderState& other) noexcept {
  std::swap(impl, other.impl);
}

Gridder::Gridder() {}

Gridder::Gridder(
  Device device,
  unsigned max_added_tasks,
  const std::array<unsigned, 4>& grid_size,
  const std::array<grid_scale_fp, 2>& grid_scale)
  : state(GridderState(device, max_added_tasks, grid_size, grid_scale)) {}

Gridder::Gridder(const volatile Gridder& other)
  : state(other.state) {}

Gridder::Gridder(Gridder&& other)
  : state(std::move(other).state) {}

Gridder::Gridder(GridderState&& st)
  : state(std::move(st)) {}

Gridder::~Gridder() {}

rval_t<Gridder>
Gridder::create(
  Device device,
  unsigned max_added_tasks,
  const std::array<unsigned, 4>& grid_size,
  const std::array<grid_scale_fp, 2>& grid_scale) noexcept {

  auto err_or_gs =
    GridderState::create(device, max_added_tasks, grid_size, grid_scale);
  if (is_value(err_or_gs))
    return rval(Gridder(std::move(get_value(err_or_gs))));
  else
    return rval<Gridder>(std::move(get_error(err_or_gs)));
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

Device
Gridder::device() const noexcept {
  return state.device();
}

unsigned
Gridder::max_added_tasks() const noexcept {
  return state.max_added_tasks();
}

const std::array<unsigned, 4>&
Gridder::grid_size() const noexcept {
  return state.grid_size();
}

const std::array<grid_scale_fp, 2>&
Gridder::grid_scale() const noexcept {
  return state.grid_scale();
}

bool
Gridder::is_null() const noexcept {
  return state.is_null();
}

#if HPG_API >= 17
std::optional<Error>
Gridder::set_convolution_function(Device host_device, CFArray&& cf) {

  std::optional<Error> result;
  auto err_or_gs =
    std::move(state).set_convolution_function(host_device, std::move(cf));
  if (std::holds_alternative<GridderState>(err_or_gs))
    state = std::move(std::get<GridderState>(err_or_gs));
  else
    result = std::move(std::get<Error>(err_or_gs));
  return result;
}
#else // HPG_API < 17
std::unique_ptr<Error>
Gridder::set_convolution_function(Device host_device, CFArray&& cf) {

  std::unique_ptr<Error> result;
  std::tie(result, state) =
    std::move(state).set_convolution_function(host_device, std::move(cf));
  return result;
}
#endif //HPG_API >= 17

#if HPG_API >= 17
std::optional<Error>
Gridder::grid_visibilities(
  Device host_device,
  std::vector<std::complex<visibility_fp>>&& visibilities,
  std::vector<unsigned>&& visibility_grid_cubes,
  std::vector<unsigned>&& visibility_cf_cubes,
  std::vector<vis_weight_fp>&& visibility_weights,
  std::vector<vis_frequency_fp>&& visibility_frequencies,
  std::vector<vis_phase_fp>&& visibility_phases,
  std::vector<vis_uvw_t>&& visibility_coordinates) {

  std::optional<Error> result;
  auto err_or_gs =
    std::move(state)
    .grid_visibilities(
      host_device,
      std::move(visibilities),
      std::move(visibility_grid_cubes),
      std::move(visibility_cf_cubes),
      std::move(visibility_weights),
      std::move(visibility_frequencies),
      std::move(visibility_phases),
      std::move(visibility_coordinates));
  if (std::holds_alternative<GridderState>(err_or_gs))
    state = std::move(std::get<GridderState>(err_or_gs));
  else
    result = std::move(std::get<Error>(err_or_gs));
  return result;
}
#else // HPG_API < 17
std::unique_ptr<Error>
Gridder::grid_visibilities(
  Device host_device,
  std::vector<std::complex<visibility_fp>>&& visibilities,
  std::vector<unsigned>&& visibility_grid_cubes,
  std::vector<unsigned>&& visibility_cf_cubes,
  std::vector<vis_weight_fp>&& visibility_weights,
  std::vector<vis_frequency_fp>&& visibility_frequencies,
  std::vector<vis_phase_fp>&& visibility_phases,
  std::vector<vis_uvw_t>&& visibility_coordinates) {

  std::unique_ptr<Error> result;
  std::tie(result, state) =
    std::move(state)
    .grid_visibilities(
      host_device,
      std::move(visibilities),
      std::move(visibility_grid_cubes),
      std::move(visibility_cf_cubes),
      std::move(visibility_weights),
      std::move(visibility_frequencies),
      std::move(visibility_phases),
      std::move(visibility_coordinates));
  return result;
}
#endif

void
Gridder::fence() const volatile {
  const_cast<Gridder*>(this)->state =
    std::move(const_cast<Gridder*>(this)->state).fence();
}

std::unique_ptr<GridWeightArray>
Gridder::grid_weights() const volatile {
  std::unique_ptr<GridWeightArray> result;
  std::tie(const_cast<Gridder*>(this)->state, result) =
    std::move(const_cast<Gridder*>(this)->state).grid_weights();
  return result;
}

std::unique_ptr<GridValueArray>
Gridder::grid_values() const volatile {
  std::unique_ptr<GridValueArray> result;
  std::tie(const_cast<Gridder*>(this)->state, result) =
    std::move(const_cast<Gridder*>(this)->state).grid_values();
  return result;
}

void
Gridder::reset_grid() {
  state = std::move(state).reset_grid();
}

void
Gridder::normalize(grid_value_fp wgt_factor) {
  state = std::move(state).normalize(wgt_factor);
}

#if HPG_API >= 17
std::optional<Error>
Gridder::apply_fft(FFTSign sign, bool in_place) {
  std::optional<Error> result;
  auto err_or_gs = std::move(state).apply_fft(sign, in_place);
  if (std::holds_alternative<GridderState>(err_or_gs))
    state = std::move(std::get<GridderState>(err_or_gs));
  else
    result = std::move(std::get<Error>(err_or_gs));
  return result;
}
#else // HPG_API < 17
std::unique_ptr<Error>
Gridder::apply_fft(FFTSign sign, bool in_place) {
  std::unique_ptr<Error> result;
  std::tie(result, state) = std::move(state).apply_fft(sign, in_place);
  return result;
}
#endif //HPG_API >= 17

void
Gridder::shift_grid() {
  state = std::move(state).shift_grid();
}

bool
hpg::initialize() {
  return Impl::initialize();
}

void
hpg::finalize() {
  Impl::finalize();
}

const std::set<Device>&
hpg::devices() noexcept {
  static const std::set<Device> result{
#ifdef HPG_ENABLE_SERIAL
    Device::Serial,
#endif
#ifdef HPG_ENABLE_OPENMP
    Device::OpenMP,
#endif
#ifdef HPG_ENABLE_CUDA
    Device::Cuda,
#endif
#ifdef HPG_ENABLE_HPX
    Device::HPX,
#endif
  };
  return result;
}

const std::set<Device>&
hpg::host_devices() noexcept {
  static const std::set<Device> result{
#ifdef HPG_ENABLE_SERIAL
    Device::Serial,
#endif
#ifdef HPG_ENABLE_OPENMP
    Device::OpenMP,
#endif
  };
  return result;
}

bool
hpg::is_initialized() noexcept {
  return Impl::is_initialized();
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
