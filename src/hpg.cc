#include "hpg_impl.hpp"

#include <optional>
#include <variant>

using namespace hpg;

struct DisabledDeviceError
  : public Error {

  DisabledDeviceError()
    : Error("Requested device is not enabled", ErrorType::DisabledDevice) {}
};

struct InvalidNumberMuellerIndexRowsError
  : public Error {

  InvalidNumberMuellerIndexRowsError()
    : Error(
      "Number of rows of Mueller indexes does not match grid",
      ErrorType::InvalidNumberMuellerIndexRows) {}

};

struct InvalidNumberPolarizationsError
  : public Error {

  InvalidNumberPolarizationsError()
    : Error(
      "Number of visibility polarizations does not match Mueller matrix",
      ErrorType::InvalidNumberPolarizations) {}

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
  allocate_convolution_function_region(GS&& st, const CFArrayShape* shape) {

    ::hpg::GridderState result(std::forward<GS>(st));
    auto error = result.impl->allocate_convolution_function_region(shape);
    if (error)
      return std::move(error.value());
    else
      return std::move(result);
  }

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
  set_model(GS&& st, Device host_device, GridValueArray&& gv) {

    if (host_devices().count(host_device) > 0) {
      ::hpg::GridderState result(std::forward<GS>(st));
      auto error = result.impl->set_model(host_device, std::move(gv));
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
  grid_visibilities(GS&& st, Device host_device, VisDataVector&& visibilities) {

    if (host_devices().count(host_device) > 0) {
      if (visibilities.m_npol == st.impl->m_num_polarizations) {
        ::hpg::GridderState result(std::forward<GS>(st));
        auto error =
          result.impl->grid_visibilities(host_device, std::move(visibilities));
        if (error)
          return std::move(error.value());
        else
          return std::move(result);
      } else {
        return InvalidNumberPolarizationsError();
      }
    } else {
      return DisabledHostDeviceError();
    }
  }

  template <typename GS>
  static std::variant<Error, ::hpg::GridderState>
  apply_grid_fft(GS&& st, grid_value_fp norm, FFTSign sign, bool in_place) {

    ::hpg::GridderState result(std::forward<GS>(st));
    auto error = result.impl->apply_grid_fft(norm, sign, in_place);
    if (error)
      return std::move(error.value());
    else
      return std::move(result);
  }

  template <typename GS>
  static std::variant<Error, ::hpg::GridderState>
  apply_model_fft(GS&& st, grid_value_fp norm, FFTSign sign, bool in_place) {

    ::hpg::GridderState result(std::forward<GS>(st));
    auto error = result.impl->apply_model_fft(norm, sign, in_place);
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
  size_t max_visibility_batch_size,
  const CFArrayShape* init_cf_shape,
  const std::array<unsigned, 4>& grid_size,
  const std::array<grid_scale_fp, 2>& grid_scale,
  const IArrayVector& mueller_indexes,
  const IArrayVector& conjugate_mueller_indexes
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
        max_visibility_batch_size,
        init_cf_shape,
        grid_size,
        grid_scale,
        mueller_indexes,
        conjugate_mueller_indexes,
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
        max_visibility_batch_size,
        init_cf_shape,
        grid_size,
        grid_scale,
        mueller_indexes,
        conjugate_mueller_indexes,
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
        max_visibility_batch_size,
        init_cf_shape,
        grid_size,
        grid_scale,
        mueller_indexes,
        conjugate_mueller_indexes,
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
        max_visibility_batch_size,
        init_cf_shape,
        grid_size,
        grid_scale,
        mueller_indexes,
        conjugate_mueller_indexes,
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
  size_t max_visibility_batch_size,
  const CFArrayShape* init_cf_shape,
  const std::array<unsigned, 4>& grid_size,
  const std::array<grid_scale_fp, 2>& grid_scale,
  IArrayVector&& mueller_indexes,
  IArrayVector&& conjugate_mueller_indexes
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  , const std::array<unsigned, 4>& versions
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  ) noexcept {

  if (grid_size[2] != mueller_indexes.size()
      || grid_size[2] != conjugate_mueller_indexes.size())
    return rval<GridderState>(InvalidNumberMuellerIndexRowsError());

  if (devices().count(device) > 0)
    return
      rval<GridderState>(
        GridderState(
          device,
          max_added_tasks,
          max_visibility_batch_size,
          init_cf_shape,
          grid_size,
          grid_scale,
          mueller_indexes,
          conjugate_mueller_indexes
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
          , versions
#endif
          ));
  else
    return rval<GridderState>(DisabledDeviceError());

}

GridderState::GridderState(const GridderState& h) {
  *this = h;
}

GridderState::GridderState(GridderState&& h) {
  *this = std::move(h);
}

GridderState&
GridderState::operator=(const GridderState& rhs) {

  const GridderState& crhs = const_cast<const GridderState&>(rhs);
  switch (crhs.impl->m_device) {
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
  return impl->m_device;
}

unsigned
GridderState::max_added_tasks() const noexcept {
  return impl->m_max_active_tasks - 1;
}

size_t
GridderState::max_visibility_batch_size() const noexcept {
  return impl->m_max_visibility_batch_size;
}

const std::array<unsigned, 4>&
GridderState::grid_size() const noexcept {
  return impl->m_grid_size;
}

std::array<grid_scale_fp, 2>
GridderState::grid_scale() const noexcept {
  return {impl->m_grid_scale[0], impl->m_grid_scale[1]};
}

unsigned
GridderState::num_polarizations() const noexcept {
  return impl->m_num_polarizations;
}

bool
GridderState::is_null() const noexcept {
  return !bool(impl);
}

size_t
GridderState::convolution_function_region_size(const CFArrayShape* shape)
  const noexcept {

  return impl->convolution_function_region_size(shape);
}

rval_t<GridderState>
GridderState::allocate_convolution_function_region(const CFArrayShape* shape)
  const & {

  return
    to_rval(
      Impl::GridderState::allocate_convolution_function_region(*this, shape));
}

rval_t<GridderState>
GridderState::allocate_convolution_function_region(const CFArrayShape* shape)
  && {

  return
    to_rval(
      Impl::GridderState
      ::allocate_convolution_function_region(std::move(*this), shape));
}

rval_t<GridderState>
GridderState::set_convolution_function(Device host_device, CFArray&& cf)
  const & {

  return
    to_rval(
      Impl::GridderState
      ::set_convolution_function(*this, host_device, std::move(cf)));
}

rval_t<GridderState>
GridderState::set_convolution_function(Device host_device, CFArray&& cf) && {

  return
    to_rval(
      Impl::GridderState
      ::set_convolution_function(std::move(*this), host_device, std::move(cf)));
}

rval_t<GridderState>
GridderState::set_model(Device host_device, GridValueArray&& gv)
  const & {

  return
    to_rval(Impl::GridderState::set_model(*this, host_device, std::move(gv)));
}

rval_t<GridderState>
GridderState::set_model(Device host_device, GridValueArray&& gv) && {

  return
    to_rval(
      Impl::GridderState
      ::set_model(std::move(*this), host_device, std::move(gv)));
}

rval_t<GridderState>
GridderState::grid_visibilities(
  Device host_device,
  VisDataVector&& visibilities) const & {

  return
    to_rval(
      Impl::GridderState::grid_visibilities(
        *this,
        host_device,
        std::move(visibilities)));
}

rval_t<GridderState>
GridderState::grid_visibilities(
  Device host_device,
  VisDataVector&& visibilities) && {

  return
    to_rval(
      Impl::GridderState::grid_visibilities(
        std::move(*this),
        std::move(host_device),
        std::move(visibilities)));
}

GridderState
GridderState::fence() const & {

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
GridderState::grid_weights() const & {

  GridderState result(*this);
  return {std::move(result), std::move(result.impl->grid_weights())};
}

std::tuple<GridderState, std::unique_ptr<GridWeightArray>>
GridderState::grid_weights() && {

  GridderState result(std::move(*this));
  return {std::move(result), std::move(result.impl->grid_weights())};
}

std::tuple<GridderState, std::unique_ptr<GridValueArray>>
GridderState::grid_values() const & {

  GridderState result(*this);
  return {std::move(result), std::move(result.impl->grid_values())};
}

std::tuple<GridderState, std::unique_ptr<GridValueArray>>
GridderState::grid_values() && {

  GridderState result(std::move(*this));
  return {std::move(result), std::move(result.impl->grid_values())};
}

std::tuple<GridderState, std::unique_ptr<GridValueArray>>
GridderState::model_values() const & {

  GridderState result(*this);
  return {std::move(result), std::move(result.impl->model_values())};
}

std::tuple<GridderState, std::unique_ptr<GridValueArray>>
GridderState::model_values() && {

  GridderState result(std::move(*this));
  auto mv = result.impl->model_values();
  return {std::move(result), std::move(mv)};
}

GridderState
GridderState::reset_grid() const & {

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
GridderState::reset_model() const & {

  GridderState result(*this);
  result.impl->reset_model();
  return result;
}

GridderState
GridderState::reset_model() && {

  GridderState result(std::move(*this));
  result.impl->reset_model();
  return result;
}

GridderState
GridderState::normalize(grid_value_fp wfactor) const & {

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
GridderState::apply_grid_fft(
  grid_value_fp norm,
  FFTSign sign,
  bool in_place) const & {

  return
    to_rval(Impl::GridderState::apply_grid_fft(*this, norm, sign, in_place));
}

rval_t<GridderState>
GridderState::apply_grid_fft(
  grid_value_fp norm,
  FFTSign sign,
  bool in_place) && {

  return
    to_rval(
      Impl::GridderState::apply_grid_fft(
        std::move(*this),
        norm,
        sign,
        in_place));
}

rval_t<GridderState>
GridderState::apply_model_fft(
  grid_value_fp norm,
  FFTSign sign,
  bool in_place) const & {

  return
  to_rval(Impl::GridderState::apply_model_fft(*this, norm, sign, in_place));
}

rval_t<GridderState>
GridderState::apply_model_fft(
  grid_value_fp norm,
  FFTSign sign,
  bool in_place) && {

  return
  to_rval(
    Impl::GridderState::apply_model_fft(
      std::move(*this),
      norm,
      sign,
      in_place));
}

GridderState
GridderState::shift_grid() const & {

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

GridderState
GridderState::shift_model() const & {

  GridderState result(*this);
  result.impl->shift_model();
  return result;
}

GridderState
GridderState::shift_model() && {

  GridderState result(std::move(*this));
  result.impl->shift_model();
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
  size_t max_visibility_batch_size,
  const CFArrayShape* init_cf_shape,
  const std::array<unsigned, 4>& grid_size,
  const std::array<grid_scale_fp, 2>& grid_scale,
  IArrayVector&& mueller_indexes,
  IArrayVector&& conjugate_mueller_indexes)
  : state(
    GridderState(
      device,
      max_added_tasks,
      max_visibility_batch_size,
      init_cf_shape,
      grid_size,
      grid_scale,
      std::move(mueller_indexes),
      std::move(conjugate_mueller_indexes))) {}

Gridder::Gridder(const Gridder& other)
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
  size_t max_visibility_batch_size,
  const CFArrayShape* init_cf_shape,
  const std::array<unsigned, 4>& grid_size,
  const std::array<grid_scale_fp, 2>& grid_scale,
  IArrayVector&& mueller_indexes,
  IArrayVector&& conjugate_mueller_indexes
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  , const std::array<unsigned, 4>& implementation_versions
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  ) noexcept {

  auto err_or_gs =
    GridderState::create(
      device,
      max_added_tasks,
      max_visibility_batch_size,
      init_cf_shape,
      grid_size,
      grid_scale,
      std::move(mueller_indexes),
      std::move(conjugate_mueller_indexes)
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
      , implementation_versions
#endif
      );
  if (is_value(err_or_gs))
    return rval(Gridder(get_value(std::move(err_or_gs))));
  else
    return rval<Gridder>(get_error(std::move(err_or_gs)));
}

Gridder&
Gridder::operator=(const Gridder& rhs) {
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

size_t
Gridder::max_visibility_batch_size() const noexcept {
  return state.max_visibility_batch_size();
}

const std::array<unsigned, 4>&
Gridder::grid_size() const noexcept {
  return state.grid_size();
}

std::array<grid_scale_fp, 2>
Gridder::grid_scale() const noexcept {
  return state.grid_scale();
}

bool
Gridder::is_null() const noexcept {
  return state.is_null();
}

unsigned
Gridder::num_polarizations() const noexcept {
  return state.num_polarizations();
}

size_t
Gridder::convolution_function_region_size(const CFArrayShape* shape)
  const noexcept {

  return state.convolution_function_region_size(shape);
}

opt_error_t
Gridder::allocate_convolution_function_region(const CFArrayShape* shape) {
#if HPG_API >= 17
  return
    fold(
      std::move(state).allocate_convolution_function_region(shape),
      [this](auto&& gs) -> std::optional<Error> {
        this->state = std::move(gs);
        return std::nullopt;
      },
      [](auto&& err) -> std::optional<Error> {
        return std::move(err);
      });
#else //HPG_API < 17
  std::unique_ptr<Error> result;
  std::tie(result, state) =
    std::move(state).allocate_convolution_function_region(shape);
  return result;
#endif //HPG_API >= 17
}

opt_error_t
Gridder::set_convolution_function(Device host_device, CFArray&& cf) {
#if HPG_API >= 17
  return
    fold(
      std::move(state).set_convolution_function(host_device, std::move(cf)),
      [this](auto&& gs) -> std::optional<Error> {
        this->state = std::move(gs);
        return std::nullopt;
      },
      [](auto&& err) -> std::optional<Error> {
        return std::move(err);
      });
#else // HPG_API < 17
  std::unique_ptr<Error> result;
  std::tie(result, state) =
    std::move(state).set_convolution_function(host_device, std::move(cf));
  return result;
#endif //HPG_API >= 17
}

opt_error_t
Gridder::set_model(Device host_device, GridValueArray&& gv) {
#if HPG_API >= 17
  return
    fold(
      std::move(state).set_model(host_device, std::move(gv)),
      [this](auto&& gs) -> std::optional<Error> {
        this->state = std::move(gs);
        return std::nullopt;
      },
      [](auto&& err) -> std::optional<Error> {
        return std::move(err);
      });
#else // HPG_API < 17
  std::unique_ptr<Error> result;
  std::tie(result, state) =
    std::move(state).set_model(host_device, std::move(gv));
  return result;
#endif //HPG_API >= 17
}

opt_error_t
Gridder::grid_visibilities(Device host_device, VisDataVector&& visibilities) {
#if HPG_API >= 17
  return
    fold(
      std::move(state)
      .grid_visibilities(
        host_device,
        std::move(visibilities)),
      [this](auto&& gs) -> std::optional<Error> {
        this->state = std::move(gs);
        return std::nullopt;
      },
      [](auto&& err) -> std::optional<Error> {
        return std::move(err);
      });
#else // HPG_API < 17
  std::unique_ptr<Error> result;
  std::tie(result, state) =
    std::move(state)
    .grid_visibilities(
      host_device,
      std::move(visibilities));
  return result;
#endif // HPG_API >= 17
}

void
Gridder::fence() const {
  state = std::move(state).fence();
}

std::unique_ptr<GridWeightArray>
Gridder::grid_weights() const {
  std::unique_ptr<GridWeightArray> result;
  std::tie(const_cast<Gridder*>(this)->state, result) =
    std::move(const_cast<Gridder*>(this)->state).grid_weights();
  return result;
}

std::unique_ptr<GridValueArray>
Gridder::grid_values() const {
  std::unique_ptr<GridValueArray> result;
  std::tie(const_cast<Gridder*>(this)->state, result) =
    std::move(const_cast<Gridder*>(this)->state).grid_values();
  return result;
}

std::unique_ptr<GridValueArray>
Gridder::model_values() const {
  std::unique_ptr<GridValueArray> result;
  std::tie(const_cast<Gridder*>(this)->state, result) =
    std::move(const_cast<Gridder*>(this)->state).model_values();
  return result;
}

void
Gridder::reset_grid() {
  state = std::move(state).reset_grid();
}

void
Gridder::reset_model() {
  state = std::move(state).reset_model();
}

void
Gridder::normalize(grid_value_fp wgt_factor) {
  state = std::move(state).normalize(wgt_factor);
}

opt_error_t
Gridder::apply_grid_fft(grid_value_fp norm, FFTSign sign, bool in_place) {
#if HPG_API >= 17
  return
    fold(
      std::move(state).apply_grid_fft(norm, sign, in_place),
      [this](auto&& gs) -> std::optional<Error> {
        this->state = std::move(gs);
        return std::nullopt;
      },
      [](auto&& err) -> std::optional<Error> {
        return std::move(err);
      });
#else // HPG_API < 17
  std::unique_ptr<Error> result;
  std::tie(result, state) =
    std::move(state).apply_grid_fft(norm, sign, in_place);
  return result;
#endif //HPG_API >= 17
}

opt_error_t
Gridder::apply_model_fft(grid_value_fp norm, FFTSign sign, bool in_place) {
#if HPG_API >= 17
  return
    fold(
      std::move(state).apply_model_fft(norm, sign, in_place),
      [this](auto&& gs) -> std::optional<Error> {
        this->state = std::move(gs);
        return std::nullopt;
      },
      [](auto&& err) -> std::optional<Error> {
        return std::move(err);
      });
#else // HPG_API < 17
  std::unique_ptr<Error> result;
  std::tie(result, state) =
    std::move(state).apply_model_fft(norm, sign, in_place);
  return result;
#endif //HPG_API >= 17
}

void
Gridder::shift_grid() {
  state = std::move(state).shift_grid();
}

void
Gridder::shift_model() {
  state = std::move(state).shift_model();
}

opt_error_t
GridValueArray::copy_to(Device host_device, value_type* dst, Layout layout)
  const {

  static_assert(
    static_cast<int>(Impl::GridAxis::x) == GridValueArray::Axis::x
    && static_cast<int>(Impl::GridAxis::y) == GridValueArray::Axis::y
    && static_cast<int>(Impl::GridAxis::mrow) == GridValueArray::Axis::mrow
    && static_cast<int>(Impl::GridAxis::cube) == GridValueArray::Axis::cube);

#if HPG_API >= 17
  if (host_devices().count(host_device) == 0)
    return DisabledHostDeviceError();
  unsafe_copy_to(host_device, dst, layout);
  return std::nullopt;
#else // HPG_API < 17
  if (host_devices().count(host_device) == 0)
    return std::unique_ptr<Error>(new DisabledHostDeviceError());
  unsafe_copy_to(host_device, dst, layout);
  return nullptr;
#endif //HPG_API >= 17
}

opt_error_t
GridWeightArray::copy_to(Device host_device, value_type* dst, Layout layout)
  const {

  static_assert(
    static_cast<int>(Impl::GridAxis::x) == GridValueArray::Axis::x
    && static_cast<int>(Impl::GridAxis::y) == GridValueArray::Axis::y
    && static_cast<int>(Impl::GridAxis::mrow) == GridValueArray::Axis::mrow
    && static_cast<int>(Impl::GridAxis::cube) == GridValueArray::Axis::cube);

#if HPG_API >= 17
  if (host_devices().count(host_device) == 0)
    return DisabledHostDeviceError();
  unsafe_copy_to(host_device, dst, layout);
  return std::nullopt;
#else // HPG_API < 17
  if (host_devices().count(host_device) == 0)
    return std::unique_ptr<Error>(new DisabledHostDeviceError());
  unsafe_copy_to(host_device, dst, layout);
  return nullptr;
#endif //HPG_API >= 17
}

const char * const
hpg::cf_layout_unspecified_version = "";

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

rval_t<std::string>
CFArray::copy_to(
  Device device,
  Device host_device,
  unsigned grp,
  value_type* dst) const {

  if (host_devices().count(host_device) == 0)
    return rval<std::string>(DisabledHostDeviceError());

  if (devices().count(device) == 0)
    return rval<std::string>(DisabledDeviceError());

  switch (device) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial:
    Impl::layout_for_device<Device::Serial>(host_device, *this, grp, dst);
    break;
#endif
#ifdef HPG_ENABLE_OPENMP
  case Device::OpenMP:
    Impl::layout_for_device<Device::OpenMP>(host_device, *this, grp, dst);
    break;
#endif
#ifdef HPG_ENABLE_CUDA
  case Device::Cuda:
    Impl::layout_for_device<Device::Cuda>(host_device, *this, grp, dst);
    break;
#endif
#ifdef HPG_ENABLE_HPX
  case Device::HPX:
    Impl::layout_for_device<Device::HPX>(host_device, *this, grp, dst);
    break;
#endif
  default:
    assert(false);
    break;
  }
  return
    rval(
      Impl::construct_cf_layout_version(
        Impl::cf_layout_version_number,
        device));
}

rval_t<size_t>
CFArray::min_buffer_size(Device device, unsigned grp) const {

  if (devices().count(device) == 0)
    return rval<size_t>(DisabledDeviceError());

  size_t alloc_sz;

  switch (device) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial: {
    auto layout = Impl::CFLayout<Device::Serial>::dimensions(this, grp);
    alloc_sz =
      Impl::cf_view<
        typename Impl::DeviceT<Device::Serial>::kokkos_device::array_layout,
        K::HostSpace>
      ::required_allocation_size(
        layout.dimension[0],
        layout.dimension[1],
        layout.dimension[2],
        layout.dimension[3],
        layout.dimension[4],
        layout.dimension[5]);
    break;
  }
#endif
#ifdef HPG_ENABLE_OPENMP
  case Device::OpenMP: {
    auto layout = Impl::CFLayout<Device::OpenMP>::dimensions(this, grp);
    alloc_sz =
      Impl::cf_view<
        typename Impl::DeviceT<Device::OpenMP>::kokkos_device::array_layout,
        K::HostSpace>
      ::required_allocation_size(
        layout.dimension[0],
        layout.dimension[1],
        layout.dimension[2],
        layout.dimension[3],
        layout.dimension[4],
        layout.dimension[5]);
    break;
  }
#endif
#ifdef HPG_ENABLE_CUDA
  case Device::Cuda: {
    auto layout = Impl::CFLayout<Device::Cuda>::dimensions(this, grp);
    alloc_sz =
      Impl::cf_view<
        typename Impl::DeviceT<Device::Cuda>::kokkos_device::array_layout,
        K::HostSpace>
      ::required_allocation_size(
        layout.dimension[0],
        layout.dimension[1],
        layout.dimension[2],
        layout.dimension[3],
        layout.dimension[4],
        layout.dimension[5]);
    break;
  }
#endif
#ifdef HPG_ENABLE_HPX
  case Device::HPX: {
    auto layout = Impl::CFLayout<Device::HPX>::dimensions(this, grp);
    alloc_sz =
      Impl::cf_view<
        typename Impl::DeviceT<Device::HPX>::kokkos_device::array_layout,
        K::HostSpace>
      ::required_allocation_size(
        layout.dimension[0],
        layout.dimension[1],
        layout.dimension[2],
        layout.dimension[3],
        layout.dimension[4],
        layout.dimension[5]);
    break;
  }
#endif
  default:
    assert(false);
    break;
  }
  return
    rval<size_t>((alloc_sz + (sizeof(Impl::cf_t) - 1)) / sizeof(Impl::cf_t));
}

rval_t<std::unique_ptr<DeviceCFArray>>
DeviceCFArray::create(
  const std::string& layout,
  unsigned oversampling,
  std::vector<
    std::tuple<std::array<unsigned, rank - 1>, std::vector<value_type>>>&&
    arrays) {

  auto opt_vn_dev = Impl::parsed_cf_layout_version(layout);
  if (!opt_vn_dev)
    return
      rval<std::unique_ptr<DeviceCFArray>>(
        Error("Provided layout is invalid", ErrorType::InvalidCFLayout));
  auto& [vn, opt_dev] = opt_vn_dev.value();
  // require an exact device match in cf layout
  if (!opt_dev)
    return rval<std::unique_ptr<DeviceCFArray>>(DisabledDeviceError());
  switch (opt_dev.value()) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial:
    return
      rval<std::unique_ptr<DeviceCFArray>>(
        std::make_unique<Impl::DeviceCFArray<Device::Serial>>(
          layout,
          oversampling,
          std::move(arrays)));
#endif // HPG_ENABLE_SERIAL
    break;
#ifdef HPG_ENABLE_OPENMP
  case Device::OpenMP:
    return
      rval<std::unique_ptr<DeviceCFArray>>(
        std::make_unique<Impl::DeviceCFArray<Device::OpenMP>>(
          layout,
          oversampling,
          std::move(arrays)));
#endif // HPG_ENABLE_OPENMP
    break;
#ifdef HPG_ENABLE_CUDA
  case Device::Cuda:
    return
      rval<std::unique_ptr<DeviceCFArray>>(
        std::make_unique<Impl::DeviceCFArray<Device::Cuda>>(
          layout,
          oversampling,
          std::move(arrays)));
#endif //HPG_ENABLE_CUDA
    break;
#ifdef HPG_ENABLE_HPX
  case Device::HPX:
    return
      rval<std::unique_ptr<DeviceCFArray>>(
        std::make_unique<Impl::DeviceCFArray<Device::HPX>>(
          layout,
          oversampling,
          std::move(arrays)));
#endif // HPG_ENABLE_HPX
    break;
  default:
    return rval<std::unique_ptr<DeviceCFArray>>(DisabledDeviceError());
    break;
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
