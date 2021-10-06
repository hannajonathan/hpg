// Copyright 2021 Associated Universities, Inc. Washington DC, USA.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#include "hpg_impl.hpp"
#include "hpg_runtime.hpp"

#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

using namespace hpg;

using ProfileRegion = runtime::impl::ProfileRegion;

// don't do "using runtime" in file scope, since that namespace also has a class
// named "GridderState", the use of which would then require disambiguation; I
// prefer putting "using runtime" into method implementations where useful

/** invalid number of Mueller index rows error
 *
 * Number of rows of Mueller indexes does not equal grid "mrow" axis size */
struct InvalidNumberMuellerIndexRowsError
  : public Error {

  InvalidNumberMuellerIndexRowsError()
    : Error(
      "Number of rows of Mueller indexes does not match grid",
      ErrorType::InvalidNumberMuellerIndexRows) {}

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

bool
hpg::is_initialized() noexcept {
  return runtime::impl::is_initialized();
}

bool
hpg::initialize() {
  return initialize(InitArguments());
}

bool
hpg::initialize(const InitArguments& args) {
  return runtime::impl::initialize(args);
}

void
hpg::finalize() {
  runtime::impl::finalize();
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

ScopeGuard::ScopeGuard()
  : init(false) {
  if (!is_initialized()) {
    initialize();
    init = true;
  }
}

ScopeGuard::ScopeGuard(const InitArguments& args)
  : init(false) {
  if (!is_initialized()) {
    initialize(args);
    init = true;
  }
}

ScopeGuard::~ScopeGuard() {
  if (is_initialized() && init)
    finalize();
}


template <typename T>
static rval_t<T>
to_rval(std::variant<Error, T>&& t) {
  if (std::holds_alternative<T>(t))
    return rval<T>(std::get<T>(std::move(t)));
  else
    return rval<T>(std::get<Error>(std::move(t)));
}

#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
const std::array<unsigned, 4> GridderState::default_versions{0, 0, 0, 0};
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS

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

  using namespace runtime;

#ifndef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  std::array<unsigned, 4> implementation_versions{0, 0, 0, 0};
#endif

  const unsigned max_active_tasks = max_added_tasks + 1;

  switch (device) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial:
    impl =
      std::make_shared<StateT<Device::Serial>>(
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
      std::make_shared<StateT<Device::OpenMP>>(
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
      std::make_shared<StateT<Device::Cuda>>(
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

GridderState::GridderState(GridderState&& h) noexcept {
  *this = std::move(h);
}

GridderState&
GridderState::operator=(const GridderState& rhs) {

  using namespace runtime;

  const GridderState& crhs = const_cast<const GridderState&>(rhs);
  switch (crhs.impl->m_device) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial:
    impl =
      std::make_shared<StateT<Device::Serial>>(
        dynamic_cast<StateT<Device::Serial>*>(crhs.impl.get())
        ->copy());
    break;
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
  case Device::OpenMP:
    impl =
      std::make_shared<StateT<Device::OpenMP>>(
        dynamic_cast<StateT<Device::OpenMP>*>(crhs.impl.get())
        ->copy());
    break;
#endif // HPG_ENABLE_OPENMP
#ifdef HPG_ENABLE_CUDA
  case Device::Cuda:
    impl =
      std::make_shared<StateT<Device::Cuda>>(
        dynamic_cast<StateT<Device::Cuda>*>(crhs.impl.get())
        ->copy());
    break;
#endif // HPG_ENABLE_CUDA
  default:
    assert(false);
    break;
  }
  return *this;
}

GridderState&
GridderState::operator=(GridderState&& rhs) noexcept {

  impl = std::move(rhs).impl;
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

  ProfileRegion region("GridderState::convolution_function_region_size");

  return impl->convolution_function_region_size(shape);
}

rval_t<GridderState>
GridderState::allocate_convolution_function_region(
  const CFArrayShape* shape) const & {

  ProfileRegion
    region("GridderState::allocate_convolution_function_region_const");

  return
    to_rval(
      runtime::GridderState::allocate_convolution_function_region(
        *this,
        shape));
}

rval_t<GridderState>
GridderState::allocate_convolution_function_region(const CFArrayShape* shape)
  && {

  ProfileRegion region("GridderState::allocate_convolution_function_region");

  return
    to_rval(
      runtime::GridderState::allocate_convolution_function_region(
        std::move(*this),
        shape));
}

rval_t<GridderState>
GridderState::set_convolution_function(Device host_device, CFArray&& cf)
  const & {

  ProfileRegion region("GridderState::set_convolution_function_const");

  return
    to_rval(
      runtime::GridderState::set_convolution_function(
        *this,
        host_device,
        std::move(cf)));
}

rval_t<GridderState>
GridderState::set_convolution_function(
  Device host_device,
  CFArray&& cf) && {

  ProfileRegion region("GridderState::set_convolution_function");

  return
    to_rval(
      runtime::GridderState::set_convolution_function(
        std::move(*this),
        host_device,
        std::move(cf)));
}

rval_t<GridderState>
GridderState::set_model(Device host_device, GridValueArray&& gv)
  const & {

  ProfileRegion region("GridderState::set_model_const");

  return
    to_rval(
      runtime::GridderState::set_model(*this, host_device, std::move(gv)));
}

rval_t<GridderState>
GridderState::set_model(Device host_device, GridValueArray&& gv) && {

  ProfileRegion region("GridderState::set_model");

  return
    to_rval(
      runtime::GridderState::set_model(
        std::move(*this),
        host_device,
        std::move(gv)));
}

rval_t<std::tuple<GridderState, future<VisDataVector>>>
GridderState::grid_visibilities_base(
  Device host_device,
  VisDataVector&& visibilities,
  bool update_grid_weights,
  bool do_degrid,
  bool return_visibilities,
  bool do_grid) const & {

  ProfileRegion region("GridderState::grid_visibilities_base_const");

  return
    to_rval(
      runtime::GridderState::grid_visibilities(
        *this,
        host_device,
        std::move(visibilities),
        update_grid_weights,
        do_degrid,
        return_visibilities,
        do_grid));
}

rval_t<std::tuple<GridderState, future<VisDataVector>>>
GridderState::grid_visibilities_base(
  Device host_device,
  VisDataVector&& visibilities,
  bool update_grid_weights,
  bool do_degrid,
  bool return_visibilities,
  bool do_grid) && {

  ProfileRegion region("GridderState::grid_visibilities_base");

  return
    to_rval(
      runtime::GridderState::grid_visibilities(
        std::move(*this),
        std::move(host_device),
        std::move(visibilities),
        update_grid_weights,
        do_degrid,
        return_visibilities,
        do_grid));
}

rval_t<GridderState>
GridderState::grid_visibilities(
  Device host_device,
  VisDataVector&& visibilities,
  bool update_grid_weights) const & {

  ProfileRegion region("GridderState::grid_visibilities_const");

  return
    map(
      grid_visibilities_base(
        host_device,
        VisDataVector(std::move(visibilities)),
        update_grid_weights,
        false, // do_degrid
        false, // return_visibilities
        true), // do_grid
      [](auto&& gs_fvs) {
        return std::get<0>(std::move(gs_fvs));
      });
};

rval_t<GridderState>
GridderState::grid_visibilities(
  Device host_device,
  VisDataVector&& visibilities,
  bool update_grid_weights) && {

  ProfileRegion region("GridderState::grid_visibilities");

  return
    map(
      std::move(*this).grid_visibilities_base(
        host_device,
        std::move(visibilities),
        update_grid_weights,
        false, // do_degrid
        false, // return_visibilities
        true), // do_grid
      [](auto&& gs_fvs) {
        return std::get<0>(std::move(gs_fvs));
      });
};

rval_t<GridderState>
GridderState::degrid_grid_visibilities(
  Device host_device,
  VisDataVector&& visibilities,
  bool update_grid_weights) const & {

  ProfileRegion region("GridderState::degrid_grid_visibilities_const");

  return
    map(
      grid_visibilities_base(
        host_device,
        std::move(visibilities),
        update_grid_weights,
        true, // do_degrid
        false, // return_visibilities
        true), // do_grid
      [](auto&& gs_fvs) {
        return std::get<0>(std::move(gs_fvs));
      });
};

rval_t<GridderState>
GridderState::degrid_grid_visibilities(
  Device host_device,
  VisDataVector&& visibilities,
  bool update_grid_weights) && {

  ProfileRegion region("GridderState::degrid_grid_visibilities");

  return
    map(
      std::move(*this).grid_visibilities_base(
        host_device,
        std::move(visibilities),
        update_grid_weights,
        true, // do_degrid
        false, // return_visibilities
        true), // do_grid
      [](auto&& gs_fvs) {
        return std::get<0>(std::move(gs_fvs));
      });
};

rval_t<std::tuple<GridderState, future<VisDataVector>>>
GridderState::degrid_get_predicted_visibilities(
  Device host_device,
  VisDataVector&& visibilities) const & {

  ProfileRegion
    region("GridderState::degrid_get_predicted_visibilities_const");

  return
    grid_visibilities_base(
      host_device,
      std::move(visibilities),
      false,  // update_grid_weights
      true,   // do_degrid
      true,   // return_visibilities
      false); // do_grid
};

rval_t<std::tuple<GridderState, future<VisDataVector>>>
GridderState::degrid_get_predicted_visibilities(
  Device host_device,
  VisDataVector&& visibilities) && {

  ProfileRegion region("GridderState::degrid_get_predicted_visibilities");

  return
    std::move(*this).grid_visibilities_base(
      host_device,
      std::move(visibilities),
      false,  // update_grid_weights
      true,   // do_degrid
      true,   // return_visibilities
      false); // do_grid
};

rval_t<std::tuple<GridderState, future<VisDataVector>>>
GridderState::degrid_grid_get_residual_visibilities(
  Device host_device,
  VisDataVector&& visibilities,
  bool update_grid_weights) const & {

  ProfileRegion
    region("GridderState::degrid_grid_get_residual_visibilities_const");

  return
    grid_visibilities_base(
      host_device,
      std::move(visibilities),
      update_grid_weights,
      true,  // do_degrid
      true,  // return_visibilities
      true); // do_grid
};

rval_t<std::tuple<GridderState, future<VisDataVector>>>
GridderState::degrid_grid_get_residual_visibilities(
  Device host_device,
  VisDataVector&& visibilities,
  bool update_grid_weights) && {

  ProfileRegion
    region("GridderState::degrid_grid_get_residual_visibilities");

  return
    std::move(*this).grid_visibilities_base(
      host_device,
      std::move(visibilities),
      update_grid_weights,
      true,  // do_degrid
      true,  // return_visibilities
      true); // do_grid
};

GridderState
GridderState::fence() const & {

  ProfileRegion region("GridderState::fence_const");

  GridderState result(*this);
  result.impl->fence();
  return result;
}

GridderState
GridderState::fence() && {

  ProfileRegion region("GridderState::fence");

  GridderState result(std::move(*this));
  result.impl->fence();
  return result;
}

std::tuple<GridderState, std::unique_ptr<GridWeightArray>>
GridderState::grid_weights() const & {

  ProfileRegion region("GridderState::grid_weights_const");

  GridderState result(*this);
  return {std::move(result), std::move(result.impl->grid_weights())};
}

std::tuple<GridderState, std::unique_ptr<GridWeightArray>>
GridderState::grid_weights() && {

  ProfileRegion region("GridderState::grid_weights");

  GridderState result(std::move(*this));
  return {std::move(result), std::move(result.impl->grid_weights())};
}

std::shared_ptr<GridWeightArray::value_type>
GridderState::grid_weights_ptr() const & {

  ProfileRegion region("GridderState::grid_weights_ptr");

  return impl->grid_weights_ptr();
}

size_t
GridderState::grid_weights_span() const & {

  ProfileRegion region("GridderState::grid_weights_span");

  return impl->grid_weights_span();
}

std::tuple<GridderState, std::unique_ptr<GridValueArray>>
GridderState::grid_values() const & {

  ProfileRegion region("GridderState::grid_values_const");

  GridderState result(*this);
  return {std::move(result), std::move(result.impl->grid_values())};
}

std::tuple<GridderState, std::unique_ptr<GridValueArray>>
GridderState::grid_values() && {

  ProfileRegion region("GridderState::grid_values");

  GridderState result(std::move(*this));
  return {std::move(result), std::move(result.impl->grid_values())};
}

std::shared_ptr<GridValueArray::value_type>
GridderState::grid_values_ptr() const & {

  ProfileRegion region("GridderState::grid_values_ptr");

  return impl->grid_values_ptr();
}

size_t
GridderState::grid_values_span() const & {

  ProfileRegion region("GridderState::grid_values_span");

  return impl->grid_values_span();
}

std::tuple<GridderState, std::unique_ptr<GridValueArray>>
GridderState::model_values() const & {

  ProfileRegion region("GridderState::model_values_const");

  GridderState result(*this);
  return {std::move(result), std::move(result.impl->model_values())};
}

std::tuple<GridderState, std::unique_ptr<GridValueArray>>
GridderState::model_values() && {

  ProfileRegion region("GridderState::model_values");

  GridderState result(std::move(*this));
  return {std::move(result), std::move(result.impl->model_values())};
}

std::shared_ptr<GridValueArray::value_type>
GridderState::model_values_ptr() const & {

  ProfileRegion region("GridderState::model_values_ptr");

  return impl->model_values_ptr();
}

size_t
GridderState::model_values_span() const & {

  ProfileRegion region("GridderState::grid_values_span");

  return impl->model_values_span();
}

GridderState
GridderState::reset_grid() const & {

  ProfileRegion region("GridderState::reset_grid_const");

  GridderState result(*this);
  result.impl->reset_grid();
  return result;
}

GridderState
GridderState::reset_grid() && {

  ProfileRegion region("GridderState::reset_grid");

  GridderState result(std::move(*this));
  result.impl->reset_grid();
  return result;
}

GridderState
GridderState::reset_model() const & {

  ProfileRegion region("GridderState::reset_model_const");

  GridderState result(*this);
  result.impl->reset_model();
  return result;
}

GridderState
GridderState::reset_model() && {

  ProfileRegion region("GridderState::reset_model");

  GridderState result(std::move(*this));
  result.impl->reset_model();
  return result;
}

GridderState
GridderState::normalize_by_weights(grid_value_fp wfactor) const & {

  ProfileRegion region("GridderState::normalize_by_weights_const");

  GridderState result(*this);
  result.impl->normalize_by_weights(wfactor);
  return result;
}

GridderState
GridderState::normalize_by_weights(grid_value_fp wfactor) && {

  ProfileRegion region("GridderState::normalize_by_weights");

  GridderState result(std::move(*this));
  result.impl->normalize_by_weights(wfactor);
  return result;
}

rval_t<GridderState>
GridderState::apply_grid_fft(
  grid_value_fp norm,
  FFTSign sign,
  bool in_place) const & {

  ProfileRegion region("GridderState::apply_grid_fft_const");

  return
    to_rval(runtime::GridderState::apply_grid_fft(*this, norm, sign, in_place));
}

rval_t<GridderState>
GridderState::apply_grid_fft(
  grid_value_fp norm,
  FFTSign sign,
  bool in_place) && {

  ProfileRegion region("GridderState::apply_grid_fft");

  return
    to_rval(
      runtime::GridderState::apply_grid_fft(
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

  ProfileRegion region("GridderState::apply_model_fft_const");

  return
  to_rval(runtime::GridderState::apply_model_fft(*this, norm, sign, in_place));
}

rval_t<GridderState>
GridderState::apply_model_fft(
  grid_value_fp norm,
  FFTSign sign,
  bool in_place) && {

  ProfileRegion region("GridderState::apply_model_fft");

  return
    to_rval(
      runtime::GridderState::apply_model_fft(
        std::move(*this),
        norm,
        sign,
        in_place));
}

GridderState
GridderState::shift_grid(ShiftDirection direction) const & {

  ProfileRegion region("GridderState::shift_grid_const");

  GridderState result(*this);
  result.impl->shift_grid(direction);
  return result;
}

GridderState
GridderState::shift_grid(ShiftDirection direction) && {

  ProfileRegion region("GridderState::shift_grid");

  GridderState result(std::move(*this));
  result.impl->shift_grid(direction);
  return result;
}

GridderState
GridderState::shift_model(ShiftDirection direction) const & {

  ProfileRegion region("GridderState::shift_model_const");

  GridderState result(*this);
  result.impl->shift_model(direction);
  return result;
}

GridderState
GridderState::shift_model(ShiftDirection direction) && {

  ProfileRegion region("GridderState::shift_model");

  GridderState result(std::move(*this));
  result.impl->shift_model(direction);
  return result;
}

void
GridderState::swap(GridderState& other) noexcept {
  std::swap(impl, other.impl);
}

template <>
future<std::vector<VisData<1>>>
GridderState::future_visibilities_narrow(future<VisDataVector>&& fvs) {

  return
    std::move(fvs).map<std::vector<VisData<1>>>(
      [](VisDataVector&& vs) {
        assert(vs.m_npol == 1);
        return std::move(*vs.m_v1);
      });
}

template <>
future<std::vector<VisData<2>>>
GridderState::future_visibilities_narrow(future<VisDataVector>&& fvs) {

  return
    std::move(fvs).map<std::vector<VisData<2>>>(
      [](VisDataVector&& vs) {
        assert(vs.m_npol == 2);
        return std::move(*vs.m_v2);
      });
}

template <>
future<std::vector<VisData<3>>>
GridderState::future_visibilities_narrow(future<VisDataVector>&& fvs) {

  return
    std::move(fvs).map<std::vector<VisData<3>>>(
      [](VisDataVector&& vs) {
        assert(vs.m_npol == 3);
        return std::move(*vs.m_v3);
      });
}

template <>
future<std::vector<VisData<4>>>
GridderState::future_visibilities_narrow(future<VisDataVector>&& fvs) {

  return
    std::move(fvs).map<std::vector<VisData<4>>>(
      [](VisDataVector&& vs) {
        assert(vs.m_npol == 4);
        return std::move(*vs.m_v4);
      });
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
  IArrayVector&& conjugate_mueller_indexes
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  , const std::array<unsigned, 4>& implementation_versions
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
)
  : state(
    GridderState(
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
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
      )) {}

Gridder::Gridder(const Gridder& other)
  : state(other.state) {}

Gridder::Gridder(Gridder&& other) noexcept
  : state(std::move(other).state) {}

Gridder::Gridder(GridderState&& st) noexcept
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
Gridder::operator=(Gridder&& rhs) noexcept {
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

opt_t<Error>
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
  std::shared_ptr<Error> result;
  std::tie(result, state) =
    std::move(state).allocate_convolution_function_region(shape);
  return result;
#endif //HPG_API >= 17
}

opt_t<Error>
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
  std::shared_ptr<Error> result;
  std::tie(result, state) =
    std::move(state).set_convolution_function(host_device, std::move(cf));
  return result;
#endif //HPG_API >= 17
}

opt_t<Error>
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
  std::shared_ptr<Error> result;
  std::tie(result, state) =
    std::move(state).set_model(host_device, std::move(gv));
  return result;
#endif //HPG_API >= 17
}

opt_t<Error>
Gridder::grid_visibilities(
  Device host_device,
  VisDataVector&& visibilities,
  bool update_grid_weights) {
#if HPG_API >= 17
  return
    fold(
      std::move(state)
      .grid_visibilities(
        host_device,
        std::move(visibilities),
        update_grid_weights),
      [this](auto&& gs) -> std::optional<Error> {
        this->state = std::move(gs);
        return std::nullopt;
      },
      [](auto&& err) -> std::optional<Error> {
        return std::move(err);
      });
#else // HPG_API < 17
  auto [err, gs] =
    std::move(state)
    .grid_visibilities(
      host_device,
      std::move(visibilities),
      update_grid_weights);
  if (!err)
    state = std::move(gs);
  return std::move(err);
#endif // HPG_API >= 17
}

opt_t<Error>
Gridder::degrid_grid_visibilities(
  Device host_device,
  VisDataVector&& visibilities,
  bool update_grid_weights) {
#if HPG_API >= 17
  return
    fold(
      std::move(state)
      .degrid_grid_visibilities(
        host_device,
        std::move(visibilities),
        update_grid_weights),
      [this](auto&& gs) -> std::optional<Error> {
        this->state = std::move(gs);
        return std::nullopt;
      },
      [](auto&& err) -> std::optional<Error> {
        return std::move(err);
      });
#else // HPG_API < 17
  auto [err, gs] =
    std::move(state)
    .degrid_grid_visibilities(
      host_device,
      std::move(visibilities),
      update_grid_weights);
  if (!err)
    state = std::move(gs);
  return std::move(err);
#endif // HPG_API >= 17
}

rval_t<future<VisDataVector>>
Gridder::degrid_get_predicted_visibilities(
  Device host_device,
  VisDataVector&& visibilities) {
#if HPG_API >= 17
  return
    fold(
      std::move(state)
      .degrid_get_predicted_visibilities(host_device, std::move(visibilities)),
      [this](auto&& gs_fvs) -> rval_t<future<VisDataVector>> {
        this->state = std::get<0>(std::move(gs_fvs));
        return std::get<1>(std::move(gs_fvs));
      },
      [](auto&& err) -> rval_t<future<VisDataVector>> {
        return std::move(err);
      });
#else // HPG_API < 17
  auto [err, gs_fvs] =
    std::move(state)
    .degrid_get_predicted_visibilities(host_device, std::move(visibilities));
  if (!err) {
    state = std::get<0>(std::move(gs_fvs));
    return rval<future<VisDataVector>>(std::get<1>(std::move(gs_fvs)));
  }
  return rval<future<VisDataVector>>(std::move(*err));
#endif // HPG_API >= 17
}

rval_t<future<VisDataVector>>
Gridder::degrid_grid_get_residual_visibilities(
  Device host_device,
  VisDataVector&& visibilities,
  bool update_grid_weights) {
#if HPG_API >= 17
  return
    fold(
      std::move(state).degrid_grid_get_residual_visibilities(
        host_device,
        std::move(visibilities),
        update_grid_weights),
      [this](auto&& gs_fvs) -> rval_t<future<VisDataVector>> {
        this->state = std::get<0>(std::move(gs_fvs));
        return std::get<1>(std::move(gs_fvs));
      },
      [](auto&& err) -> rval_t<future<VisDataVector>> {
        return std::move(err);
      });
#else // HPG_API < 17
  auto [err, gs_fvs] =
    std::move(state).degrid_grid_get_residual_visibilities(
      host_device,
      std::move(visibilities),
      update_grid_weights);
  if (!err) {
    state = std::get<0>(std::move(gs_fvs));
    return rval<future<VisDataVector>>(std::get<1>(std::move(gs_fvs)));
  }
  return rval<future<VisDataVector>>(std::move(*err));
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

std::shared_ptr<GridWeightArray::value_type>
Gridder::grid_weights_ptr() const & {
  return state.grid_weights_ptr();
}

size_t
Gridder::grid_weights_span() const & {
  return state.grid_weights_span();
}

std::unique_ptr<GridValueArray>
Gridder::grid_values() const {
  std::unique_ptr<GridValueArray> result;
  std::tie(const_cast<Gridder*>(this)->state, result) =
    std::move(const_cast<Gridder*>(this)->state).grid_values();
  return result;
}

std::shared_ptr<GridValueArray::value_type>
Gridder::grid_values_ptr() const & {
  return state.grid_values_ptr();
}

size_t
Gridder::grid_values_span() const & {
  return state.grid_values_span();
}

std::unique_ptr<GridValueArray>
Gridder::model_values() const {
  std::unique_ptr<GridValueArray> result;
  std::tie(const_cast<Gridder*>(this)->state, result) =
    std::move(const_cast<Gridder*>(this)->state).model_values();
  return result;
}

std::shared_ptr<GridValueArray::value_type>
Gridder::model_values_ptr() const & {
  return state.model_values_ptr();
}

size_t
Gridder::model_values_span() const & {
  return state.model_values_span();
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
Gridder::normalize_by_weights(grid_value_fp wgt_factor) {
  state = std::move(state).normalize_by_weights(wgt_factor);
}

opt_t<Error>
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
  std::shared_ptr<Error> result;
  std::tie(result, state) =
    std::move(state).apply_grid_fft(norm, sign, in_place);
  return result;
#endif //HPG_API >= 17
}

opt_t<Error>
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
  std::shared_ptr<Error> result;
  std::tie(result, state) =
    std::move(state).apply_model_fft(norm, sign, in_place);
  return result;
#endif //HPG_API >= 17
}

void
Gridder::shift_grid(ShiftDirection direction) {
  state = std::move(state).shift_grid(direction);
}

void
Gridder::shift_model(ShiftDirection direction) {
  state = std::move(state).shift_model(direction);
}

opt_t<Error>
GridValueArray::copy_to(Device host_device, value_type* dst, Layout layout)
  const {

  using namespace runtime;

  static_assert(
    int(impl::core::GridAxis::x) == GridValueArray::Axis::x
    && int(impl::core::GridAxis::y) == GridValueArray::Axis::y
    && int(impl::core::GridAxis::mrow) == GridValueArray::Axis::mrow
    && int(impl::core::GridAxis::cube) == GridValueArray::Axis::cube);

#if HPG_API >= 17
  if (host_devices().count(host_device) == 0)
    return DisabledHostDeviceError();
  unsafe_copy_to(host_device, dst, layout);
  return std::nullopt;
#else // HPG_API < 17
  if (host_devices().count(host_device) == 0)
    return std::shared_ptr<Error>(new DisabledHostDeviceError());
  unsafe_copy_to(host_device, dst, layout);
  return nullptr;
#endif //HPG_API >= 17
}

std::unique_ptr<GridValueArray>
GridValueArray::copy_from(
  const std::string& name,
  Device target_device,
  Device host_device,
  const value_type* src,
  const std::array<unsigned, GridValueArray::rank>& extents,
  Layout layout) {

  using namespace runtime;

  static_assert(
    int(impl::core::GridAxis::x) == GridValueArray::Axis::x
    && int(impl::core::GridAxis::y) == GridValueArray::Axis::y
    && int(impl::core::GridAxis::mrow) == GridValueArray::Axis::mrow
    && int(impl::core::GridAxis::cube) == GridValueArray::Axis::cube);

  switch (target_device) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial:
    return
      impl::GridValueViewArray<Device::Serial>::copy_from(
        name,
        host_device,
        src,
        extents,
        layout);
    break;
#endif
#ifdef HPG_ENABLE_OPENMP
  case Device::OpenMP:
    return
      impl::GridValueViewArray<Device::OpenMP>::copy_from(
        name,
        host_device,
        src,
        extents,
        layout);
    break;
#endif
#ifdef HPG_ENABLE_CUDA
  case Device::Cuda:
    return
      impl::GridValueViewArray<Device::Cuda>::copy_from(
        name,
        host_device,
        src,
        extents,
        layout);
    break;
#endif
  default:
    assert(false);
    return nullptr;
    break;
  }
}

opt_t<Error>
GridWeightArray::copy_to(Device host_device, value_type* dst, Layout layout)
  const {

  using namespace runtime;

  static_assert(
    int(impl::core::GridAxis::x) == GridValueArray::Axis::x
    && int(impl::core::GridAxis::y) == GridValueArray::Axis::y
    && int(impl::core::GridAxis::mrow) == GridValueArray::Axis::mrow
    && int(impl::core::GridAxis::cube) == GridValueArray::Axis::cube);

#if HPG_API >= 17
  if (host_devices().count(host_device) == 0)
    return DisabledHostDeviceError();
  unsafe_copy_to(host_device, dst, layout);
  return std::nullopt;
#else // HPG_API < 17
  if (host_devices().count(host_device) == 0)
    return std::shared_ptr<Error>(new DisabledHostDeviceError());
  unsafe_copy_to(host_device, dst, layout);
  return nullptr;
#endif //HPG_API >= 17
}


std::unique_ptr<GridWeightArray>
GridWeightArray::copy_from(
  const std::string& name,
  Device target_device,
  Device host_device,
  const value_type* src,
  const std::array<unsigned, GridWeightArray::rank>& extents,
  Layout layout) {

  using namespace runtime;

  switch (target_device) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial:
    return
      impl::GridWeightViewArray<Device::Serial>::copy_from(
        name,
        host_device,
        src,
        extents,
        layout);
    break;
#endif
#ifdef HPG_ENABLE_OPENMP
  case Device::OpenMP:
    return
      impl::GridWeightViewArray<Device::OpenMP>::copy_from(
        name,
        host_device,
        src,
        extents,
        layout);
    break;
#endif
#ifdef HPG_ENABLE_CUDA
  case Device::Cuda:
    return
      impl::GridWeightViewArray<Device::Cuda>::copy_from(
        name,
        host_device,
        src,
        extents,
        layout);
    break;
#endif
  default:
    assert(false);
    return nullptr;
    break;
  }
}

const char * const
hpg::cf_layout_unspecified_version = "";

rval_t<std::string>
CFArray::copy_to(
  Device device,
  Device host_device,
  unsigned grp,
  value_type* dst) const {

  using namespace runtime;

  if (host_devices().count(host_device) == 0)
    return rval<std::string>(DisabledHostDeviceError());

  if (devices().count(device) == 0)
    return rval<std::string>(DisabledDeviceError());

  switch (device) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial:
    impl::layout_for_device<Device::Serial>(host_device, *this, grp, dst);
    break;
#endif
#ifdef HPG_ENABLE_OPENMP
  case Device::OpenMP:
    impl::layout_for_device<Device::OpenMP>(host_device, *this, grp, dst);
    break;
#endif
#ifdef HPG_ENABLE_CUDA
  case Device::Cuda:
    impl::layout_for_device<Device::Cuda>(host_device, *this, grp, dst);
    break;
#endif
  default:
    assert(false);
    break;
  }
  return
    rval(
      impl::construct_cf_layout_version(
        impl::cf_layout_version_number,
        device));
}

rval_t<size_t>
CFArray::min_buffer_size(Device device, unsigned grp) const {

  return runtime::impl::min_cf_buffer_size(device, *this, grp);
}

rval_t<std::unique_ptr<DeviceCFArray>>
DeviceCFArray::create(
  const std::string& layout,
  unsigned oversampling,
  std::vector<
    std::tuple<std::array<unsigned, rank - 1>, std::vector<value_type>>>&&
    arrays) {

  using namespace runtime;

  auto opt_vn_dev = impl::parsed_cf_layout_version(layout);
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
        std::make_unique<impl::DeviceCFArray<Device::Serial>>(
          layout,
          oversampling,
          std::move(arrays)));
    break;
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
  case Device::OpenMP:
    return
      rval<std::unique_ptr<DeviceCFArray>>(
        std::make_unique<impl::DeviceCFArray<Device::OpenMP>>(
          layout,
          oversampling,
          std::move(arrays)));
    break;
#endif // HPG_ENABLE_OPENMP
#ifdef HPG_ENABLE_CUDA
  case Device::Cuda:
    return
      rval<std::unique_ptr<DeviceCFArray>>(
        std::make_unique<impl::DeviceCFArray<Device::Cuda>>(
          layout,
          oversampling,
          std::move(arrays)));
    break;
#endif //HPG_ENABLE_CUDA
  default:
    return rval<std::unique_ptr<DeviceCFArray>>(DisabledDeviceError());
    break;
  }
}

rval_t<std::unique_ptr<RWDeviceCFArray>>
RWDeviceCFArray::create(Device device, const CFArrayShape& shape) {

  using namespace runtime;

  switch (device) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial:
    return
      rval<std::unique_ptr<RWDeviceCFArray>>(
        std::make_unique<impl::DeviceCFArray<Device::Serial>>(shape));
    break;
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
  case Device::OpenMP:
    return
      rval<std::unique_ptr<RWDeviceCFArray>>(
        std::make_unique<impl::DeviceCFArray<Device::OpenMP>>(shape));
    break;
#endif // HPG_ENABLE_OPENMP
#ifdef HPG_ENABLE_CUDA
  case Device::Cuda:
    return
      rval<std::unique_ptr<RWDeviceCFArray>>(
        std::make_unique<impl::DeviceCFArray<Device::Cuda>>(shape));
    break;
#endif //HPG_ENABLE_CUDA
  default:
    return rval<std::unique_ptr<RWDeviceCFArray>>(DisabledDeviceError());
    break;
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
