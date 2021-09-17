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
#pragma once

#include "hpg.hpp"
#include "hpg_core.hpp"
#include "hpg_layouts.hpp"

#include <algorithm>
#include <any>
#include <cassert>
#include <cfenv>
#include <deque>
#include <memory>
#include <mutex>
#include <type_traits>
#include <variant>
#include <vector>

#ifndef NDEBUG
# include <iostream>
#endif

#if defined(HPG_ENABLE_SERIAL) || defined(HPG_ENABLE_OPENMP)
# include <fftw3.h>
#endif

/** helper type for std::visit */
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
/** explicit deduction guide (not needed as of C++20) */
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

/** @file hpg_impl.hpp
 *
 * HPG implementation header file
 */
namespace hpg {

namespace K = Kokkos;

/** disabled host device error
 *
 * Host device is not enabled by HPG configuration.
 */
struct DisabledHostDeviceError
  : public Error {

  DisabledHostDeviceError()
    : Error(
      "Requested host device is not enabled",
      ErrorType::DisabledHostDevice) {}
};

// TODO: reimplement CFIndex bounds checking
// struct OutOfBoundsCFIndexError
//   : public Error {

//   OutOfBoundsCFIndexError(const vis_cf_index_t& idx)
//     : Error(
//       "vis_cf_index_t value (" + std::to_string(std::get<0>(idx))
//       + "," + std::to_string(std::get<1>(idx))
//       + ") is out of bounds for current CFArray",
//       ErrorType::OutOfBoundsCFIndex) {}
// };

/** invalid model grid size error
 *
 * Dimensions of model grid do not equal those of (visibility) value grid.
 */
struct InvalidModelGridSizeError
  : public Error {

  /** constructor */
  InvalidModelGridSizeError(
    const std::array<unsigned, GridValueArray::rank>& model_size,
    const std::array<unsigned, GridValueArray::rank>& grid_size)
    : Error(
      "model grid size " + sz2str(model_size)
      + " is different from visibility grid size " + sz2str(grid_size),
      ErrorType::InvalidModelGridSize) {}

  /** array extents as string */
  static std::string
  sz2str(const std::array<unsigned, GridValueArray::rank>& sz) {
    std::ostringstream oss;
    oss << "[" << sz[0]
        << "," << sz[1]
        << "," << sz[2]
        << "," << sz[3]
        << "]" << std::endl;
    return oss.str();
  }
};

/** CF support exceeds grid size error
 *
 * CF support exceeds size of grid
 */
struct CFSupportExceedsGridError
  : public Error {

  CFSupportExceedsGridError()
    : Error(
      "CF support size exceeds grid size",
      ErrorType::CFSupportExceedsGrid) {}
};

namespace Impl {

/** type for (plain) vector data */
template <typename T>
using vector_data = std::shared_ptr<std::vector<T>>;

template <typename Layout, typename memory_space>
struct GridWeightPtr
  : public std::enable_shared_from_this<GridWeightPtr<Layout, memory_space>> {

  core::weight_view<Layout, memory_space> m_gw;

  GridWeightPtr(const core::weight_view<Layout, memory_space>& gw)
    : m_gw(gw) {}

  std::shared_ptr<GridWeightArray::value_type>
  ptr() const {
    return
      std::shared_ptr<GridWeightArray::value_type>(
        this->shared_from_this(),
        reinterpret_cast<GridWeightArray::value_type*>(m_gw.data()));
  }

  virtual ~GridWeightPtr() {}
};

template <typename Layout, typename memory_space>
struct GridValuePtr
  : public std::enable_shared_from_this<GridValuePtr<Layout, memory_space>> {

  core::grid_view<Layout, memory_space> m_gv;

  GridValuePtr(const core::grid_view<Layout, memory_space>& gv)
    : m_gv(gv) {}

  std::shared_ptr<GridValueArray::value_type>
  ptr() const {
    return
      std::shared_ptr<GridValueArray::value_type>(
        this->shared_from_this(),
        reinterpret_cast<GridValueArray::value_type*>(m_gv.data()));
  }

  virtual ~GridValuePtr() {}
};

/** abstract base class for state implementations */
struct State {

  Device m_device; /**< device type */
  unsigned m_max_active_tasks; /**< maximum number of active tasks */
  size_t m_max_visibility_batch_size; /**< maximum number of visibilities to
                                         sent to gridding kernel at once */
  std::array<unsigned, 4> m_grid_size; /**< grid size */
  K::Array<grid_scale_fp, 2> m_grid_scale; /**< grid scale */
  unsigned m_num_polarizations; /**< number of visibility polarizations */
  std::array<unsigned, 4> m_implementation_versions; /**< impl versions*/

  using maybe_vis_t =
    std::shared_ptr<std::shared_ptr<std::optional<VisDataVector>>>;

  State(Device device)
    : m_device(device) {}

  State(
    Device device,
    unsigned max_active_tasks,
    size_t max_visibility_batch_size,
    const std::array<unsigned, 4>& grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale,
    unsigned num_polarizations,
    const std::array<unsigned, 4>& implementation_versions)
    : m_device(device)
    , m_max_active_tasks(max_active_tasks)
    , m_max_visibility_batch_size(max_visibility_batch_size)
    , m_grid_size(grid_size)
    , m_grid_scale({grid_scale[0], grid_scale[1]})
    , m_num_polarizations(num_polarizations)
    , m_implementation_versions(implementation_versions) {}

  unsigned
  visibility_gridder_version() const {
    return m_implementation_versions[0];
  }

  unsigned
  grid_normalizer_version() const {
    return m_implementation_versions[1];
  }

  unsigned
  fft_version() const {
    return m_implementation_versions[2];
  }

  unsigned
  grid_shifter_version() const {
    return m_implementation_versions[3];
  }

  virtual size_t
  convolution_function_region_size(const CFArrayShape* shape)
    const noexcept = 0;

  virtual std::optional<Error>
  allocate_convolution_function_region(const CFArrayShape* shape) = 0;

  virtual std::optional<Error>
  set_convolution_function(Device host_device, CFArray&& cf) = 0;

  virtual std::optional<Error>
  set_model(Device host_device, GridValueArray&& gv) = 0;

  virtual std::variant<Error, maybe_vis_t>
  grid_visibilities(
    Device host_device,
    VisDataVector&& visibilities,
    bool update_grid_weights,
    bool do_degrid,
    bool return_visibilities,
    bool do_grid) = 0;

  virtual void
  fence() const noexcept = 0;

  virtual std::unique_ptr<GridWeightArray>
  grid_weights() const = 0;

  virtual std::shared_ptr<GridWeightArray::value_type>
  grid_weights_ptr() const = 0;

  virtual size_t
  grid_weights_span() const = 0;

  virtual std::unique_ptr<GridValueArray>
  grid_values() const = 0;

  virtual std::shared_ptr<GridValueArray::value_type>
  grid_values_ptr() const = 0;

  virtual size_t
  grid_values_span() const = 0;

  virtual void
  reset_grid() = 0;

  virtual std::unique_ptr<GridValueArray>
  model_values() const = 0;

  virtual std::shared_ptr<GridValueArray::value_type>
  model_values_ptr() const = 0;

  virtual size_t
  model_values_span() const = 0;

  virtual void
  reset_model() = 0;

  virtual void
  normalize_by_weights(grid_value_fp wfactor) = 0;

  virtual std::optional<Error>
  apply_grid_fft(grid_value_fp norm, FFTSign sign, bool in_place) = 0;

  virtual std::optional<Error>
  apply_model_fft(grid_value_fp norm, FFTSign sign, bool in_place) = 0;

  virtual void
  shift_grid(ShiftDirection direction) = 0;

  virtual void
  shift_model(ShiftDirection direction) = 0;

  virtual ~State() {}
};

/** concrete sub-class of abstract GridValueArray */
template <Device D>
class HPG_EXPORT GridValueViewArray final
  : public GridValueArray {
public:

  using memory_space = typename core::DeviceT<D>::kokkos_device::memory_space;
  using layout = typename layouts::GridLayout<D>::layout;
  using grid_t = typename core::grid_view<layout, memory_space>::HostMirror;

  grid_t grid;

  GridValueViewArray(const grid_t& grid_)
    : grid(grid_) {}

  virtual ~GridValueViewArray() {}

  unsigned
  extent(unsigned dim) const override {
    return grid.extent(dim);
  }

  static_assert(
    GridValueArray::Axis::x == 0
    && GridValueArray::Axis::y == 1
    && GridValueArray::Axis::mrow == 2
    && GridValueArray::Axis::cube == 3);

  const value_type&
  operator()(unsigned x, unsigned y, unsigned mrow, unsigned cube)
    const override {

    static_assert(
      static_cast<int>(core::GridAxis::x) == 0
      && static_cast<int>(core::GridAxis::y) == 1
      && static_cast<int>(core::GridAxis::mrow) == 2
      && static_cast<int>(core::GridAxis::cube) == 3);
    return reinterpret_cast<const value_type&>(grid(x, y, mrow, cube));
  }

  value_type&
  operator()(unsigned x, unsigned y, unsigned mrow, unsigned cube) override {

    static_assert(
      static_cast<int>(core::GridAxis::x) == 0
      && static_cast<int>(core::GridAxis::y) == 1
      && static_cast<int>(core::GridAxis::mrow) == 2
      && static_cast<int>(core::GridAxis::cube) == 3);
    return reinterpret_cast<value_type&>(grid(x, y, mrow, cube));
  }

  template <Device H>
  void
  copy_to(value_type* dst, Layout lyo) const {

    // we're assuming that a K::LayoutLeft or K::LayoutRight copy has no padding
    // (otherwise, the following is broken, not least because it may result in
    // an out-of-bounds access on dst)

    auto espace = typename core::DeviceT<H>::kokkos_device::execution_space();

    switch (lyo) {
    case Layout::Left: {
      K::View<
        typename grid_t::data_type,
        K::LayoutLeft,
        typename grid_t::memory_space,
        K::MemoryTraits<K::Unmanaged>> dstv(
          reinterpret_cast<typename grid_t::pointer_type>(dst),
          grid.extent(0), grid.extent(1), grid.extent(2), grid.extent(3));
      K::deep_copy(espace, dstv, grid);
      espace.fence();
      break;
    }
    case Layout::Right: {
      K::View<
        typename grid_t::data_type,
        K::LayoutRight,
        typename grid_t::memory_space,
        K::MemoryTraits<K::Unmanaged>> dstv(
          reinterpret_cast<typename grid_t::pointer_type>(dst),
          grid.extent(0), grid.extent(1), grid.extent(2), grid.extent(3));
      K::deep_copy(espace, dstv, grid);
      espace.fence();
      break;
    }
    default:
      assert(false);
      break;
    }
  }

protected:

  void
  unsafe_copy_to(Device host_device, value_type* dst, Layout lyo)
    const override {

    switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
    case Device::Serial:
      copy_to<Device::Serial>(dst, lyo);
      break;
#endif
#ifdef HPG_ENABLE_OPENMP
    case Device::OpenMP:
      copy_to<Device::OpenMP>(dst, lyo);
      break;
#endif
    default:
      assert(false);
      break;
    }
  }

public:

  template <Device H>
  static std::unique_ptr<GridValueViewArray>
  copy_from(
    const std::string& name,
    const value_type* src,
    const std::array<unsigned, rank>& extents,
    Layout lyo) {

    std::array<int, rank> iext{
      static_cast<int>(extents[0]),
      static_cast<int>(extents[1]),
      static_cast<int>(extents[2]),
      static_cast<int>(extents[3])};
    grid_t grid(
      K::ViewAllocateWithoutInitializing(name),
      layouts::GridLayout<D>::dimensions(iext));

    // we're assuming that a K::LayoutLeft or K::LayoutRight copy has no padding
    // (otherwise, the following is broken, not least because it may result in
    // an out-of-bounds access on dst)

    auto espace = typename core::DeviceT<H>::kokkos_device::execution_space();

    switch (lyo) {
    case Layout::Left: {
      K::View<
        typename grid_t::data_type,
        K::LayoutLeft,
        typename grid_t::memory_space,
        K::MemoryTraits<K::Unmanaged>> srcv(
          reinterpret_cast<typename grid_t::pointer_type>(
            const_cast<value_type *>(src)),
          grid.extent(0), grid.extent(1), grid.extent(2), grid.extent(3));
      K::deep_copy(espace, grid, srcv);
      espace.fence();
      break;
    }
    case Layout::Right: {
      K::View<
        typename grid_t::data_type,
        K::LayoutRight,
        typename grid_t::memory_space,
        K::MemoryTraits<K::Unmanaged>> srcv(
          reinterpret_cast<typename grid_t::pointer_type>(
            const_cast<value_type *>(src)),
          grid.extent(0), grid.extent(1), grid.extent(2), grid.extent(3));
      K::deep_copy(espace, grid, srcv);
      espace.fence();
      break;
    }
    default:
      assert(false);
      break;
    }
    return std::make_unique<GridValueViewArray>(grid);
  }

  static std::unique_ptr<GridValueViewArray>
  copy_from(
    const std::string& name,
    Device host_device,
    const value_type* src,
    const std::array<unsigned, rank>& extents,
    Layout lyo) {

    switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
    case Device::Serial:
      return copy_from<Device::Serial>(name, src, extents, lyo);
      break;
#endif
#ifdef HPG_ENABLE_OPENMP
    case Device::OpenMP:
      return copy_from<Device::OpenMP>(name, src, extents, lyo);
      break;
#endif
    default:
      assert(false);
      return nullptr;
      break;
    }
  }
};

/** concrete sub-class of abstract GridWeightArray */
template <Device D>
class HPG_EXPORT GridWeightViewArray final
  : public GridWeightArray {
 public:

  using memory_space = typename core::DeviceT<D>::kokkos_device::memory_space;
  using layout = typename core::DeviceT<D>::kokkos_device::array_layout;
  using weight_t = typename core::weight_view<layout, memory_space>::HostMirror;

  weight_t weight;

  GridWeightViewArray(const weight_t& weight_)
    : weight(weight_) {}

  virtual ~GridWeightViewArray() {}

  unsigned
  extent(unsigned dim) const override {
    return weight.extent(dim);
  }

  const value_type&
  operator()(unsigned mrow, unsigned cube) const override {

    return weight(mrow, cube);
  }

  value_type&
  operator()(unsigned mrow, unsigned cube) override {

    return weight(mrow, cube);
  }

  template <Device H>
  void
  copy_to(value_type* dst, Layout lyo) const {

    // we're assuming that a K::LayoutLeft or K::LayoutRight copy has no padding
    // (otherwise, the following is broken, not least because it may result in
    // an out-of-bounds access on dst)

    auto espace = typename core::DeviceT<H>::kokkos_device::execution_space();

    switch (lyo) {
    case Layout::Left: {
      K::View<
        typename weight_t::data_type,
        K::LayoutLeft,
        typename weight_t::memory_space,
        K::MemoryTraits<K::Unmanaged>> dstv(
          reinterpret_cast<typename weight_t::pointer_type>(dst),
          weight.extent(0), weight.extent(1));
      K::deep_copy(espace, dstv, weight);
      espace.fence();
      break;
    }
    case Layout::Right: {
      K::View<
        typename weight_t::data_type,
        K::LayoutRight,
        typename weight_t::memory_space,
        K::MemoryTraits<K::Unmanaged>> dstv(
          reinterpret_cast<typename weight_t::pointer_type>(dst),
          weight.extent(0), weight.extent(1));
      K::deep_copy(espace, dstv, weight);
      espace.fence();
      break;
    }
    default:
      assert(false);
      break;
    }
  }

protected:

  void
  unsafe_copy_to(Device host_device, value_type* dst, Layout lyo)
    const override {

    switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
    case Device::Serial:
      copy_to<Device::Serial>(dst, lyo);
      break;
#endif
#ifdef HPG_ENABLE_OPENMP
    case Device::OpenMP:
      copy_to<Device::OpenMP>(dst, lyo);
      break;
#endif
    default:
      assert(false);
      break;
    }
  }

public:

  template <Device H>
  static std::unique_ptr<GridWeightViewArray>
  copy_from(
    const std::string& name,
    const value_type* src,
    const std::array<unsigned, rank>& extents,
    Layout lyo) {

    weight_t weight(
      K::ViewAllocateWithoutInitializing(name),
      layout(extents[0], extents[1]));

    // we're assuming that a K::LayoutLeft or K::LayoutRight copy has no padding
    // (otherwise, the following is broken, not least because it may result in
    // an out-of-bounds access on dst)

    auto espace = typename core::DeviceT<H>::kokkos_device::execution_space();

    switch (lyo) {
    case Layout::Left: {
      K::View<
        typename weight_t::data_type,
        K::LayoutLeft,
        typename weight_t::memory_space,
        K::MemoryTraits<K::Unmanaged>> srcv(
          reinterpret_cast<typename weight_t::pointer_type>(
            const_cast<value_type *>(src)),
          weight.extent(0), weight.extent(1));
      K::deep_copy(espace, weight, srcv);
      espace.fence();
      break;
    }
    case Layout::Right: {
      K::View<
        typename weight_t::data_type,
        K::LayoutRight,
        typename weight_t::memory_space,
        K::MemoryTraits<K::Unmanaged>> srcv(
          reinterpret_cast<typename weight_t::pointer_type>(
            const_cast<value_type *>(src)),
          weight.extent(0), weight.extent(1));
      K::deep_copy(espace, weight, srcv);
      espace.fence();
      break;
    }
    default:
      assert(false);
      break;
    }
    return std::make_unique<GridWeightViewArray>(weight);
  }

  static std::unique_ptr<GridWeightViewArray>
  copy_from(
    const std::string& name,
    Device host_device,
    const value_type* src,
    const std::array<unsigned, rank>& extents,
    Layout lyo) {

    switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
    case Device::Serial:
      return copy_from<Device::Serial>(name, src, extents, lyo);
      break;
#endif
#ifdef HPG_ENABLE_OPENMP
    case Device::OpenMP:
      return copy_from<Device::OpenMP>(name, src, extents, lyo);
      break;
#endif
    default:
      assert(false);
      return nullptr;
      break;
    }
  }
};

/** concrete sub-class of abstract GridValueArray for identically zero model
 * grids */
class HPG_EXPORT UnallocatedModelValueArray final
  : public GridValueArray {
public:

  value_type m_zero;

  std::array<unsigned, rank> m_extents;

  UnallocatedModelValueArray(const std::array<unsigned, rank>& extents)
    : m_zero(0)
    , m_extents(extents) {}

  virtual ~UnallocatedModelValueArray() {}

  unsigned
  extent(unsigned dim) const override {
    return m_extents[dim];
  }

  const value_type&
  operator()(unsigned, unsigned, unsigned, unsigned)
    const override {

    return m_zero;
  }

  value_type&
  operator()(unsigned, unsigned, unsigned, unsigned) override {

    return m_zero;
  }

  template <Device H>
  void
  copy_to(value_type* dst, Layout lyo) const {

    // we're assuming that a K::LayoutLeft or K::LayoutRight copy has no padding
    // (otherwise, the following is broken, not least because it may result in
    // an out-of-bounds access on dst)

    auto espace = typename core::DeviceT<H>::kokkos_device::execution_space();

    switch (lyo) {
    case Layout::Left: {
      K::View<
        core::gv_t****,
        K::LayoutLeft,
        typename core::DeviceT<H>::kokkos_device::memory_space,
        K::MemoryTraits<K::Unmanaged>> dstv(
          reinterpret_cast<core::gv_t*>(dst),
          m_extents[0], m_extents[1], m_extents[2], m_extents[3]);
      K::deep_copy(espace, dstv, m_zero);
      espace.fence();
      break;
    }
    case Layout::Right: {
      K::View<
        core::gv_t****,
        K::LayoutRight,
        typename core::DeviceT<H>::kokkos_device::memory_space,
        K::MemoryTraits<K::Unmanaged>> dstv(
          reinterpret_cast<core::gv_t*>(dst),
          m_extents[0], m_extents[1], m_extents[2], m_extents[3]);
      K::deep_copy(espace, dstv, m_zero);
      espace.fence();
      break;
    }
    default:
      assert(false);
      break;
    }
  }

protected:

  void
  unsafe_copy_to(Device host_device, value_type* dst, Layout lyo)
    const override {

    switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
    case Device::Serial:
      copy_to<Device::Serial>(dst, lyo);
      break;
#endif
#ifdef HPG_ENABLE_OPENMP
    case Device::OpenMP:
      copy_to<Device::OpenMP>(dst, lyo);
      break;
#endif
    default:
      assert(false);
      break;
    }
  }
};

/** initialize CF array view from CFArray instance */
template <Device D, typename CFH>
static void
init_cf_host(CFH& cf_h, const CFArray& cf, unsigned grp) {
  static_assert(
    K::SpaceAccessibility<
    typename core::DeviceT<D>::kokkos_device::memory_space,
      K::HostSpace>
    ::accessible);
  static_assert(
    static_cast<int>(core::CFAxis::x_major) == 0
    && static_cast<int>(core::CFAxis::y_major) == 1
    && static_cast<int>(core::CFAxis::mueller) == 2
    && static_cast<int>(core::CFAxis::cube) == 3
    && static_cast<int>(core::CFAxis::x_minor) == 4
    && static_cast<int>(core::CFAxis::y_minor) == 5);
  static_assert(
    CFArray::Axis::x == 0
    && CFArray::Axis::y == 1
    && CFArray::Axis::mueller == 2
    && CFArray::Axis::cube == 3
    && CFArray::Axis::group == 4);

  auto extents = cf.extents(grp);
  auto oversampling = cf.oversampling();
  K::parallel_for(
    "cf_init",
    K::MDRangePolicy<K::Rank<4>, typename core::DeviceT<D>::kokkos_device>(
      {0, 0, 0, 0},
      {static_cast<int>(extents[0]),
       static_cast<int>(extents[1]),
       static_cast<int>(extents[2]),
       static_cast<int>(extents[3])}),
    [&](int i, int j, int mueller, int cube) {
      auto X = i / oversampling;
      auto x = i % oversampling;
      auto Y = j / oversampling;
      auto y = j % oversampling;
      cf_h(X, Y, mueller, cube, x, y) = cf(i, j, mueller, cube, grp);
    });
}

static std::optional<std::tuple<unsigned, std::optional<Device>>>
parsed_cf_layout_version(const std::string& layout) {
  auto dash = layout.find('-');
  std::optional<int> vn;
  if (dash != std::string::npos) {
    try {
      vn = std::stoi(layout.substr(0, dash));
      if (vn.value() < 0)
        vn.reset();
    } catch (...) {}
  }
  if (vn) {
    std::string dev = layout.substr(dash + 1);
#ifdef HPG_ENABLE_SERIAL
    if (dev == core::DeviceT<Device::Serial>::name)
      return
        std::make_tuple(
          static_cast<unsigned>(vn.value()),
          std::optional<Device>(Device::Serial));
#endif
#ifdef HPG_ENABLE_OPENMP
    if (dev == core::DeviceT<Device::OpenMP>::name)
      return
        std::make_tuple(
          static_cast<unsigned>(vn.value()),
          std::optional<Device>(Device::OpenMP));
#endif
#ifdef HPG_ENABLE_CUDA
    if (dev == core::DeviceT<Device::Cuda>::name)
      return
        std::make_tuple(
          static_cast<unsigned>(vn.value()),
          std::optional<Device>(Device::Cuda));
#endif
    return std::make_tuple(static_cast<unsigned>(vn.value()), std::nullopt);
  }
  return std::nullopt;
}

static std::string
construct_cf_layout_version(unsigned vn, Device device) {
  std::ostringstream oss;
  oss << vn << "-";
  switch (device) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial:
    oss << core::DeviceT<Device::Serial>::name;
    break;
#endif
#ifdef HPG_ENABLE_OPENMP
  case Device::OpenMP:
    oss << core::DeviceT<Device::OpenMP>::name;
    break;
#endif
#ifdef HPG_ENABLE_CUDA
  case Device::Cuda:
    oss << core::DeviceT<Device::Cuda>::name;
    break;
#endif
  default:
    assert(false);
    break;
  }
  return oss.str();
}

/** device-specific implementation sub-class of hpg::DeviceCFArray class */
template <Device D>
class DeviceCFArray
  : public hpg::DeviceCFArray {
public:

  // notice layout for device D, but in HostSpace
  using cfd_view_h = core::cf_view<typename layouts::CFLayout<D>::layout, K::HostSpace>;

  /** layout version string */
  std::string m_version;
  /** oversampling factor */
  unsigned m_oversampling;
  /** extents by group */
  std::vector<std::array<unsigned, rank - 1>> m_extents;
  /** buffers in host memory with CF values */
  std::vector<std::vector<value_type>> m_arrays;
  /** Views of host memory buffers */
  std::vector<cfd_view_h> m_views;

  DeviceCFArray(
    const std::string& version,
    unsigned oversampling,
    std::vector<
      std::tuple<std::array<unsigned, rank - 1>, std::vector<value_type>>>&&
      arrays)
    : m_version(version)
    , m_oversampling(oversampling) {

    for (auto&& e_v : arrays) {
      m_extents.push_back(std::get<0>(e_v));
      m_arrays.push_back(std::get<1>(std::move(e_v)));
      m_views.emplace_back(
        reinterpret_cast<core::cf_t*>(m_arrays.back().data()),
        layouts::CFLayout<D>::dimensions(this, m_extents.size() - 1));
    }
  }

  virtual ~DeviceCFArray() {}

  unsigned
  oversampling() const override {
    return m_oversampling;
  }

  unsigned
  num_groups() const override {
    return m_arrays.size();
  }

  std::array<unsigned, rank - 1>
  extents(unsigned grp) const override {
    return m_extents[grp];
  }

  const char*
  layout() const override {
    return m_version.c_str();
  }

  static_assert(
    CFArray::Axis::x == 0
    && CFArray::Axis::y == 1
    && CFArray::Axis::mueller == 2
    && CFArray::Axis::cube == 3
    && CFArray::Axis::group == 4);

  std::complex<cf_fp>
  operator()(
    unsigned x,
    unsigned y,
    unsigned mueller,
    unsigned cube,
    unsigned grp)
    const override {

    static_assert(
      static_cast<int>(core::CFAxis::x_major) == 0
      && static_cast<int>(core::CFAxis::y_major) == 1
      && static_cast<int>(core::CFAxis::mueller) == 2
      && static_cast<int>(core::CFAxis::cube) == 3
      && static_cast<int>(core::CFAxis::x_minor) == 4
      && static_cast<int>(core::CFAxis::y_minor) == 5);
    return
      m_views[grp](
        x / m_oversampling,
        y / m_oversampling,
        mueller,
        cube,
        x % m_oversampling,
        y % m_oversampling);
  }

  Device
  device() const override {
    return D;
  }
};

template <Device D>
static void
layout_for_device(
  Device host_device,
  const CFArray& cf,
  unsigned grp,
  CFArray::value_type* dst) {

  auto layout = layouts::CFLayout<D>::dimensions(&cf, grp);
  typename DeviceCFArray<D>::cfd_view_h
    cfd(reinterpret_cast<core::cf_t*>(dst), layout);
  switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial:
    init_cf_host<Device::Serial>(cfd, cf, grp);
    typename core::DeviceT<Device::Serial>::kokkos_device::execution_space().fence();
    break;
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
  case Device::OpenMP:
    init_cf_host<Device::OpenMP>(cfd, cf, grp);
    typename core::DeviceT<Device::OpenMP>::kokkos_device::execution_space().fence();
    break;
#endif // HPG_ENABLE_SERIAL
  default:
    assert(false);
    break;
  }
}

/** initialize model visibilities view from GridValueArray instance */
template <Device D, typename GVH>
static void
init_model(GVH& gv_h, const GridValueArray& gv) {
  static_assert(
    K::SpaceAccessibility<
      typename core::DeviceT<D>::kokkos_device::memory_space,
      K::HostSpace>
    ::accessible);
  static_assert(
    static_cast<int>(core::GridAxis::x) == 0
    && static_cast<int>(core::GridAxis::y) == 1
    && static_cast<int>(core::GridAxis::mrow) == 2
    && static_cast<int>(core::GridAxis::cube) == 3);
  static_assert(
    GridValueArray::Axis::x == 0
    && GridValueArray::Axis::y == 1
    && GridValueArray::Axis::mrow == 2
    && GridValueArray::Axis::cube == 3);

  K::parallel_for(
    "init_model",
    K::MDRangePolicy<K::Rank<4>, typename core::DeviceT<D>::kokkos_device>(
      {0, 0, 0, 0},
      {static_cast<int>(gv.extent(0)),
       static_cast<int>(gv.extent(1)),
       static_cast<int>(gv.extent(2)),
       static_cast<int>(gv.extent(3))}),
    [&](int x, int y, int mr, int cb) {
      gv_h(x, y, mr, cb) = gv(x, y, mr, cb);
    });
}

/** names for stream states */
enum class StreamPhase {
  PRE_GRIDDING,
  GRIDDING
};

/** formatted output for StreamPhase value */
std::ostream&
operator<<(std::ostream& str, const StreamPhase& ph) {
  switch (ph) {
  case StreamPhase::PRE_GRIDDING:
    str << "PRE_GRIDDING";
    break;
  case StreamPhase::GRIDDING:
    str << "GRIDDING";
    break;
  }
  return str;
}

template <Device D>
struct StateT;

/** memory pool and views therein for elements of a (sparse) CF */
template <Device D>
struct CFPool final {

  using kokkos_device = typename core::DeviceT<D>::kokkos_device;
  using execution_space = typename kokkos_device::execution_space;
  using memory_space = typename execution_space::memory_space;
  using cfd_view = core::cf_view<typename layouts::CFLayout<D>::layout, memory_space>;
  using cfh_view = typename cfd_view::HostMirror;

  StateT<D> *state;
  K::View<core::cf_t*, memory_space> pool;
  unsigned num_cf_groups;
  unsigned max_cf_extent_y;
  K::Array<cfd_view, HPG_MAX_NUM_CF_GROUPS> cf_d; // unmanaged (in pool)
  std::vector<std::any> cf_h;
  K::Array<K::Array<int, 2>, HPG_MAX_NUM_CF_GROUPS> cf_radii;

  CFPool()
    : state(nullptr)
    , num_cf_groups(0)
    , max_cf_extent_y(0) {

    for (size_t i = 0; i < HPG_MAX_NUM_CF_GROUPS; ++i)
      cf_d[i] = cfd_view();
  }

  CFPool(StateT<D>* st)
    : state(st)
    , num_cf_groups(0)
    , max_cf_extent_y(0) {

    for (size_t i = 0; i < HPG_MAX_NUM_CF_GROUPS; ++i)
      cf_d[i] = cfd_view();
  }

  CFPool(const CFPool& other) = delete;

  CFPool(CFPool&& other) noexcept {

    if (other.state)
      other.state->fence();
    std::swap(pool, other.pool);
    std::swap(num_cf_groups, other.num_cf_groups);
    std::swap(max_cf_extent_y, other.max_cf_extent_y);
    std::swap(cf_d, other.cf_d);
    std::swap(cf_h, other.cf_h);
    std::swap(cf_radii, other.cf_radii);
  }

  CFPool&
  operator=(const CFPool& rhs) {

    // TODO: make this exception-safe?
    reset();
    num_cf_groups = rhs.num_cf_groups;
    max_cf_extent_y = rhs.max_cf_extent_y;
    cf_radii = rhs.cf_radii;
    if (rhs.pool.extent(0) > 0) {
      pool =
        decltype(pool)(
          K::ViewAllocateWithoutInitializing("cf"),
          rhs.pool.extent(0));
      // use the rhs execution space for the copy, because otherwise we'd need a
      // fence on that execution space after the copy, and this way we can
      // possibly avoid a fence before the copy
      K::deep_copy(
        rhs.state
        ->m_exec_spaces[
          rhs.state->next_exec_space_unlocked(StreamPhase::PRE_GRIDDING)]
        .space,
        pool,
        rhs.pool);
      rhs.state->fence_unlocked();
      for (size_t i = 0; i < num_cf_groups; ++i) {
        // don't need cf_h, since the previous fence ensures that the copy from
        // host has completed
        cf_d[i] =
          cfd_view(
            pool.data() + (rhs.cf_d[i].data() - rhs.pool.data()),
            rhs.cf_d[i].layout());
      }
    }
    return *this;
  }

  CFPool&
  operator=(CFPool&& rhs) noexcept {

    if (state)
      state->fence();
    if (rhs.state)
      rhs.state->fence();
    std::swap(pool, rhs.pool);
    std::swap(num_cf_groups, rhs.num_cf_groups);
    std::swap(max_cf_extent_y, rhs.max_cf_extent_y);
    std::swap(cf_d, rhs.cf_d);
    std::swap(cf_h, rhs.cf_h);
    std::swap(cf_radii, rhs.cf_radii);
    return *this;
  }

  virtual ~CFPool() {
    reset();
  }

  static size_t
  cf_size(const CFArrayShape* cf, unsigned grp) {
    auto layout = layouts::CFLayout<D>::dimensions(cf, grp);
    // TODO: it would be best to use the following to compute
    // allocation size, but it is not implemented in Kokkos
    // 'auto alloc_sz = cfd_view::required_allocation_size(layout)'
    auto alloc_sz =
      core::cf_view<typename core::DeviceT<D>::kokkos_device::array_layout, memory_space>
      ::required_allocation_size(
        layout.dimension[0],
        layout.dimension[1],
        layout.dimension[2],
        layout.dimension[3],
        layout.dimension[4],
        layout.dimension[5]);
    return ((alloc_sz + (sizeof(core::cf_t) - 1)) / sizeof(core::cf_t));
  }

  static size_t
  pool_size(const CFArrayShape* cf) {
    size_t result = 0;
    if (cf)
      for (unsigned grp = 0; grp < cf->num_groups(); ++grp)
        result += cf_size(cf, grp);
    return result;
  }

  void
  prepare_pool(const CFArrayShape* cf, bool force = false) {
    auto current_pool_size = pool.extent(0);
    auto min_pool = pool_size(cf);
    reset((min_pool > current_pool_size) || force);
    if ((min_pool > current_pool_size) || (force && min_pool > 0))
      pool = decltype(pool)(K::ViewAllocateWithoutInitializing("cf"), min_pool);
  }

  void
  add_cf_group(
    const std::array<unsigned, 2>& radii,
    cfd_view cfd,
    std::any cfh) {

    assert(num_cf_groups < HPG_MAX_NUM_CF_GROUPS);
    cf_d[num_cf_groups] = cfd;
    cf_h.push_back(cfh);
    cf_radii[num_cf_groups] =
      {static_cast<int>(radii[0]), static_cast<int>(radii[1])};
    ++num_cf_groups;
    max_cf_extent_y =
      std::max(
        max_cf_extent_y,
        static_cast<unsigned>(cfd.extent(static_cast<int>(core::CFAxis::y_major))));
  }

  void
  add_host_cfs(Device host_device, execution_space espace, CFArray&& cf_array) {
    prepare_pool(&cf_array);
    num_cf_groups = 0;
    size_t offset = 0;
    for (unsigned grp = 0; grp < cf_array.num_groups(); ++grp) {
      cfd_view cf_init(
        pool.data() + offset,
        layouts::CFLayout<D>::dimensions(&cf_array, grp));
#ifndef NDEBUG
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
#endif // NDEBUG

      typename decltype(cf_init)::HostMirror cf_h;
      switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
      case Device::Serial:
        cf_h = K::create_mirror_view(cf_init);
        init_cf_host<Device::Serial>(cf_h, cf_array, grp);
        K::deep_copy(espace, cf_init, cf_h);
        break;
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
      case Device::OpenMP:
        cf_h = K::create_mirror_view(cf_init);
        init_cf_host<Device::OpenMP>(cf_h, cf_array, grp);
        K::deep_copy(espace, cf_init, cf_h);
        break;
#endif // HPG_ENABLE_SERIAL
      default:
        assert(false);
        break;
      }
      offset += cf_size(&cf_array, grp);
      add_cf_group(cf_array.radii(grp), cf_init, cf_h);
    }
  }

  void
  add_device_cfs(execution_space espace, DeviceCFArray<D>&& cf_array) {
    prepare_pool(&cf_array);
    num_cf_groups = 0;
    size_t offset = 0;
    for (unsigned grp = 0; grp < cf_array.num_groups(); ++grp) {
      cfd_view cf_init(
        pool.data() + offset,
        layouts::CFLayout<D>::dimensions(&cf_array, grp));
#ifndef NDEBUG
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
#endif // NDEBUG

      K::deep_copy(espace, cf_init, cf_array.m_views[grp]);
      offset += cf_size(&cf_array, grp);
      add_cf_group(
        cf_array.radii(grp),
        cf_init,
        std::make_tuple(
          std::move(cf_array.m_arrays[grp]),
          cf_array.m_views[grp]));
    }
  }

  void
  reset(bool free_pool = true) {
    if (state && pool.is_allocated()) {
      if (free_pool)
        pool = decltype(pool)();
      cf_h.clear();
      for (size_t i = 0; i < num_cf_groups; ++i)
        cf_d[i] = cfd_view();
      num_cf_groups = 0;
    }
  }
};

/** container for data and views associated with one stream of an execution
 * space*/
template <Device D>
struct ExecSpace final {
  using kokkos_device = typename core::DeviceT<D>::kokkos_device;
  using execution_space = typename kokkos_device::execution_space;
  using memory_space = typename execution_space::memory_space;

  execution_space space;
  core::visbuff_view<memory_space> visbuff;
  core::gvisbuff_view<memory_space> gvisbuff;
  std::variant<
    core::visdata_view<1, memory_space>,
    core::visdata_view<2, memory_space>,
    core::visdata_view<3, memory_space>,
    core::visdata_view<4, memory_space>> visibilities;
  std::variant<
    core::vector_view<core::VisData<1>>,
    core::vector_view<core::VisData<2>>,
    core::vector_view<core::VisData<3>>,
    core::vector_view<core::VisData<4>>> visibilities_h;
  std::variant<
    std::vector<::hpg::VisData<1>>,
    std::vector<::hpg::VisData<2>>,
    std::vector<::hpg::VisData<3>>,
    std::vector<::hpg::VisData<4>>> vis_vector;
  mutable State::maybe_vis_t vis_promise;
  size_t num_visibilities;

  ExecSpace(execution_space sp)
    : space(sp) {
  }

  ExecSpace(const ExecSpace&) = delete;

  ExecSpace(ExecSpace&& other) noexcept
    : space(std::move(other).space)
    , visbuff(std::move(other).visbuff)
    , gvisbuff(std::move(other).gvisbuff)
    , visibilities(std::move(other).visibilities)
    , visibilities_h(std::move(other).visibilities_h)
    , vis_vector(std::move(other).vis_vector)
    , vis_promise(std::move(other).vis_promise)
    , num_visibilities(std::move(other).num_visibilities) {
  }

  virtual ~ExecSpace() {}

  ExecSpace&
  operator=(ExecSpace&& rhs) noexcept {
    space = std::move(rhs).space;
    visbuff = std::move(rhs).visbuff;
    gvisbuff = std::move(rhs).gvisbuff;
    visibilities = std::move(rhs).visibilities;
    visibilities_h = std::move(rhs).visibilities_h;
    vis_vector = std::move(rhs).vis_vector;
    vis_promise = std::move(rhs).vis_promise;
    num_visibilities = std::move(rhs).num_visibilities;
    return *this;
  }

  template <unsigned N>
  constexpr core::visdata_view<N, memory_space>
  visdata() const {
    return std::get<core::visdata_view<N, memory_space>>(visibilities);
  }

  template <unsigned N>
  size_t
  copy_visibilities_to_device(std::vector<::hpg::VisData<N>>&& in_vis) {

    num_visibilities = in_vis.size();
    if (num_visibilities > 0) {
      visibilities_h =
        core::vector_view<core::VisData<N>>(
          reinterpret_cast<core::VisData<N>*>(in_vis.data()),
          num_visibilities);
      auto hview = std::get<core::vector_view<core::VisData<N>>>(visibilities_h);
      if constexpr (!std::is_same_v<K::HostSpace, memory_space>) {
        visibilities =
          core::visdata_view<N, memory_space>(
            reinterpret_cast<core::VisData<N>*>(visbuff.data()),
            num_visibilities);
        auto dview =
          K::subview(core::VisData<N>(), std::pair((size_t)0, num_visibilities));
        K::deep_copy(space, dview, hview);
      } else {
        visibilities =
          core::visdata_view<N, memory_space>(
            reinterpret_cast<core::VisData<N>*>(&hview(0)),
            num_visibilities);
      }
    }
    vis_vector = std::move(in_vis);
    return num_visibilities;
  }

  State::maybe_vis_t
  copy_visibilities_to_host(bool return_visibilities) const {

    State::maybe_vis_t result;
    if (return_visibilities) {
      vis_promise =
        std::make_shared<std::shared_ptr<std::optional<VisDataVector>>>(
          std::make_shared<std::optional<VisDataVector>>(std::nullopt));
      result = vis_promise;
      std::visit(
        overloaded {
          [this](auto& v) {
            using v_t =
              std::remove_const_t<std::remove_reference_t<decltype(v)>>;
            if constexpr (!std::is_same_v<K::HostSpace, memory_space>) {
              constexpr unsigned N = v_t::value_type::npol;
              if (num_visibilities > 0) {
                auto hview = std::get<core::vector_view<core::VisData<N>>>(visibilities_h);
                auto dview =
                  K::subview(
                    core::VisData<N>(),
                    std::pair((size_t)0, num_visibilities));
                K::deep_copy(space, hview, dview);
              }
            }
          }
        },
        vis_vector);
    }
    return result;
  }

  void
  fence() noexcept {
    space.fence();
    if (vis_promise) {
      std::visit(
        overloaded {
          [this](auto& v) {
            using v_t =
              std::remove_const_t<std::remove_reference_t<decltype(v)>>;
            auto vdv =
              std::make_shared<std::optional<VisDataVector>>(
                VisDataVector(std::get<v_t>(std::move(vis_vector))));
            std::atomic_store(&*vis_promise, vdv);
          }
        },
        vis_vector);
      vis_promise.reset();
    }
  }
};

/** Kokkos state implementation for a device type */
template <Device D>
struct HPG_EXPORT StateT final
  : public State {
public:

  using kokkos_device = typename core::DeviceT<D>::kokkos_device;
  using execution_space = typename kokkos_device::execution_space;
  using memory_space = typename execution_space::memory_space;
  using stream_type = typename core::DeviceT<D>::stream_type;

  core::grid_view<typename layouts::GridLayout<D>::layout, memory_space> m_grid;
  core::weight_view<typename execution_space::array_layout, memory_space> m_weights;
  core::grid_view<typename layouts::GridLayout<D>::layout, memory_space> m_model;
  core::const_mindex_view<memory_space> m_mueller_indexes;
  core::const_mindex_view<memory_space> m_conjugate_mueller_indexes;

  // use multiple execution spaces to support overlap of data copying with
  // computation when possible
  std::vector<std::conditional_t<std::is_void_v<stream_type>, int, stream_type>>
    m_streams;

private:
  mutable std::mutex m_mtx;
  // access to the following members in const methods must be protected by m_mtx
  // (intentionally do not provide any thread safety guarantee outside of const
  // methods!)
  mutable std::vector<ExecSpace<D>> m_exec_spaces;
  mutable std::vector<std::tuple<CFPool<D>, std::optional<int>>> m_cfs;
  mutable std::deque<int> m_exec_space_indexes;
  mutable std::deque<int> m_cf_indexes;
  mutable StreamPhase m_current = StreamPhase::PRE_GRIDDING;

public:
  StateT(
    unsigned max_active_tasks,
    size_t max_visibility_batch_size,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4> grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale,
    const IArrayVector& mueller_indexes,
    const IArrayVector& conjugate_mueller_indexes,
    const std::array<unsigned, 4>& implementation_versions)
    : State(
      D,
      std::min(max_active_tasks, core::DeviceT<D>::active_task_limit),
      max_visibility_batch_size,
      grid_size,
      grid_scale,
      mueller_indexes.m_npol,
      implementation_versions) {

    init_state(init_cf_shape);
    m_mueller_indexes =
      init_mueller("mueller_indexes", mueller_indexes);
    m_conjugate_mueller_indexes =
      init_mueller("conjugate_mueller_indexes", conjugate_mueller_indexes);
    new_grid(true, true);
  }

  StateT(const StateT& st)
    : State(
      D,
      st.m_max_active_tasks,
      st.m_max_visibility_batch_size,
      st.m_grid_size,
      {st.m_grid_scale[0], st.m_grid_scale[1]},
      st.m_num_polarizations,
      st.m_implementation_versions) {

    std::scoped_lock lock(st.m_mtx);
    st.fence_unlocked();
    init_state(&st);
    m_mueller_indexes = st.m_mueller_indexes;
    m_conjugate_mueller_indexes = st.m_conjugate_mueller_indexes;
    new_grid(&st, true);
  }

  StateT(StateT&& st) noexcept
    : State(D) {

    m_max_active_tasks = std::move(st).m_max_active_tasks;
    m_max_visibility_batch_size = std::move(st).m_max_visibility_batch_size;
    m_grid_size = std::move(st).m_grid_size;
    m_grid_scale = std::move(st).m_grid_scale;
    m_num_polarizations = std::move(st).m_num_polarizations;
    m_implementation_versions = std::move(st).m_implementation_versions;

    m_grid = std::move(st).m_grid;
    m_weights = std::move(st).m_weights;
    m_model = std::move(st).m_model;
    m_mueller_indexes = std::move(st).m_mueller_indexes;
    m_conjugate_mueller_indexes = std::move(st).m_conjugate_mueller_indexes;
    m_streams = std::move(st).m_streams;
    m_exec_spaces = std::move(st).m_exec_spaces;
    m_exec_space_indexes = std::move(st).m_exec_space_indexes;
    m_current = std::move(st).m_current;

    m_cf_indexes = std::move(st).m_cf_indexes;
    m_cfs.resize(m_max_active_tasks);
    for (auto& [cf, last] : m_cfs)
      cf.state = this;
    for (size_t i = 0; i < m_max_active_tasks; ++i) {
      auto tmp_cf = std::move(m_cfs[i]);
      std::get<0>(tmp_cf).state = this;
      m_cfs[i] = std::move(st.m_cfs[i]);
      st.m_cfs[i] = std::move(tmp_cf);
    }
  }

  virtual ~StateT() {
    fence();
    m_grid = decltype(m_grid)();
    m_weights = decltype(m_weights)();
    m_model = decltype(m_model)();
    m_mueller_indexes = decltype(m_mueller_indexes)();
    m_conjugate_mueller_indexes = decltype(m_conjugate_mueller_indexes)();
    for (auto& [cf, last] : m_cfs)
      cf.reset();
    m_exec_spaces.clear();
    if constexpr(!std::is_void_v<stream_type>) {
      for (auto& str : m_streams) {
        auto rc = core::DeviceT<D>::destroy_stream(str);
        assert(rc);
      }
    }
  }

  StateT&
  operator=(const StateT& st) {
    StateT tmp(st);
    this->swap(tmp);
    return *this;
  }

  StateT&
  operator=(StateT&& st) noexcept {
    StateT tmp(std::move(st));
    this->swap(tmp);
    return *this;
  }

  StateT
  copy() const {
    return StateT(*this);
  }

  size_t
  convolution_function_region_size(const CFArrayShape* shape)
    const noexcept override {
    std::scoped_lock lock(m_mtx);
    return
      shape
      ? std::get<0>(m_cfs[0]).pool_size(shape)
      : std::get<0>(m_cfs[m_cf_indexes.front()]).pool.extent(0);
  }

  std::optional<Error>
  allocate_convolution_function_region(const CFArrayShape* shape) override {
    fence();
    for (auto& [cf, last] : m_cfs) {
      last = std::nullopt;
      cf.prepare_pool(shape, true);
    }
    return std::nullopt;
  }

  std::optional<Error>
  set_convolution_function(Device host_device, CFArray&& cf_array) override {

    for (unsigned grp = 0; grp < cf_array.num_groups(); ++grp) {
      auto extents = cf_array.extents(grp);
      if ((extents[CFArray::Axis::x] >
           m_grid_size[static_cast<int>(core::GridAxis::x)]
           * cf_array.oversampling())
          || (extents[CFArray::Axis::y] >
              m_grid_size[static_cast<int>(core::GridAxis::y)]
              * cf_array.oversampling()))
        return CFSupportExceedsGridError();
    }

    switch_cf_pool();
    auto& exec = m_exec_spaces[next_exec_space(StreamPhase::PRE_GRIDDING)];
    auto& cf = std::get<0>(m_cfs[m_cf_indexes.front()]);
    try {
      cf.add_device_cfs(
        exec.space,
        std::move(dynamic_cast<DeviceCFArray<D>&&>(cf_array)));
    } catch (const std::bad_cast&) {
      cf.add_host_cfs(host_device, exec.space, std::move(cf_array));
    }
    return std::nullopt;
  }

  std::optional<Error>
  set_model(Device host_device, GridValueArray&& gv) override {
    std::array<unsigned, 4>
      model_sz{gv.extent(0), gv.extent(1), gv.extent(2), gv.extent(3)};
    if (m_grid_size != model_sz)
      return InvalidModelGridSizeError(model_sz, m_grid_size);

    fence();
    auto& exec = m_exec_spaces[next_exec_space(StreamPhase::PRE_GRIDDING)];

    if (!m_model.is_allocated()) {
      std::array<int, 4> ig{
        static_cast<int>(m_grid_size[0]),
        static_cast<int>(m_grid_size[1]),
        static_cast<int>(m_grid_size[2]),
        static_cast<int>(m_grid_size[3])};
      m_model =
        decltype(m_model)(
          K::ViewAllocateWithoutInitializing("model"),
          layouts::GridLayout<D>::dimensions(ig));
    }

    try {
      GridValueViewArray<D> gvv =
        std::move(dynamic_cast<GridValueViewArray<D>&&>(gv));
      K::deep_copy(exec.space, m_model, gvv.grid);
    } catch (const std::bad_cast&) {
      auto model_h = K::create_mirror_view(m_model);
      switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
      case Device::Serial:
        init_model<Device::Serial>(model_h, gv);
        break;
#endif
#ifdef HPG_ENABLE_OPENMP
      case Device::OpenMP:
        init_model<Device::OpenMP>(model_h, gv);
        break;
#endif
      default:
        return DisabledHostDeviceError();
        break;
      }
      K::deep_copy(exec.space, m_model, model_h);
    }
    return std::nullopt;
  }

  template <unsigned N>
  State::maybe_vis_t
  default_grid_visibilities(
    Device /*host_device*/,
    std::vector<::hpg::VisData<N>>&& visibilities,
    bool update_grid_weights,
    bool do_degrid,
    bool return_visibilities,
    bool do_grid) {

    auto& exec_pre = m_exec_spaces[next_exec_space(StreamPhase::PRE_GRIDDING)];
    auto len =
      exec_pre.copy_visibilities_to_device(std::move(visibilities));

    auto& exec_grid = m_exec_spaces[next_exec_space(StreamPhase::GRIDDING)];
    auto& cf = std::get<0>(m_cfs[m_cf_indexes.front()]);
    core::const_grid_view<typename layouts::GridLayout<D>::layout, memory_space> model
      = m_model;
    core::VisibilityGridder<N, execution_space, 0>::kernel(
      exec_grid.space,
      cf.cf_d,
      cf.cf_radii,
      cf.max_cf_extent_y,
      m_mueller_indexes,
      m_conjugate_mueller_indexes,
      update_grid_weights,
      do_degrid,
      do_grid,
      len,
      exec_grid.template visdata<N>(),
      exec_grid.gvisbuff,
      m_grid_scale,
      model,
      m_grid,
      m_weights);
    return exec_grid.copy_visibilities_to_host(return_visibilities);
  }

  template <unsigned N>
  State::maybe_vis_t
  grid_visibilities(
    Device host_device,
    std::vector<::hpg::VisData<N>>&& visibilities,
    bool update_grid_weights,
    bool do_degrid,
    bool return_visibilities,
    bool do_grid) {

// #ifndef NDEBUG
//     for (auto& [cube, supp] : *cf_indexes) {
//       auto& cfpool = std::get<0>(m_cfs[m_cf_indexes.front()]);
//       if ((supp >= cfpool.num_cf_groups)
//           || (cube >= cfpool.cf_d[supp].extent_int(5)))
//         return OutOfBoundsCFIndexError({cube, supp});
//     }
// #endif // NDEBUG

    switch (visibility_gridder_version()) {
    case 0:
      return
        default_grid_visibilities(
          host_device,
          std::move(visibilities),
          update_grid_weights,
          do_degrid,
          return_visibilities,
          do_grid);
      break;
    default:
      assert(false);
      std::abort();
      break;
    }
  }

  std::variant<Error, State::maybe_vis_t>
  grid_visibilities(
    Device host_device,
    VisDataVector&& visibilities,
    bool update_grid_weights,
    bool do_degrid,
    bool return_visibilities,
    bool do_grid)
    override {

    switch (visibilities.m_npol) {
    case 1:
      return
        grid_visibilities(
          host_device,
          std::move(*visibilities.m_v1),
          update_grid_weights,
          do_degrid,
          return_visibilities,
          do_grid);
        break;
    case 2:
      return
        grid_visibilities(
          host_device,
          std::move(*visibilities.m_v2),
          update_grid_weights,
          do_degrid,
          return_visibilities,
          do_grid);
      break;
    case 3:
      return
        grid_visibilities(
          host_device,
          std::move(*visibilities.m_v3),
          update_grid_weights,
          do_degrid,
          return_visibilities,
          do_grid);
      break;
    case 4:
      return
        grid_visibilities(
          host_device,
          std::move(*visibilities.m_v4),
          update_grid_weights,
          do_degrid,
          return_visibilities,
          do_grid);
      break;
    default:
      assert(false);
      return Error("Assertion violation");
      break;
    }
  }

  void
  fence() const noexcept override {
    std::scoped_lock lock(m_mtx);
    fence_unlocked();
  }

  std::unique_ptr<GridWeightArray>
  grid_weights() const override {
    std::scoped_lock lock(m_mtx);
    fence_unlocked();
    auto& exec =
      m_exec_spaces[next_exec_space_unlocked(StreamPhase::PRE_GRIDDING)];
    auto wgts_h = K::create_mirror(m_weights);
    K::deep_copy(exec.space, wgts_h, m_weights);
    exec.fence();
    return std::make_unique<GridWeightViewArray<D>>(wgts_h);
  }

  std::shared_ptr<GridWeightArray::value_type>
  grid_weights_ptr() const override {
    return
      std::make_shared<GridWeightPtr<
        typename execution_space::array_layout,
        memory_space>>(m_weights)->ptr();
  }

  size_t
  grid_weights_span() const override {
    return m_weights.span();
  }

  std::unique_ptr<GridValueArray>
  grid_values() const override {
    std::scoped_lock lock(m_mtx);
    fence_unlocked();
    auto& exec =
      m_exec_spaces[next_exec_space_unlocked(StreamPhase::PRE_GRIDDING)];
    auto grid_h = K::create_mirror(m_grid);
    K::deep_copy(exec.space, grid_h, m_grid);
    exec.fence();
    return std::make_unique<GridValueViewArray<D>>(grid_h);
  }

  std::shared_ptr<GridValueArray::value_type>
  grid_values_ptr() const override {
    return
      std::make_shared<GridValuePtr<
        typename layouts::GridLayout<D>::layout,
        memory_space>>(m_grid)->ptr();
  }

  size_t
  grid_values_span() const override {
    return m_grid.span();
  }

  std::unique_ptr<GridValueArray>
  model_values() const override {
    std::scoped_lock lock(m_mtx);
    fence_unlocked();
    auto& exec =
      m_exec_spaces[next_exec_space_unlocked(StreamPhase::PRE_GRIDDING)];
    if (m_model.is_allocated()) {
      auto model_h = K::create_mirror(m_model);
      K::deep_copy(exec.space, model_h, m_model);
      exec.fence();
      return std::make_unique<GridValueViewArray<D>>(model_h);
    } else {
      std::array<unsigned, 4> ex{
        static_cast<unsigned>(m_grid.extent(0)),
        static_cast<unsigned>(m_grid.extent(1)),
        static_cast<unsigned>(m_grid.extent(2)),
        static_cast<unsigned>(m_grid.extent(3))};
      return std::make_unique<UnallocatedModelValueArray>(ex);
    }
  }

  std::shared_ptr<GridValueArray::value_type>
  model_values_ptr() const override {
    return
      std::make_shared<GridValuePtr<
        typename layouts::GridLayout<D>::layout,
        memory_space>>(m_model)->ptr();
  }

  size_t
  model_values_span() const override {
    return m_model.span();
  }

  void
  reset_grid() override {
    fence();
    new_grid(true, true);
  }

  void
  reset_model() override {
    fence();
    m_model = decltype(m_model)();
  }

  void
  normalize_by_weights(grid_value_fp wfactor) override {
    core::const_weight_view<typename execution_space::array_layout, memory_space>
      cweights = m_weights;
    switch (grid_normalizer_version()) {
    case 0:
      core::GridNormalizer<execution_space, 0>::kernel(
        m_exec_spaces[next_exec_space(StreamPhase::GRIDDING)].space,
        m_grid,
        cweights,
        wfactor);
      break;
    default:
      assert(false);
      break;
    }
  }

  std::optional<Error>
  apply_grid_fft(grid_value_fp norm, FFTSign sign, bool in_place)
    override {

    std::optional<Error> err;
    if (in_place) {
      switch (fft_version()) {
      case 0:
        err =
          core::FFT<execution_space, 0>
          ::in_place_kernel(
            m_exec_spaces[next_exec_space(StreamPhase::GRIDDING)].space,
            sign,
            m_grid);
        break;
      default:
        assert(false);
        break;
      }
    } else {
      core::const_grid_view<typename layouts::GridLayout<D>::layout, memory_space> pre_grid
        = m_grid;
      new_grid(false, false);
      switch (fft_version()) {
      case 0:
        err =
          core::FFT<execution_space, 0>::out_of_place_kernel(
            m_exec_spaces[next_exec_space(StreamPhase::GRIDDING)].space,
            sign,
            pre_grid,
            m_grid);
        break;
      default:
        assert(false);
        break;
      }
    }
    // apply normalization
    if (norm != 1)
      switch (grid_normalizer_version()) {
      case 0:
        core::GridNormalizer<execution_space, 0>::kernel(
          m_exec_spaces[next_exec_space(StreamPhase::GRIDDING)].space,
          m_grid,
          norm);
        break;
      default:
        assert(false);
        break;
      }
    return err;
  }

  std::optional<Error>
  apply_model_fft(grid_value_fp norm, FFTSign sign, bool in_place)
    override {

    std::optional<Error> err;
    if (m_model.is_allocated()){
      if (in_place) {
        switch (fft_version()) {
        case 0:
          err =
            core::FFT<execution_space, 0>
            ::in_place_kernel(
              m_exec_spaces[next_exec_space(StreamPhase::PRE_GRIDDING)].space,
              sign,
              m_model);
          break;
        default:
          assert(false);
          break;
        }
      } else {
        core::const_grid_view<typename layouts::GridLayout<D>::layout, memory_space> pre_model
          = m_model;
        std::array<int, 4> ig{
          static_cast<int>(m_grid_size[0]),
          static_cast<int>(m_grid_size[1]),
          static_cast<int>(m_grid_size[2]),
          static_cast<int>(m_grid_size[3])};
        m_model =
          decltype(m_model)(
            K::ViewAllocateWithoutInitializing("grid"),
            layouts::GridLayout<D>::dimensions(ig));
        switch (fft_version()) {
        case 0:
          err =
            core::FFT<execution_space, 0>::out_of_place_kernel(
              m_exec_spaces[next_exec_space(StreamPhase::PRE_GRIDDING)].space,
              sign,
              pre_model,
              m_model);
          break;
        default:
          assert(false);
          break;
        }
      }
      // apply normalization
      if (norm != 1)
        switch (grid_normalizer_version()) {
        case 0:
          core::GridNormalizer<execution_space, 0>::kernel(
            m_exec_spaces[next_exec_space(StreamPhase::PRE_GRIDDING)].space,
            m_model,
            norm);
          break;
        default:
          assert(false);
          break;
        }
    }
    return err;
  }

  void
  shift_grid(ShiftDirection direction) override {
    switch (grid_shifter_version()) {
    case 0:
      core::GridShifter<execution_space, 0>::kernel(
        direction,
        m_exec_spaces[next_exec_space(StreamPhase::GRIDDING)].space,
        m_grid);
      break;
    default:
      assert(false);
      break;
    }
  }

  void
  shift_model(ShiftDirection direction) override {
    switch (grid_shifter_version()) {
    case 0:
      core::GridShifter<execution_space, 0>::kernel(
        direction,
        m_exec_spaces[next_exec_space(StreamPhase::GRIDDING)].space,
        m_model);
      break;
    default:
      assert(false);
      break;
    }
  }

private:

  void
  fence_unlocked() const noexcept {
    for (auto& i : m_exec_space_indexes) {
      auto& exec = m_exec_spaces[i];
      exec.fence();
    }
    m_current = StreamPhase::PRE_GRIDDING;
  }

  void
  swap(StateT& other) noexcept {
    assert(m_max_active_tasks == other.m_max_active_tasks);
    std::swap(m_max_visibility_batch_size, other.m_max_visibility_batch_size);
    std::swap(m_grid_size, other.m_grid_size);
    std::swap(m_grid_scale, other.m_grid_scale);
    std::swap(m_implementation_versions, other.m_implementation_versions);

    std::swap(m_grid, other.m_grid);
    std::swap(m_weights, other.m_weights);
    std::swap(m_model, other.m_model);
    std::swap(m_mueller_indexes, other.m_mueller_indexes);
    std::swap(m_conjugate_mueller_indexes, other.m_conjugate_mueller_indexes);
    std::swap(m_streams, other.m_streams);
    std::swap(m_exec_spaces, other.m_exec_spaces);
    std::swap(m_exec_space_indexes, other.m_exec_space_indexes);
    std::swap(m_current, other.m_current);

    for (size_t i = 0; i < m_max_active_tasks; ++i) {
      auto tmp_cf = std::move(m_cfs[i]);
      std::get<0>(tmp_cf).state = this;
      m_cfs[i] = std::move(other.m_cfs[i]);
      other.m_cfs[i] = std::move(tmp_cf);
    }
    std::swap(m_cf_indexes, other.m_cf_indexes);
  }

  void
  init_state(const std::variant<const CFArrayShape*, const StateT*>& init) {
    m_streams.resize(m_max_active_tasks);
    m_cfs.resize(m_max_active_tasks);
    for (unsigned i = 0; i < m_max_active_tasks; ++i) {
      if constexpr (!std::is_void_v<stream_type>) {
        auto rc = core::DeviceT<D>::create_stream(m_streams[i]);
        assert(rc);
        m_exec_spaces.emplace_back(execution_space(m_streams[i]));
        if (std::holds_alternative<const CFArrayShape*>(init)) {
          m_exec_space_indexes.push_back(i);
          m_cf_indexes.push_back(i);
        }
      } else {
        m_exec_spaces.emplace_back(execution_space());
        if (std::holds_alternative<const CFArrayShape*>(init)) {
          m_exec_space_indexes.push_back(i);
          m_cf_indexes.push_back(i);
        }
      }
      auto& esp = m_exec_spaces.back();
      if constexpr (!std::is_same_v<K::HostSpace, memory_space>)
        esp.visbuff =
          decltype(esp.visbuff)(
            K::ViewAllocateWithoutInitializing("visibility_buffer"),
            m_max_visibility_batch_size);
      esp.gvisbuff =
        decltype(esp.gvisbuff)(
          K::ViewAllocateWithoutInitializing("gvis_buffer"),
          m_max_visibility_batch_size);
    }

    if (std::holds_alternative<const CFArrayShape*>(init)) {
      const CFArrayShape* init_cf_shape = std::get<const CFArrayShape*>(init);
      for (auto& [cf, last] : m_cfs) {
        cf.state = this;
        cf.prepare_pool(init_cf_shape, true);
        last.reset();
      }
    } else {
      const StateT* ost = std::get<const StateT*>(init);
      for (auto& i : ost->m_exec_space_indexes) {
        auto& esp = m_exec_spaces[i];
        m_exec_space_indexes.push_back(i);
        auto& st_esp = ost->m_exec_spaces[i];
        K::deep_copy(esp.space, esp.visbuff, st_esp.visbuff);
        K::deep_copy(esp.space, esp.gvisbuff, st_esp.gvisbuff);
        if constexpr (std::is_same_v<K::HostSpace, memory_space>) {
          std::visit(
            overloaded {
              [&esp](auto& v) {
                esp.visibilities = v;
              }
            },
            st_esp.visibilities);
        } else {
          std::visit(
            overloaded {
              [&esp](auto& v) {
                using v_t = std::remove_reference_t<decltype(v)>;
                constexpr unsigned N = v_t::value_type::npol;
                esp.visibilities =
                  core::visdata_view<N, memory_space>(
                    reinterpret_cast<core::VisData<N>*>(esp.visbuff.data()),
                    v.extent(0));
              }
            },
            st_esp.visibilities);
        }
        // FIXME: esp.copy_state = st_esp.copy_state;
      }
      for (auto& i : ost->m_cf_indexes) {
        auto& [cf, last] = m_cfs[i];
        m_cf_indexes.push_back(i);
        cf = std::get<0>(ost->m_cfs[i]);
        cf.state = this;
      }
      if (ost->m_model.is_allocated()) {
        std::array<int, 4> ig{
          static_cast<int>(m_grid_size[0]),
          static_cast<int>(m_grid_size[1]),
          static_cast<int>(m_grid_size[2]),
          static_cast<int>(m_grid_size[3])};
        m_model =
          decltype(m_model)(
            K::ViewAllocateWithoutInitializing("model"),
            layouts::GridLayout<D>::dimensions(ig));
        K::deep_copy(m_exec_spaces[0].space, m_model, ost->m_model);
      }
    }
    m_current = StreamPhase::PRE_GRIDDING;
  }

  /** copy Mueller indexes to device */
  template <size_t N>
    core::mindex_view<memory_space>
  copy_mueller_indexes(
    const std::string& name,
    const std::vector<iarray<N>>& mindexes) {

    auto& esp = m_exec_spaces[next_exec_space(StreamPhase::PRE_GRIDDING)];
    core::mindex_view<memory_space> result(name);
    auto mueller_indexes_h = K::create_mirror_view(result);
    size_t mr = 0;
    for (; mr < mindexes.size(); ++mr) {
      auto& mi_row = mindexes[mr];
      size_t mc = 0;
      for (; mc < N; ++mc)
        mueller_indexes_h(mr, mc) = static_cast<int>(mi_row[mc]);
      for (; mc < result.extent(1); ++mc)
        mueller_indexes_h(mr, mc) = -1;
    }
    for (; mr < result.extent(0); ++mr)
      for (size_t mc = 0; mc < result.extent(1); ++mc)
        mueller_indexes_h(mr, mc) = -1;
    K::deep_copy(esp.space, result, mueller_indexes_h);
    return result;
  }

  core::mindex_view<memory_space>
  init_mueller(const std::string& name, const IArrayVector& mueller_indexes) {

    switch (mueller_indexes.m_npol) {
    case 1:
      return copy_mueller_indexes(name, *mueller_indexes.m_v1);
      break;
    case 2:
      return copy_mueller_indexes(name, *mueller_indexes.m_v2);
      break;
    case 3:
      return copy_mueller_indexes(name, *mueller_indexes.m_v3);
      break;
    case 4:
      return copy_mueller_indexes(name, *mueller_indexes.m_v4);
      break;
    default:
      assert(false);
      return core::mindex_view<memory_space>(name);
      break;
    }
  }

protected:

  friend class CFPool<D>;

  int
  next_exec_space_unlocked(StreamPhase next) const {
    int old_idx = m_exec_space_indexes.front();
    int new_idx = old_idx;
    if (m_current == StreamPhase::GRIDDING
        && next == StreamPhase::PRE_GRIDDING) {
      if (m_max_active_tasks > 1) {
        m_exec_space_indexes.push_back(old_idx);
        m_exec_space_indexes.pop_front();
        new_idx = m_exec_space_indexes.front();
      }
      m_exec_spaces[new_idx].fence();
      std::get<1>(m_cfs[m_cf_indexes.front()]) = new_idx;
    }
#ifndef NDEBUG
    std::cout << m_current << "(" << old_idx << ")->"
              << next << "(" << new_idx << ")"
              << std::endl;
#endif // NDEBUG
    m_current = next;
    return new_idx;
  }

  int
  next_exec_space(StreamPhase next) const {
    std::scoped_lock lock(m_mtx);
    return next_exec_space_unlocked(next);
  }

  void
  switch_cf_pool() {
    auto esp_idx = next_exec_space(StreamPhase::PRE_GRIDDING);
    m_cf_indexes.push_back(m_cf_indexes.front());
    m_cf_indexes.pop_front();
    auto& [cf, last] = m_cfs[m_cf_indexes.front()];
    if (last.value_or(esp_idx) != esp_idx)
      m_exec_spaces[last.value()].fence();
    last = esp_idx;
  }

private:

  void
  new_grid(std::variant<const StateT*, bool> source, bool also_weights) {

    const bool create_without_init =
      std::holds_alternative<const StateT*>(source) || !std::get<bool>(source);

    // in the following, we don't call next_exec_space() except when a stream is
    // required, as there are code paths that never use a stream, and thus we
    // can avoid unnecessary stream switches
    std::array<int, 4> ig{
      static_cast<int>(m_grid_size[0]),
      static_cast<int>(m_grid_size[1]),
      static_cast<int>(m_grid_size[2]),
      static_cast<int>(m_grid_size[3])};
    if (create_without_init)
      m_grid =
        decltype(m_grid)(
          K::ViewAllocateWithoutInitializing("grid"),
          layouts::GridLayout<D>::dimensions(ig));
    else
      m_grid =
        decltype(m_grid)(
          K::view_alloc(
            "grid",
            m_exec_spaces[next_exec_space(StreamPhase::PRE_GRIDDING)].space),
          layouts::GridLayout<D>::dimensions(ig));
#ifndef NDEBUG
    std::cout << "alloc grid sz " << m_grid.extent(0)
              << " " << m_grid.extent(1)
              << " " << m_grid.extent(2)
              << " " << m_grid.extent(3)
              << std::endl;
    std::cout << "alloc grid str " << m_grid.stride(0)
              << " " << m_grid.stride(1)
              << " " << m_grid.stride(2)
              << " " << m_grid.stride(3)
              << std::endl;
#endif // NDEBUG

    static_assert(
      GridWeightArray::Axis::mrow == 0 && GridWeightArray::Axis::cube == 1);
    if (also_weights) {
      if (create_without_init)
        m_weights =
          decltype(m_weights)(
            K::ViewAllocateWithoutInitializing("weights"),
            static_cast<int>(m_grid_size[static_cast<int>(core::GridAxis::mrow)]),
            static_cast<int>(m_grid_size[static_cast<int>(core::GridAxis::cube)]));
      else
        m_weights =
          decltype(m_weights)(
            K::view_alloc(
              "weights",
              m_exec_spaces[next_exec_space(StreamPhase::PRE_GRIDDING)].space),
            static_cast<int>(m_grid_size[static_cast<int>(core::GridAxis::mrow)]),
            static_cast<int>(m_grid_size[static_cast<int>(core::GridAxis::cube)]));
    }
    if (std::holds_alternative<const StateT*>(source)) {
      auto st = std::get<const StateT*>(source);
      auto& exec = m_exec_spaces[next_exec_space(StreamPhase::PRE_GRIDDING)];
      K::deep_copy(exec.space, m_grid, st->m_grid);
      if (also_weights)
        K::deep_copy(exec.space, m_weights, st->m_weights);
    }
  }
};

} // end namespace Impl

std::unique_ptr<GridValueArray>
GridValueArray::copy_from(
  const std::string& name,
  Device target_device,
  Device host_device,
  const value_type* src,
  const std::array<unsigned, GridValueArray::rank>& extents,
  Layout layout) {

  static_assert(
    static_cast<int>(core::GridAxis::x) == GridValueArray::Axis::x
    && static_cast<int>(core::GridAxis::y) == GridValueArray::Axis::y
    && static_cast<int>(core::GridAxis::mrow) == GridValueArray::Axis::mrow
    && static_cast<int>(core::GridAxis::cube) == GridValueArray::Axis::cube);

  switch (target_device) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial:
    return
      Impl::GridValueViewArray<Device::Serial>::copy_from(
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
      Impl::GridValueViewArray<Device::OpenMP>::copy_from(
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
      Impl::GridValueViewArray<Device::Cuda>::copy_from(
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

std::unique_ptr<GridWeightArray>
GridWeightArray::copy_from(
  const std::string& name,
  Device target_device,
  Device host_device,
  const value_type* src,
  const std::array<unsigned, GridWeightArray::rank>& extents,
  Layout layout) {

  switch (target_device) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial:
    return
      Impl::GridWeightViewArray<Device::Serial>::copy_from(
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
      Impl::GridWeightViewArray<Device::OpenMP>::copy_from(
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
      Impl::GridWeightViewArray<Device::Cuda>::copy_from(
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

} // end namespace hpg

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
