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

#include "hpg_config.hpp"
#include "hpg.hpp"
#include "hpg_core.hpp"
// #inlude "hpg_export.h"

#include <cassert>
#include <memory>
#include <optional>
#include <ostream>
#include <tuple>
#include <vector>

#include <Kokkos_Core.hpp>

#ifdef __NVCC__
# define WORKAROUND_NVCC_IF_CONSTEXPR_BUG
#endif

/** @file hpg_impl.hpp
 *
 * HPG implementation header file
 */
namespace hpg {

/** disabled device error
 *
 * Device is not enabled in HPG configuration.
 */
struct DisabledDeviceError
  : public Error {

  DisabledDeviceError()
    : Error("Requested device is not enabled", ErrorType::DisabledDevice) {}
};

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

namespace impl {

namespace K = Kokkos;

/** scoped Kokkos profiling region value */
struct /*HPG_EXPORT*/ ProfileRegion {
  inline ProfileRegion(const char* nm) {
    K::Profiling::pushRegion(nm);
  }

  inline ~ProfileRegion() {
    K::Profiling::popRegion();
  }
};

bool
is_initialized() noexcept;

bool
initialize(const InitArguments& args);

void
finalize();

/** type trait associating Kokkos device with hpg Device */
template <Device D>
struct /*HPG_EXPORT*/ DeviceT {
  using kokkos_device = void;

  static constexpr const char* const name = "";
};

#ifdef HPG_ENABLE_SERIAL
/** Serial device type trait */
template <>
struct /*HPG_EXPORT*/ DeviceT<Device::Serial> {
  using kokkos_device = K::Serial;

  static constexpr const char* const name = "Serial";
};
#endif // HPG_ENABLE_SERIAL

#ifdef HPG_ENABLE_OPENMP
/** OpenMP device type trait */
template <>
struct /*HPG_EXPORT*/ DeviceT<Device::OpenMP> {
  using kokkos_device = K::OpenMP;

  static constexpr const char* const name = "OpenMP";
};
#endif // HPG_ENABLE_OPENMP

#ifdef HPG_ENABLE_CUDA
/** Cuda device type trait */
template <>
struct /*HPG_EXPORT*/ DeviceT<Device::Cuda> {
  using kokkos_device = K::Cuda;

  static constexpr const char* const name = "Cuda";
};
#endif // HPG_ENABLE_CUDA

/** axis order for strided grid layout */
static const std::array<int, 4> strided_grid_layout_order{
  int(core::GridAxis::y),
  int(core::GridAxis::mrow),
  int(core::GridAxis::x),
  int(core::GridAxis::cube)};

/** device-specific grid array layout */
template <typename Device>
struct /*HPG_EXPORT*/ GridLayout {

  /** Kokkos layout type */
  using layout =
    std::conditional_t<
      std::is_same_v<typename Device::array_layout, K::LayoutLeft>,
      K::LayoutLeft,
      K::LayoutStride>;

  /** create Kokkos layout using given grid dimensions
   *
   * logical index order matches GridAxis definition
   */
  static layout
  dimensions(const std::array<int, 4>& dims) {
    if constexpr (std::is_same_v<layout, K::LayoutLeft>) {
      return K::LayoutLeft(dims[0], dims[1], dims[2], dims[3]);
    } else {
      return
        K::LayoutStride::order_dimensions(
          4,
          strided_grid_layout_order.data(),
          dims.data());
    }
#ifdef WORKAROUND_NVCC_IF_CONSTEXPR_BUG
    return layout();
#endif
  }
};

/** CFLayout version number
 *
 * @todo make something useful of this, maybe add a value template parameter to
 * CFLayout?
 */
static const unsigned cf_layout_version_number = 0;

/** axis order for strided CF array layout */
static const std::array<int, 6> strided_cf_layout_order{
  int(core::CFAxis::mueller),
  int(core::CFAxis::y_major),
  int(core::CFAxis::x_major),
  int(core::CFAxis::cube),
  int(core::CFAxis::x_minor),
  int(core::CFAxis::y_minor)};

/** device-specific constant-support CF array layout */
template <typename Device>
struct /*HPG_EXPORT*/ CFLayout {

  /** Kokkos layout type */
  using layout =
    std::conditional_t<
      std::is_same_v<typename Device::array_layout, K::LayoutLeft>,
      K::LayoutLeft,
      K::LayoutStride>;

  /**
   * create Kokkos layout using given CFArray slice
   *
   * logical index order matches CFAxis definition
   *
   * @todo: verify these layouts
   */
  static layout
  dimensions(const CFArrayShape* cf, unsigned grp) {
    auto extents = cf->extents(grp);
    auto os = cf->oversampling();
    std::array<int, 6> dims{
      int((extents[CFArray::Axis::x] + os - 1) / os),
      int((extents[CFArray::Axis::y] + os - 1) / os),
      int(extents[CFArray::Axis::mueller]),
      int(extents[CFArray::Axis::cube]),
      int(os),
      int(os)
    };
    if constexpr (std::is_same_v<layout, K::LayoutLeft>) {
      return
        K::LayoutLeft(dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]);
    } else {
      return
        K::LayoutStride::order_dimensions(
          6,
          strided_cf_layout_order.data(),
          dims.data());
    }
#ifdef WORKAROUND_NVCC_IF_CONSTEXPR_BUG
    return layout();
#endif
  }
};

/** sign of a value
 *
 * @return -1, if less than 0; +1, if greater than 0; 0, if equal to 0
 */
template <typename T>
inline constexpr int
sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

/** type for (plain) vector data */
template <typename T>
using vector_data = std::shared_ptr<std::vector<T>>;

template <typename Layout, typename memory_space>
struct /*HPG_EXPORT*/ GridWeightPtr
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
struct /*HPG_EXPORT*/ GridValuePtr
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

/** concrete sub-class of abstract GridValueArray */
template <Device D>
class /*HPG_EXPORT*/ GridValueViewArray final
  : public GridValueArray {
public:

  using kokkos_device = typename DeviceT<D>::kokkos_device;
  using memory_space = typename kokkos_device::memory_space;
  using grid_layout = GridLayout<kokkos_device>;

  using grid_t =
    typename core::grid_view<
      typename grid_layout::layout,
      memory_space>::HostMirror;

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
      int(core::GridAxis::x) == 0
      && int(core::GridAxis::y) == 1
      && int(core::GridAxis::mrow) == 2
      && int(core::GridAxis::cube) == 3);
    return reinterpret_cast<const value_type&>(grid(x, y, mrow, cube));
  }

  value_type&
  operator()(unsigned x, unsigned y, unsigned mrow, unsigned cube) override {

    static_assert(
      int(core::GridAxis::x) == 0
      && int(core::GridAxis::y) == 1
      && int(core::GridAxis::mrow) == 2
      && int(core::GridAxis::cube) == 3);
    return reinterpret_cast<value_type&>(grid(x, y, mrow, cube));
  }

  template <Device H>
  void
  copy_to(value_type* dst, Layout lyo) const {

    // we're assuming that a K::LayoutLeft or K::LayoutRight copy has no padding
    // (otherwise, the following is broken, not least because it may result in
    // an out-of-bounds access on dst)

    auto espace = typename DeviceT<H>::kokkos_device::execution_space();

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

    std::array<int, rank>
      iext{int(extents[0]), int(extents[1]), int(extents[2]), int(extents[3])};
    grid_t grid(
      K::ViewAllocateWithoutInitializing(name),
      grid_layout::dimensions(iext));

    // we're assuming that a K::LayoutLeft or K::LayoutRight copy has no padding
    // (otherwise, the following is broken, not least because it may result in
    // an out-of-bounds access on dst)

    auto espace = typename DeviceT<H>::kokkos_device::execution_space();

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
class /*HPG_EXPORT*/ GridWeightViewArray final
  : public GridWeightArray {
 public:

  using kokkos_device = typename DeviceT<D>::kokkos_device;
  using memory_space = typename kokkos_device::memory_space;
  using layout = typename kokkos_device::array_layout;
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

    auto espace = typename DeviceT<H>::kokkos_device::execution_space();

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

    auto espace = typename DeviceT<H>::kokkos_device::execution_space();

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
class /*HPG_EXPORT*/ UnallocatedModelValueArray final
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

    auto espace = typename DeviceT<H>::kokkos_device::execution_space();

    switch (lyo) {
    case Layout::Left: {
      K::View<
        core::gv_t****,
        K::LayoutLeft,
        typename DeviceT<H>::kokkos_device::memory_space,
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
        typename DeviceT<H>::kokkos_device::memory_space,
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

std::optional<std::tuple<unsigned, std::optional<Device>>>
parsed_cf_layout_version(const std::string& layout);

std::string
construct_cf_layout_version(unsigned vn, Device device);

/** initialize model visibilities view from GridValueArray instance */
template <Device D, typename GVH>
static void
init_model(GVH& gv_h, const GridValueArray& gv) {
  static_assert(
    K::SpaceAccessibility<
      typename DeviceT<D>::kokkos_device::memory_space,
      K::HostSpace>
    ::accessible);
  static_assert(
    int(core::GridAxis::x) == 0
    && int(core::GridAxis::y) == 1
    && int(core::GridAxis::mrow) == 2
    && int(core::GridAxis::cube) == 3);
  static_assert(
    GridValueArray::Axis::x == 0
    && GridValueArray::Axis::y == 1
    && GridValueArray::Axis::mrow == 2
    && GridValueArray::Axis::cube == 3);

  K::parallel_for(
    "init_model",
    K::MDRangePolicy<K::Rank<4>, typename DeviceT<D>::kokkos_device>(
      {0, 0, 0, 0},
      {int(gv.extent(0)),
       int(gv.extent(1)),
       int(gv.extent(2)),
       int(gv.extent(3))}),
    [&](int x, int y, int mr, int cb) {
      gv_h(x, y, mr, cb) = gv(x, y, mr, cb);
    });
}

/** device-specific implementation sub-class of hpg::DeviceCFArray class */
template <Device D>
class /*HPG_EXPORT*/ DeviceCFArray
  : public hpg::DeviceCFArray {
public:

  using kokkos_device = typename DeviceT<D>::kokkos_device;
  using cflayout = CFLayout<kokkos_device>;

  // notice layout for device D, but in HostSpace
  using cfd_view_h = core::cf_view<typename cflayout::layout, K::HostSpace>;

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
        cflayout::dimensions(this, m_extents.size() - 1));
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
      int(core::CFAxis::x_major) == 0
      && int(core::CFAxis::y_major) == 1
      && int(core::CFAxis::mueller) == 2
      && int(core::CFAxis::cube) == 3
      && int(core::CFAxis::x_minor) == 4
      && int(core::CFAxis::y_minor) == 5);
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

/** initialize CF array view from CFArray instance */
template <typename D, typename CFH>
static void
init_cf_host(CFH& cf_h, const CFArray& cf, unsigned grp) {
  static_assert(
    K::SpaceAccessibility<typename D::memory_space, K::HostSpace>
    ::accessible);
  static_assert(
    int(core::CFAxis::x_major) == 0
    && int(core::CFAxis::y_major) == 1
    && int(core::CFAxis::mueller) == 2
    && int(core::CFAxis::cube) == 3
    && int(core::CFAxis::x_minor) == 4
    && int(core::CFAxis::y_minor) == 5);
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
    K::MDRangePolicy<K::Rank<4>, D>(
      {0, 0, 0, 0},
      {int(extents[0]), int(extents[1]), int(extents[2]), int(extents[3])}),
    [&](int i, int j, int mueller, int cube) {
      auto X = i / oversampling;
      auto x = i % oversampling;
      auto Y = j / oversampling;
      auto y = j % oversampling;
      cf_h(X, Y, mueller, cube, x, y) = cf(i, j, mueller, cube, grp);
    });
}

template <Device D>
static void
layout_for_device(
  Device host_device,
  const CFArray& cf,
  unsigned grp,
  CFArray::value_type* dst) {

  using kokkos_device = typename DeviceT<D>::kokkos_device;

  auto layout = CFLayout<kokkos_device>::dimensions(&cf, grp);
  typename DeviceCFArray<D>::cfd_view_h
    cfd(reinterpret_cast<core::cf_t*>(dst), layout);
  switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial: {
    using host_device = DeviceT<Device::Serial>::kokkos_device;
    init_cf_host<host_device>(cfd, cf, grp);
    typename host_device::execution_space().fence();
    break;
  }
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
  case Device::OpenMP: {
    using host_device = DeviceT<Device::OpenMP>::kokkos_device;
    init_cf_host<host_device>(cfd, cf, grp);
    typename host_device::execution_space().fence();
    break;
  }
#endif // HPG_ENABLE_SERIAL
  default:
    assert(false);
    break;
  }
}

rval_t<size_t>
min_cf_buffer_size(Device device, const CFArray& cf, unsigned grp);

} // end namespace impl

} // end namespace hpg

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
