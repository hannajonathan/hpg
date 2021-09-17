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
#include "hpg_core.hpp"

#include <array>

#include <Kokkos_Core.hpp>

#ifdef __NVCC__
# define WORKAROUND_NVCC_IF_CONSTEXPR_BUG
#endif

namespace hpg::layouts {

/** axis order for strided grid layout */
static const std::array<int, 4> strided_grid_layout_order{
  static_cast<int>(core::GridAxis::y),
  static_cast<int>(core::GridAxis::mrow),
  static_cast<int>(core::GridAxis::x),
  static_cast<int>(core::GridAxis::cube)};

/** device-specific grid array layout */
template <Device D>
struct GridLayout {

  /** Kokkos layout type */
  using layout =
    std::conditional_t<
      std::is_same_v<
        typename core::DeviceT<D>::kokkos_device::array_layout,
        K::LayoutLeft>,
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
  static_cast<int>(core::CFAxis::mueller),
  static_cast<int>(core::CFAxis::y_major),
  static_cast<int>(core::CFAxis::x_major),
  static_cast<int>(core::CFAxis::cube),
  static_cast<int>(core::CFAxis::x_minor),
  static_cast<int>(core::CFAxis::y_minor)};

/** device-specific constant-support CF array layout */
template <Device D>
struct CFLayout {

  /** Kokkos layout type */
  using layout =
    std::conditional_t<
      std::is_same_v<
        typename core::DeviceT<D>::kokkos_device::array_layout,
        K::LayoutLeft>,
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
      static_cast<int>((extents[CFArray::Axis::x] + os - 1) / os),
      static_cast<int>((extents[CFArray::Axis::y] + os - 1) / os),
      static_cast<int>(extents[CFArray::Axis::mueller]),
      static_cast<int>(extents[CFArray::Axis::cube]),
      static_cast<int>(os),
      static_cast<int>(os)
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

} // end namespace hpg::layout

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
