// Copyright 2021 Associated Universities, Inc. Washington DC, USA.
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
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "hpg.hpp"
#include <array>

namespace hpg {

/** index for CFCell in CASA CFStore2
 *
 * can also be used for any dense subset of CFCells in a CFStore2
 */
struct HPG_EXPORT CFCellIndex final {
  /** default constructor
   */
  CFCellIndex() {}

  /** constructor
   */
  CFCellIndex(
    unsigned baseline_class,
    unsigned time,
    unsigned w_plane,
    unsigned frequency,
    unsigned mueller)
    : m_baseline_class(baseline_class)
    , m_time(time)
    , m_w_plane(w_plane)
    , m_frequency(frequency)
    , m_mueller(mueller) {}

  /** equality operator
   *
   * instances are equal iff all fields are equal
   */
  bool
  operator==(const CFCellIndex& rhs) const {
    return
      m_baseline_class == rhs.m_baseline_class
      && m_time == rhs.m_time
      && m_w_plane == rhs.m_w_plane
      && m_frequency == rhs.m_frequency
      && m_mueller == rhs.m_mueller;
  }

  unsigned m_baseline_class; /**< baseline class index */
  unsigned m_time; /**< time index */
  unsigned m_w_plane; /**< w plane index */
  unsigned m_frequency; /**< frequency index */
  unsigned m_mueller; /**< mueller element index */
};

/** index converter class between cf_index_t and CFCellIndex
 *
 * May be used for any dense subset of a uniform CFStore2 x
 * CFBuffer. Nonuniformity in the CFCells is allowed, but no indexing is
 * provided at that level by this class.
 */
class HPG_EXPORT CFSimpleIndexer final {
public:

  /** axis descriptor
   *
   *  pair elements: (axis length, allows variable-size CF)
   */
  using axis_desc_t = std::pair<unsigned, bool>;

  /** CF index
   *
   * array elements: (mueller, cube, grp)
   */
  using cf_index_t = std::array<unsigned, 3>;

  axis_desc_t m_baseline_class; /**< baseline class axis descriptor */
  axis_desc_t m_time; /**< time axis descriptor */
  axis_desc_t m_w_plane; /**< w plane axis descriptor */
  axis_desc_t m_frequency; /**< frequency axis descriptor */
  unsigned m_mueller; /**< Mueller element axis descriptor */

  /** constructor
   */
  CFSimpleIndexer(
    const axis_desc_t& baseline_class,
    const axis_desc_t& time,
    const axis_desc_t& w_plane,
    const axis_desc_t& frequency,
    unsigned mueller)
    : m_baseline_class(baseline_class)
    , m_time(time)
    , m_w_plane(w_plane)
    , m_frequency(frequency)
    , m_mueller(mueller) {}

  /** convert CFCellIndex to cf_index_t
   */
  cf_index_t
  cf_index(const CFCellIndex& index) const {
    cf_index_t result{index.m_mueller, 0, 0};
    acc_index(result, index.m_baseline_class, m_baseline_class);
    acc_index(result, index.m_time, m_time);
    acc_index(result, index.m_w_plane, m_w_plane);
    acc_index(result, index.m_frequency, m_frequency);
    return result;
  }

  /** convert cf_index_t to CFCellIndex
   */
  CFCellIndex
  cell_index(cf_index_t index) const {
    CFCellIndex result;
    result.m_mueller = index[0];
    sep_index(result.m_frequency, index, m_frequency);
    sep_index(result.m_w_plane, index, m_w_plane);
    sep_index(result.m_time, index, m_time);
    sep_index(result.m_baseline_class, index, m_baseline_class);
    return result;
  }

  /** extents of vis_cf_index_t
   */
  cf_index_t
  cf_extents() const {
    cf_index_t result{m_mueller, 1, 1};
    ext_index(result, m_baseline_class);
    ext_index(result, m_time);
    ext_index(result, m_w_plane);
    ext_index(result, m_frequency);
    return result;
  }

  /** extents of CFCellIndex
   */
  std::array<unsigned, 5>
  cell_extents() const {
    return {
      m_baseline_class.first,
      m_time.first,
      m_w_plane.first,
      m_frequency.first,
      m_mueller};
  }

private:

  /** accumulate index value to linearized cube or group index  */
  static void
  acc_index(cf_index_t& index, unsigned i, const axis_desc_t& ad) {
    unsigned& index_part = (ad.second ? index[2] : index[1]);
    index_part = index_part * ad.first + i;
  }

  /** separate index value from linearized cube or group index */
  static void
  sep_index(unsigned& index, cf_index_t& i, const axis_desc_t& ad) {
    unsigned& i_part = (ad.second ? i[2] : i[1]);
    index = i_part % ad.first;
    i_part /= ad.first;
  }

  /** extent of linearized cube or group index */
  static void
  ext_index(cf_index_t& index, const axis_desc_t& ad) {
    unsigned& index_part = (ad.second ? index[2] : index[1]);
    index_part *= ad.first;
  }
};

}  // end namespace hpg

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
