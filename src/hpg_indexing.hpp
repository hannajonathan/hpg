#pragma once
#include "hpg.hpp"

namespace hpg {

struct HPG_EXPORT CFCellIndex final {
  CFCellIndex() {}

  CFCellIndex(
    unsigned baseline_class,
    unsigned time,
    unsigned w_plane,
    unsigned frequency,
    unsigned polarization_product)
    : m_baseline_class(baseline_class)
    , m_time(time)
    , m_w_plane(w_plane)
    , m_frequency(frequency)
    , m_polarization_product(polarization_product) {}

  bool
  operator==(const CFCellIndex& rhs) const {
    return
      m_baseline_class == rhs.m_baseline_class
      && m_time == rhs.m_time
      && m_w_plane == rhs.m_w_plane
      && m_frequency == rhs.m_frequency
      && m_polarization_product == rhs.m_polarization_product;
  }

  unsigned m_baseline_class;
  unsigned m_time;
  unsigned m_w_plane;
  unsigned m_frequency;
  unsigned m_polarization_product;
};

class HPG_EXPORT CFSimpleIndexer final {
public:

  // pair elements: (axis length, is variable CF size)
  using axis_desc_t = std::pair<unsigned, bool>;

  axis_desc_t m_baseline_class;
  axis_desc_t m_time;
  axis_desc_t m_w_plane;
  axis_desc_t m_frequency;
  axis_desc_t m_polarization_product;

  CFSimpleIndexer(
    const axis_desc_t& baseline_class,
    const axis_desc_t& time,
    const axis_desc_t& w_plane,
    const axis_desc_t& frequency,
    const axis_desc_t& polarization_product)
    : m_baseline_class(baseline_class)
    , m_time(time)
    , m_w_plane(w_plane)
    , m_frequency(frequency)
    , m_polarization_product(polarization_product) {}

  vis_cf_index_t
  cf_index(const CFCellIndex& index) const {
    vis_cf_index_t result;
    acc_index(result, index.m_baseline_class, m_baseline_class);
    acc_index(result, index.m_time, m_time);
    acc_index(result, index.m_w_plane, m_w_plane);
    acc_index(result, index.m_frequency, m_frequency);
    acc_index(result, index.m_polarization_product, m_polarization_product);
    return result;
  }

  CFCellIndex
  cell_index(vis_cf_index_t index) const {
    CFCellIndex result;
    sep_index(result.m_polarization_product, index, m_polarization_product);
    sep_index(result.m_frequency, index, m_frequency);
    sep_index(result.m_w_plane, index, m_w_plane);
    sep_index(result.m_time, index, m_time);
    sep_index(result.m_baseline_class, index, m_baseline_class);
    return result;
  }

  vis_cf_index_t
  extents() const {
    vis_cf_index_t result{1, 1};
    ext_index(result, m_baseline_class);
    ext_index(result, m_time);
    ext_index(result, m_w_plane);
    ext_index(result, m_frequency);
    ext_index(result, m_polarization_product);
    return result;
  }

private:

  static void
  acc_index(vis_cf_index_t& index, unsigned i, const axis_desc_t& ad) {
    unsigned& index_part = (ad.second ? index.second : index.first);
    index_part = index_part * ad.first + i;
  }

  static void
  sep_index(unsigned& index, vis_cf_index_t& i, const axis_desc_t& ad) {
    unsigned& i_part = (ad.second ? i.second : i.first);
    index = i_part % ad.first;
    i_part /= ad.first;
  }

  static void
  ext_index(vis_cf_index_t& index, const axis_desc_t& ad) {
    unsigned& index_part = (ad.second ? index.second : index.first);
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
