#define HPG_INTERNAL
#include "hpg.hpp"
#include "gtest/gtest.h"

#include <array>
#include <cassert>
#include <complex>
#include <experimental/array>
#include <iostream>
#include <random>

#if defined(HPG_ENABLE_OPENMP)
hpg::Device default_host_device = hpg::Device::OpenMP;
#elif defined(HPG_ENABLE_SERIAL)
hpg::Device default_host_device = hpg::Device::Serial;
#else
# error "At least one host device needs to be enabled"
#endif

#if defined(HPG_ENABLE_CUDA)
hpg::Device default_device = hpg::Device::Cuda;
#elif defined(HPG_ENABLE_OPENMP)
hpg::Device default_device = hpg::Device::OpenMP;
#elif defined(HPG_ENABLE_SERIAL)
hpg::Device default_device = hpg::Device::Serial;
#else
# error "At least one device needs to be enabled"
#endif

using namespace std::complex_literals;

template <typename T>
struct epsilon {
  using F = void;
  // static constexpr F val:
};
template <>
struct epsilon<float> {
  using F = float;
  static constexpr float val = 1.0e-3;
};
template <>
struct epsilon<double> {
  using F = double;
  static constexpr double val = 1.0e-3;
};
template <>
struct epsilon<std::complex<float>> {
  using F = float;
  static constexpr float val = 1.0e-3;
};
template <>
struct epsilon<std::complex<double>> {
  using F = double;
  static constexpr double val = 1.0e-3;
};

template <typename T>
bool
near(const T& x, const T& y) {
  return std::abs(x - y) <= epsilon<T>::val * std::abs(x);
}
template <>
bool
near(const std::complex<double>& x, const std::complex<double>& y) {
  auto d = x - y;
  return
    std::abs(d.real()) <= epsilon<double>::val * std::abs(x.real())
    && std::abs(d.imag()) <= epsilon<double>::val * std::abs(x.imag());
}
template <>
bool
near(const std::complex<float>& x, const std::complex<float>& y) {
  auto d = x - y;
  return
    std::abs(d.real()) <= epsilon<float>::val * std::abs(x.real())
    && std::abs(d.imag()) <= epsilon<float>::val * std::abs(x.imag());
}

struct MyCFArrayShape
  : public hpg::CFArrayShape {

  unsigned m_oversampling;
  std::vector<std::array<unsigned, 4>> m_extents;

  MyCFArrayShape() {}

  MyCFArrayShape(
    unsigned oversampling,
    const std::vector<std::array<unsigned, 4>>& sizes)
    : m_oversampling(oversampling) {

    for (auto& sz : sizes)
      m_extents.push_back(
        {sz[0] * oversampling, sz[1] * oversampling, sz[2], sz[3]});
  }

  unsigned
  oversampling() const override {
    return m_oversampling;
  }

  unsigned
  num_groups() const override {
    return static_cast<unsigned>(m_extents.size());
  }

  std::array<unsigned, 4>
  extents(unsigned grp) const override {
    return m_extents[grp];
  }
};

struct MyCFArray final
  : public hpg::CFArray {

  unsigned m_oversampling;
  std::vector<std::array<unsigned, 4>> m_extents;
  std::vector<std::vector<std::complex<hpg::cf_fp>>> m_values;

  MyCFArray() {}

  MyCFArray(
    unsigned oversampling,
    const std::vector<std::array<unsigned, 4>>& sizes,
    const std::vector<std::vector<std::complex<hpg::cf_fp>>>& values)
    : m_oversampling(oversampling)
    , m_values(values) {

    for (auto& sz : sizes)
      m_extents.push_back(
        {sz[0] * oversampling, sz[1] * oversampling, sz[2], sz[3]});
  }

  unsigned
  oversampling() const override {
    return m_oversampling;
  }

  unsigned
  num_groups() const override {
    return static_cast<unsigned>(m_extents.size());
  }

  std::array<unsigned, 4>
  extents(unsigned grp) const override {
    return m_extents[grp];
  }

  std::complex<hpg::cf_fp>
  operator()(unsigned x, unsigned y, unsigned copol, unsigned cube, unsigned grp)
    const override {
    auto& vals = m_values[grp];
    auto& ext = m_extents[grp];
    return vals[((x * ext[1] + y) * ext[2] + copol) * ext[3] + cube];
  }
};

template <typename Generator>
MyCFArray
create_cf(
  unsigned oversampling,
  const std::vector<std::array<unsigned, 4>>& sizes,
  Generator& gen) {

  std::vector<std::vector<std::complex<hpg::cf_fp>>> values;
  for (auto& sz : sizes) {
    decltype(values)::value_type vs;
    const unsigned num_values =
      oversampling * sz[0] * oversampling * sz[1] * sz[2] * sz[3];
    vs.reserve(num_values);
    std::uniform_real_distribution<hpg::cf_fp> dist(-1.0, 1.0);
    for (auto i = 0; i < num_values; ++i)
      vs.emplace_back(dist(gen), dist(gen));
    values.push_back(std::move(vs));
  }
  return MyCFArray(oversampling, sizes, values);
}

template <typename Generator>
void
init_visibilities(
  unsigned num_vis,
  const std::array<unsigned, 4>& grid_size,
  const std::array<hpg::grid_scale_fp, 2>& grid_scale,
  const MyCFArray& cf,
  Generator& gen,
  std::vector<std::complex<hpg::visibility_fp>>& vis,
  std::vector<unsigned>& grid_cubes,
  std::vector<hpg::vis_cf_index_t>& cf_indexes,
  std::vector<hpg::vis_weight_fp>& weights,
  std::vector<hpg::vis_frequency_fp>& frequencies,
  std::vector<hpg::vis_phase_fp>& phases,
  std::vector<hpg::vis_uvw_t>& coordinates,
  std::vector<hpg::cf_phase_screen_t>& cf_phase_screens) {

  vis.clear();
  vis.reserve(num_vis);
  grid_cubes.clear();
  grid_cubes.reserve(num_vis);
  cf_indexes.clear();
  cf_indexes.reserve(num_vis);
  weights.clear();
  weights.reserve(num_vis);
  frequencies.clear();
  frequencies.reserve(num_vis);
  phases.clear();
  phases.reserve(num_vis);
  coordinates.clear();
  coordinates.reserve(num_vis);
  cf_phase_screens.reserve(num_vis);

  const double inv_lambda = 9.75719;
  const double freq = 299792458.0 * inv_lambda;
  std::uniform_int_distribution<unsigned> dist_gcube(0, grid_size[3] - 1);
  std::uniform_int_distribution<unsigned> dist_gcopol(0, grid_size[2] - 1);
  std::uniform_real_distribution<hpg::visibility_fp> dist_vis(-1.0, 1.0);
  std::uniform_real_distribution<hpg::vis_weight_fp> dist_weight(0.0, 1.0);
  std::uniform_real_distribution<hpg::cf_phase_screen_fp>
    dist_cfscreen(-3.141, 3.141);
  std::uniform_int_distribution<unsigned> dist_cfgrp(0, cf.num_groups() - 1);
  auto x0 = (cf.oversampling() * (grid_size[0] - 2)) / 2;
  auto y0 = (cf.oversampling() * (grid_size[1] - 2)) / 2;
  double uscale = grid_scale[0] * cf.oversampling() * inv_lambda;
  double vscale = grid_scale[1] * cf.oversampling() * inv_lambda;
  for (auto i = 0; i < num_vis; ++i) {
    auto grp = dist_cfgrp(gen);
    auto cfextents = cf.extents(grp);
    std::uniform_int_distribution<unsigned> dist_cfcube(0, cfextents[3] - 1);
    double ulim = (x0 - (cfextents[0]) / 2) / uscale;
    double vlim = (y0 - (cfextents[1]) / 2) / vscale;
    std::uniform_real_distribution<hpg::vis_uvw_fp> dist_u(-ulim, ulim);
    std::uniform_real_distribution<hpg::vis_uvw_fp> dist_v(-vlim, vlim);
    vis.emplace_back(dist_vis(gen), dist_vis(gen));
    grid_cubes.push_back(dist_gcube(gen));
    cf_indexes.push_back({dist_cfcube(gen), grp});
    weights.push_back(dist_weight(gen));
    frequencies.push_back(freq);
    phases.emplace_back(0.0);
    coordinates.push_back(hpg::vis_uvw_t({dist_u(gen), dist_v(gen), 0.0}));
    cf_phase_screens.push_back({dist_cfscreen(gen), dist_cfscreen(gen)});
  }
}

template <typename T>
bool
has_non_zero(const T* array) {
  if constexpr (T::rank == 2) {
    for (unsigned i = 0; i < array->extent(0); ++i)
      for (unsigned j = 0; j < array->extent(1); ++j)
        if ((*array)(i, j) != typename T::scalar_type(0))
          return true;
  } else {
    for (unsigned i = 0; i < array->extent(0); ++i)
      for (unsigned j = 0; j < array->extent(1); ++j)
        for (unsigned k = 0; k < array->extent(2); ++k)
          for (unsigned m = 0; m < array->extent(3); ++m)
            if ((*array)(i, j, k, m) != typename T::scalar_type(0))
              return true;
  }
  return false;
}

std::complex<hpg::grid_value_fp>
grid_value_encode(unsigned x, unsigned y, unsigned mr, unsigned cb) {
  return {hpg::grid_value_fp(x * 1000 + y), hpg::grid_value_fp(mr * 1000 + cb)};
}

hpg::grid_value_fp
grid_weight_encode(unsigned mr, unsigned cb) {
  return hpg::grid_value_fp(1000 * mr) +  hpg::grid_value_fp(cb);
}

TEST(GridArrays, CopyToValuesLayouts) {
  std::array<unsigned, 4> grid_size{8, 8, 4, 3};
  std::array<hpg::grid_scale_fp, 2> grid_scale{0.1, -0.1};
  auto padding = 2 * hpg::CFArray::padding;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{
      {3 + padding, 3 + padding, 4, 3},
      {2 + padding, 2 + padding, 4, 2}};

  std::mt19937 rng(42);
  MyCFArray cf = create_cf(10, cf_sizes, rng);

  auto gs_or_err =
    hpg::GridderState::create(
      default_device,
      0,
      4,
      &cf,
      grid_size,
      grid_scale);
  ASSERT_TRUE(hpg::is_value(gs_or_err));
  auto gs = hpg::get_value(std::move(gs_or_err));
  auto gvals = std::get<1>(gs.grid_values());
  for (unsigned x = 0; x < grid_size[0]; ++x)
    for (unsigned y = 0; y < grid_size[1]; ++y)
      for (unsigned mr = 0; mr < grid_size[2]; ++mr)
        for (unsigned cb = 0; cb < grid_size[3]; ++cb)
          (*gvals)(x, y, mr, cb) = grid_value_encode(x, y, mr, cb);

  auto gvals_sz = gvals->min_buffer_size();
  // copy grid values to left layout, and check results
  {
    std::vector<hpg::GridValueArray::scalar_type> gvals_left(gvals_sz);
    gvals->copy_to(default_host_device, gvals_left.data(), hpg::Layout::Left);

    bool eq = true;
    for (unsigned x = 0; eq && x < grid_size[0]; ++x)
      for (unsigned y = 0; eq && y < grid_size[1]; ++y)
        for (unsigned mr = 0; eq && mr < grid_size[2]; ++mr)
          for (unsigned cb = 0; eq && cb < grid_size[3]; ++cb) {
            auto& l =
              gvals_left[
                x + grid_size[0] * (y + grid_size[1] * (mr + grid_size[2] * cb))];
            auto val = grid_value_encode(x, y, mr, cb);
            EXPECT_EQ(l, val);
            eq = (l == val);
          }
  }
  // copy grid values to right layout, and check results
  {
    std::vector<hpg::GridValueArray::scalar_type> gvals_right(gvals_sz);
    gvals->copy_to(default_host_device, gvals_right.data(), hpg::Layout::Right);

    bool eq = true;
    for (unsigned x = 0; eq && x < grid_size[0]; ++x)
      for (unsigned y = 0; eq && y < grid_size[1]; ++y)
        for (unsigned mr = 0; eq && mr < grid_size[2]; ++mr)
          for (unsigned cb = 0; eq && cb < grid_size[3]; ++cb) {
            auto& r =
              gvals_right[
                cb + grid_size[3] * (mr + grid_size[2] * (y + grid_size[1] * x))];
            auto val = grid_value_encode(x, y, mr, cb);
            EXPECT_EQ(r, val);
            eq = (r == val);
          }
  }
}

TEST(GridArrays, CopyFromValuesLayouts) {
  std::array<unsigned, 4> grid_size{8, 8, 4, 3};

  auto gvals_sz = grid_size[0] * grid_size[1] * grid_size[2] * grid_size[3];
  // copy grid values from left layout, and check results
  {
    std::vector<hpg::GridValueArray::scalar_type> gvals_left(gvals_sz);
    for (unsigned x = 0; x < grid_size[0]; ++x)
      for (unsigned y = 0; y < grid_size[1]; ++y)
        for (unsigned mr = 0; mr < grid_size[2]; ++mr)
          for (unsigned cb = 0; cb < grid_size[3]; ++cb)
            gvals_left[
              x + grid_size[0] * (y + grid_size[1] * (mr + grid_size[2] * cb))]
              = grid_value_encode(x, y, mr, cb);
    auto gvals =
      hpg::GridValueArray::copy_from(
        "v0",
        default_device,
        default_host_device,
        gvals_left.data(),
        grid_size,
        hpg::Layout::Left);

    bool eq = true;
    for (unsigned x = 0; eq && x < grid_size[0]; ++x)
      for (unsigned y = 0; eq && y < grid_size[1]; ++y)
        for (unsigned mr = 0; eq && mr < grid_size[2]; ++mr)
          for (unsigned cb = 0; eq && cb < grid_size[3]; ++cb) {
            auto val = grid_value_encode(x, y, mr, cb);
            EXPECT_EQ((*gvals)(x, y, mr, cb), val);
            eq = ((*gvals)(x, y, mr, cb) == val);
          }
  }
  // copy grid values to right layout, and check results
  {
    std::vector<hpg::GridValueArray::scalar_type> gvals_right(gvals_sz);
    for (unsigned x = 0; x < grid_size[0]; ++x)
      for (unsigned y = 0; y < grid_size[1]; ++y)
        for (unsigned mr = 0; mr < grid_size[2]; ++mr)
          for (unsigned cb = 0; cb < grid_size[3]; ++cb)
            gvals_right[
              cb + grid_size[3] * (mr + grid_size[2] * (y + grid_size[1] * x))]
              = grid_value_encode(x, y, mr, cb);

    auto gvals =
      hpg::GridValueArray::copy_from(
        "v1",
        default_device,
        default_host_device,
        gvals_right.data(),
        grid_size,
        hpg::Layout::Right);

    bool eq = true;
    for (unsigned x = 0; eq && x < grid_size[0]; ++x)
      for (unsigned y = 0; eq && y < grid_size[1]; ++y)
        for (unsigned mr = 0; eq && mr < grid_size[2]; ++mr)
          for (unsigned cb = 0; eq && cb < grid_size[3]; ++cb) {
            auto val = grid_value_encode(x, y, mr, cb);
            EXPECT_EQ((*gvals)(x, y, mr, cb), val);
            eq = ((*gvals)(x, y, mr, cb) == val);
          }
  }
}

TEST(GridArrays, CopyFromWeightsLayouts) {
  std::array<unsigned, 2> grid_size{4, 3};

  auto gwgts_sz = grid_size[0] * grid_size[1];
  // copy grid weights from left layout, and check results
  {
    std::vector<hpg::GridWeightArray::scalar_type> gwgts_left(gwgts_sz);
    for (unsigned mr = 0; mr < grid_size[0]; ++mr)
      for (unsigned cb = 0; cb < grid_size[1]; ++cb)
        gwgts_left[mr + grid_size[0] * cb] = grid_weight_encode(mr, cb);
    auto gwgts =
      hpg::GridWeightArray::copy_from(
        "w0",
        default_device,
        default_host_device,
        gwgts_left.data(),
        grid_size,
        hpg::Layout::Left);

    bool eq = true;
    for (unsigned mr = 0; eq && mr < grid_size[0]; ++mr)
      for (unsigned cb = 0; eq && cb < grid_size[1]; ++cb) {
        auto wgt = grid_weight_encode(mr, cb);
        EXPECT_EQ((*gwgts)(mr, cb), wgt);
        eq = ((*gwgts)(mr, cb) == wgt);
      }
  }
  // copy grid weights to right layout, and check results
  {
    std::vector<hpg::GridWeightArray::scalar_type> gwgts_right(gwgts_sz);
    for (unsigned mr = 0; mr < grid_size[0]; ++mr)
      for (unsigned cb = 0; cb < grid_size[1]; ++cb)
        gwgts_right[cb + grid_size[1] * mr] = grid_weight_encode(mr, cb);

    auto gwgts =
      hpg::GridWeightArray::copy_from(
        "w1",
        default_device,
        default_host_device,
        gwgts_right.data(),
        grid_size,
        hpg::Layout::Right);

    bool eq = true;
    for (unsigned mr = 0; eq && mr < grid_size[0]; ++mr)
      for (unsigned cb = 0; eq && cb < grid_size[1]; ++cb) {
        auto wgt = grid_weight_encode(mr, cb);
        EXPECT_EQ((*gwgts)(mr, cb), wgt);
        eq = ((*gwgts)(mr, cb) == wgt);
      }
  }
}
TEST(GridArrays, CopyToWeightsLayouts) {
  std::array<unsigned, 4> grid_size{8, 8, 4, 3};
  std::array<hpg::grid_scale_fp, 2> grid_scale{0.1, -0.1};
  auto padding = 2 * hpg::CFArray::padding;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{
      {3 + padding, 3 + padding, 4, 3},
      {2 + padding, 2 + padding, 4, 2}};

  std::mt19937 rng(42);
  MyCFArray cf = create_cf(10, cf_sizes, rng);

  auto gs_or_err =
    hpg::GridderState::create(
      default_device,
      0,
      4,
      &cf,
      grid_size,
      grid_scale);
  ASSERT_TRUE(hpg::is_value(gs_or_err));
  auto gs = hpg::get_value(std::move(gs_or_err));
  auto gwgts = std::get<1>(gs.grid_weights());
  for (unsigned mr = 0; mr < grid_size[2]; ++mr)
    for (unsigned cb = 0; cb < grid_size[3]; ++cb)
      (*gwgts)(mr, cb) = grid_weight_encode(mr, cb);

  auto gwgts_sz = gwgts->min_buffer_size();
  // copy grid weights to left layout, and check results
  {
    std::vector<hpg::GridWeightArray::scalar_type> gwgts_left(gwgts_sz);
    gwgts->copy_to(default_host_device, gwgts_left.data(), hpg::Layout::Left);

    bool eq = true;
    for (unsigned mr = 0; eq && mr < grid_size[2]; ++mr)
      for (unsigned cb = 0; eq && cb < grid_size[3]; ++cb) {
        auto& l = gwgts_left[mr + grid_size[2] * cb];
        auto wgt = grid_weight_encode(mr, cb);
        EXPECT_EQ(l, wgt);
        eq = (l == wgt);
      }
  }
  // copy grid weights to right layout, and check results
  {
    std::vector<hpg::GridWeightArray::scalar_type> gwgts_right(gwgts_sz);
    gwgts->copy_to(default_host_device, gwgts_right.data(), hpg::Layout::Right);

    bool eq = true;
    for (unsigned mr = 0; eq && mr < grid_size[2]; ++mr)
      for (unsigned cb = 0; eq && cb < grid_size[3]; ++cb) {
        auto& r = gwgts_right[cb + grid_size[3] * mr];
        auto wgt = grid_weight_encode(mr, cb);
        EXPECT_EQ(r, wgt);
        eq = (r == wgt);
      }
  }
}

TEST(GridArrays, CompareLayouts) {
  std::array<unsigned, 4> grid_size{20, 20, 4, 3};
  std::array<hpg::grid_scale_fp, 2> grid_scale{0.1, -0.1};
  size_t num_vis = 100;
  auto padding = 2 * hpg::CFArray::padding;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{
      {3 + padding, 3 + padding, 4, 3},
      {2 + padding, 2 + padding, 4, 2}};

  std::mt19937 rng(42);
  MyCFArray cf = create_cf(10, cf_sizes, rng);

  std::vector<std::complex<hpg::visibility_fp>> vis;
  std::vector<unsigned> grid_cubes;
  std::vector<hpg::vis_cf_index_t> cf_indexes;
  std::vector<hpg::vis_weight_fp> weights;
  std::vector<hpg::vis_frequency_fp> frequencies;
  std::vector<hpg::vis_phase_fp> phases;
  std::vector<hpg::vis_uvw_t> coordinates;
  std::vector<hpg::cf_phase_screen_t> cf_phase_screens;
  init_visibilities(
    num_vis,
    grid_size,
    grid_scale,
    cf,
    rng,
    vis,
    grid_cubes,
    cf_indexes,
    weights,
    frequencies,
    phases,
    coordinates,
    cf_phase_screens);

  auto gridding =
    hpg::RvalM<void, hpg::GridderState>::pure(
      [&]() {
        return
          hpg::GridderState::create(
            default_device,
            0,
            num_vis,
            &cf,
            grid_size,
            grid_scale);
      })
    .and_then(
      [&](auto&& gs) mutable {
        return
          std::move(gs)
          .set_convolution_function(default_host_device, std::move(cf));
      })
    .and_then(
      [&](auto&& gs) mutable {
        return
          std::move(gs)
          .grid_visibilities(
            default_host_device,
            std::move(vis),
            std::move(grid_cubes),
            std::move(cf_indexes),
            std::move(weights),
            std::move(frequencies),
            std::move(phases),
            std::move(coordinates),
            std::move(cf_phase_screens));
      })
    .map(
      [](auto&& gs) {
        auto [gs1, values] = std::move(gs).grid_values();
        auto [gs2, weights] = std::move(gs1).grid_weights();
        return std::make_tuple(std::move(values), std::move(weights));
      });

  auto arrays_or_err = gridding();
  ASSERT_TRUE(hpg::is_value(arrays_or_err));
  auto [gvals, gwgts] = hpg::get_value(std::move(arrays_or_err));
  ASSERT_TRUE(bool(gvals));
  ASSERT_TRUE(bool(gwgts));
  ASSERT_TRUE(has_non_zero(gvals.get()));
  ASSERT_TRUE(has_non_zero(gwgts.get()));
  // copy gvals into arrays with left and right layouts
  auto gvals_sz = gvals->min_buffer_size();
  std::vector<hpg::GridValueArray::scalar_type> gvals_left(gvals_sz);
  std::vector<hpg::GridValueArray::scalar_type> gvals_right(gvals_sz);
  gvals->copy_to(default_host_device, gvals_left.data(), hpg::Layout::Left);
  gvals->copy_to(default_host_device, gvals_right.data(), hpg::Layout::Right);
  // copy gwgts into arrays with left and right layouts
  auto gwgts_sz = gwgts->min_buffer_size();
  std::vector<hpg::GridWeightArray::scalar_type> gwgts_left(gwgts_sz);
  std::vector<hpg::GridWeightArray::scalar_type> gwgts_right(gwgts_sz);
  gwgts->copy_to(default_host_device, gwgts_left.data(), hpg::Layout::Left);
  gwgts->copy_to(default_host_device, gwgts_right.data(), hpg::Layout::Right);
  // check equality of gvals and gwgts in all layouts
  bool eq = true;
  for (unsigned mr = 0; eq && mr < grid_size[2]; ++mr) {
    for (unsigned cb = 0; eq && cb < grid_size[3]; ++cb) {
      auto& wl = gwgts_left[mr + grid_size[2] * cb];
      auto& wr = gwgts_right[cb + grid_size[3] * mr];
      EXPECT_EQ(wl, wr);
      eq = (wl == wr) && (wl == (*gwgts)(mr, cb));
      if (!eq)
        std::cout << "weight at "
                  << mr << ","
                  << cb << ": "
                  << wl << " "
                  << wr << " "
                  << (*gwgts)(mr, cb) << std::endl;
      for (unsigned x = 0; eq && x < grid_size[0]; ++x) {
        for (unsigned y = 0; eq && y < grid_size[1]; ++y) {
          auto& vl =
            gvals_left[
              x + grid_size[0] * (y + grid_size[1] * (mr + grid_size[2] * cb))];
          auto& vr =
            gvals_right[
              cb + grid_size[3] * (mr + grid_size[2] * (y + grid_size[1] * x))];
          EXPECT_EQ(vl, vr);
          eq = (vl == vr) && (vl == (*gvals)(x, y, mr, cb));
          if (!eq)
            std::cout << "value at "
                      << x << ","
                      << y << ","
                      << mr << ","
                      << cb << ": "
                      << vl << " "
                      << vr << " "
                      << (*gvals)(x, y, mr, cb) << std::endl;
        }
      }
    }
  }
}

int
main(int argc, char **argv) {
  std::ostringstream oss;
  oss << "Using ";
  switch (default_device) {
#ifdef HPG_ENABLE_SERIAL
  case hpg::Device::Serial:
    oss << "Serial";
    break;
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
  case hpg::Device::OpenMP:
    oss << "OpenMP";
    break;
#endif // HPG_ENABLE_OPENMP
#ifdef HPG_ENABLE_CUDA
  case hpg::Device::Cuda:
    oss << "Cuda";
    break;
#endif // HPG_ENABLE_CUDA
  default:
    assert(false);
    break;
  }
  oss << " device for tests";
  std::cout << oss.str() << std::endl;

  ::testing::InitGoogleTest(&argc, argv);
  int rc;
  {
    // weird, but using ScopeGuard/initialize/finalize at function scope can
    // sometimes hang this program on exit (but not when executed by ctest)
    hpg::initialize();
    rc = RUN_ALL_TESTS();
    hpg::finalize();
  }
  return rc;
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End: