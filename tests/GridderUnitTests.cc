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
#include "hpg.hpp"
#include "gtest/gtest.h"

#include <array>
#include <cassert>
#include <complex>
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
    return unsigned(m_extents.size());
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
    return unsigned(m_extents.size());
  }

  std::array<unsigned, 4>
  extents(unsigned grp) const override {
    return m_extents[grp];
  }

  std::complex<hpg::cf_fp>
  operator()(
    unsigned x,
    unsigned y,
    unsigned mueller,
    unsigned channel,
    unsigned grp)
    const override {
    auto& vals = m_values[grp];
    auto& ext = m_extents[grp];
    return vals[((x * ext[1] + y) * ext[2] + mueller) * ext[3] + channel];
  }
};

struct ConeCFArray final
  : public hpg::CFArray {

  int m_nmueller;
  int m_oversampling;
  int m_radius;
  int m_oversampled_radius;

  ConeCFArray() {}

  ConeCFArray(unsigned nmueller, unsigned oversampling, unsigned radius)
    : m_nmueller(nmueller)
    , m_oversampling(oversampling)
    , m_radius(radius)
    , m_oversampled_radius((radius + padding) * oversampling) {
  }

  ConeCFArray(const ConeCFArray& other)
    : m_nmueller(other.m_nmueller)
    , m_oversampling(other.m_oversampling)
    , m_radius(other.m_radius)
    , m_oversampled_radius(other.m_oversampled_radius) {
  }

  unsigned
  oversampling() const override {
    return m_oversampling;
  }

  unsigned
  num_groups() const override {
    return 1;
  }

  std::array<unsigned, 4>
  extents(unsigned) const override {
    unsigned w = 2 * m_oversampled_radius + 1;
    return {w, w, unsigned(m_nmueller), 1};
  }

  std::complex<hpg::cf_fp>
  operator()(unsigned x, unsigned y, unsigned mueller, unsigned, unsigned)
    const override {

    std::complex<float> p(
      (-m_oversampled_radius + int(x)) + 0.5f,
      (-m_oversampled_radius + int(y)) + 0.5f);
    return
      std::polar(
        (mueller + 1) * std::max(m_oversampled_radius - std::abs(p), 0.0f),
        std::arg(std::abs(p.real()) + 1.0if * std::abs(p.imag())));
  }

  std::complex<hpg::grid_value_fp>
  operator()(unsigned x, unsigned y, unsigned mueller) const {

    return operator()(x, y, mueller, 0, 0);
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
    for (unsigned i = 0; i < num_values; ++i)
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
  std::vector<hpg::VisData<1>>& vis,
  std::vector<std::map<unsigned, hpg::vis_weight_fp>>& ch_maps) {

  vis.clear();
  vis.reserve(num_vis);
  ch_maps.clear();
  ch_maps.reserve(num_vis);

  const double inv_lambda = 9.75719;
  const double freq = 299792458.0 * inv_lambda;
  std::uniform_int_distribution<unsigned> dist_gchannel(0, grid_size[3] - 1);
  std::uniform_int_distribution<unsigned> dist_gcopol(0, grid_size[2] - 1);
  std::uniform_real_distribution<hpg::visibility_fp> dist_vis(-1.0, 1.0);
  std::uniform_real_distribution<hpg::vis_weight_fp> dist_weight(0.0, 1.0);
  std::uniform_int_distribution<unsigned> dist_cfgrp(0, cf.num_groups() - 1);
  auto x0 = (cf.oversampling() * (grid_size[0] - 2)) / 2;
  auto y0 = (cf.oversampling() * (grid_size[1] - 2)) / 2;
  double uscale = grid_scale[0] * cf.oversampling() * inv_lambda;
  double vscale = grid_scale[1] * cf.oversampling() * inv_lambda;
  for (unsigned i = 0; i < num_vis; ++i) {
    auto grp = dist_cfgrp(gen);
    auto cfextents = cf.extents(grp);
    std::uniform_int_distribution<unsigned> dist_cfchannel(0, cfextents[3] - 1);
    double ulim = (x0 - (cfextents[0]) / 2) / uscale;
    double vlim = (y0 - (cfextents[1]) / 2) / vscale;
    std::uniform_real_distribution<hpg::vis_uvw_fp> dist_u(-ulim, ulim);
    std::uniform_real_distribution<hpg::vis_uvw_fp> dist_v(-vlim, vlim);
    vis.push_back(
      hpg::VisData<1>(
        {std::complex<hpg::visibility_fp>(dist_vis(gen), dist_vis(gen))},
        freq,
        0.0,
        hpg::vis_uvw_t({dist_u(gen), dist_v(gen), 0.0}),
        {dist_cfchannel(gen), grp}));
    ch_maps.emplace_back(
      std::map<unsigned, hpg::vis_weight_fp>{
        {dist_gchannel(gen), dist_weight(gen)}});
  }
}

template <typename T>
bool
has_non_zero(const T* array) {
  if constexpr (T::rank == 2) {
    for (unsigned i = 0; i < array->extent(0); ++i)
      for (unsigned j = 0; j < array->extent(1); ++j)
        if ((*array)(i, j) != typename T::value_type(0))
          return true;
  } else {
    for (unsigned i = 0; i < array->extent(0); ++i)
      for (unsigned j = 0; j < array->extent(1); ++j)
        for (unsigned k = 0; k < array->extent(2); ++k)
          for (unsigned m = 0; m < array->extent(3); ++m)
            if ((*array)(i, j, k, m) != typename T::value_type(0))
              return true;
  }
  return false;
}

template <typename T>
bool
values_eq(const T* array0, const T* array1) {
  if constexpr (T::rank == 2) {
    if (array0->extent(0) != array1->extent(0)
        || array0->extent(1) != array1->extent(1))
      return false;
    for (unsigned i = 0; i < array0->extent(0); ++i)
      for (unsigned j = 0; j < array0->extent(1); ++j)
        if ((*array0)(i, j) != (*array1)(i, j))
          return false;
  } else {
    if (array0->extent(0) != array1->extent(0)
        || array0->extent(1) != array1->extent(1)
        || array0->extent(2) != array1->extent(2)
        || array0->extent(3) != array1->extent(3))
      return false;
    for (unsigned i = 0; i < array0->extent(0); ++i)
      for (unsigned j = 0; j < array0->extent(1); ++j)
        for (unsigned k = 0; k < array0->extent(2); ++k)
          for (unsigned m = 0; m < array0->extent(3); ++m)
            if ((*array0)(i, j, k, m) != (*array1)(i, j, k, m))
              return false;
  }
  return true;
}

// test that Gridder constructor arguments are properly set
TEST(Gridder, ConstructorArgs) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<hpg::grid_scale_fp, 2> grid_scale{0.12, -0.34};
  size_t batch_size = 21;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3, 3, 3, 3}, {2, 2, 2, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  hpg::Gridder g0;
  auto g1 =
    std::get<1>(
      hpg::Gridder::create<1>(
        default_device,
        0,
        0,
        batch_size,
        1,
        &cf,
        grid_size,
        grid_scale,
        {{0}, {0}, {0}, {0}},
        {{0}, {0}, {0}, {0}}));

  EXPECT_TRUE(g0.is_null());
  EXPECT_FALSE(g1.is_null());
  EXPECT_EQ(g1.device(), default_device);
  EXPECT_EQ(g1.grid_size(), grid_size);
  EXPECT_EQ(g1.grid_scale(), grid_scale);
  EXPECT_EQ(g1.num_active_tasks(), 1);
  EXPECT_EQ(g1.num_contexts(), 1);
  EXPECT_EQ(g1.visibility_batch_size(), batch_size);
  auto sz_or_err = g1.current_convolution_function_region_size();
  ASSERT_TRUE(hpg::is_value(sz_or_err));
  EXPECT_EQ(hpg::get_value(sz_or_err), g1.convolution_function_region_size(cf));
}

// test that Gridder copies have correct parameters
TEST(Gridder, Copies) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<hpg::grid_scale_fp, 2> grid_scale{0.12, -0.34};
  size_t batch_size = 31;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3, 3, 3, 3}, {2, 2, 2, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  auto g0 =
    std::get<1>(
      hpg::Gridder::create<1>(
        default_device,
        0,
        0,
        batch_size,
        1,
        &cf,
        grid_size,
        grid_scale,
        {{0}, {0}, {0}, {0}},
        {{0}, {0}, {0}, {0}}));
  hpg::Gridder g1 = g0;

  EXPECT_FALSE(g0.is_null());
  EXPECT_EQ(g1.device(), default_device);
  EXPECT_EQ(g1.grid_size(), grid_size);
  EXPECT_EQ(g1.grid_scale(), grid_scale);
  EXPECT_EQ(g1.visibility_batch_size(), batch_size);
  auto sz0_or_err = g0.current_convolution_function_region_size();
  auto sz1_or_err = g1.current_convolution_function_region_size();
  ASSERT_TRUE(hpg::is_value(sz0_or_err));
  ASSERT_TRUE(hpg::is_value(sz1_or_err));
  EXPECT_EQ(hpg::get_value(sz1_or_err), hpg::get_value(sz0_or_err));

  hpg::Gridder g2(g0);
  EXPECT_FALSE(g0.is_null());
  EXPECT_EQ(g2.device(), default_device);
  EXPECT_EQ(g2.grid_size(), grid_size);
  EXPECT_EQ(g2.grid_scale(), grid_scale);
  EXPECT_EQ(g2.visibility_batch_size(), batch_size);
  auto sz2_or_err = g2.current_convolution_function_region_size();
  ASSERT_TRUE(hpg::is_value(sz2_or_err));
  EXPECT_EQ(hpg::get_value(sz2_or_err), hpg::get_value(sz0_or_err));
}

// test that Gridder moves have expected outcomes
TEST(Gridder, Moves) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<hpg::grid_scale_fp, 2> grid_scale{0.12, -0.34};
  size_t batch_size = 11;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3, 3, 3, 3}, {2, 2, 2, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  auto g0 =
    std::get<1>(
      hpg::Gridder::create<1>(
        default_device,
        0,
        0,
        batch_size,
        1,
        &cf,
        grid_size,
        grid_scale,
        {{0}, {0}, {0}, {0}},
        {{0}, {0}, {0}, {0}}));
  auto cf_region_sz =
    hpg::get_value(g0.current_convolution_function_region_size());
  hpg::Gridder g1 = std::move(g0);

  EXPECT_TRUE(g0.is_null());
  EXPECT_EQ(g1.device(), default_device);
  EXPECT_EQ(g1.grid_size(), grid_size);
  EXPECT_EQ(g1.grid_scale(), grid_scale);
  EXPECT_EQ(g1.visibility_batch_size(), batch_size);
  EXPECT_EQ(
    hpg::get_value(g1.current_convolution_function_region_size()),
    cf_region_sz);

  hpg::Gridder g2(std::move(g1));
  EXPECT_TRUE(g1.is_null());
  EXPECT_EQ(g2.device(), default_device);
  EXPECT_EQ(g2.grid_size(), grid_size);
  EXPECT_EQ(g2.grid_scale(), grid_scale);
  EXPECT_EQ(g2.visibility_batch_size(), batch_size);
  EXPECT_EQ(
    hpg::get_value(g2.current_convolution_function_region_size()),
    cf_region_sz);
}

// test that Gridder grid values and weights are properly initialized
TEST(Gridder, InitValues) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<hpg::grid_scale_fp, 2> grid_scale{0.12, -0.34};
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3, 3, 3, 3}, {2, 2, 2, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  auto g =
    std::get<1>(
      hpg::Gridder::create<1>(
        default_device,
        0,
        0,
        10,
        1,
        &cf,
        grid_size,
        grid_scale,
        {{0}, {0}, {0}, {0}},
        {{0}, {0}, {0}, {0}}));

  auto values = g.grid_values();
  for (size_t i = 0; i < 4; ++i)
    EXPECT_EQ(values->extent(i), grid_size[i]);
  EXPECT_FALSE(has_non_zero(values.get()));

  auto weights = g.grid_weights();
  for (size_t i = 2; i < 4; ++i)
    EXPECT_EQ(weights->extent(i - 2), grid_size[i]);
  EXPECT_FALSE(has_non_zero(weights.get()));
}

// tests for Gridder::set_convolution_function()
TEST(Gridder, CF) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<hpg::grid_scale_fp, 2> grid_scale{0.1, -0.1};
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3, 3, 3, 3}, {2, 2, 2, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  auto g =
    std::get<1>(
      hpg::Gridder::create<1>(
        default_device,
        0,
        0,
        22,
        1,
        &cf,
        grid_size,
        grid_scale,
        {{0}, {0}, {0}, {0}},
        {{0}, {0}, {0}, {0}}));

  std::mt19937 rng(42);

  {
    MyCFArray cf = create_cf(10, cf_sizes, rng);
    auto oerr = g.set_convolution_function(default_host_device, MyCFArray(cf));
    EXPECT_FALSE(bool(oerr));
    // do it again
    auto oerr1 = g.set_convolution_function(default_host_device, std::move(cf));
    EXPECT_FALSE(bool(oerr1));
  }
}

// test that Gridder::reset_grid() correctly resets grid weights and values
TEST(Gridder, Reset) {
  std::array<unsigned, 4> grid_size{16, 15, 1, 3};
  std::array<hpg::grid_scale_fp, 2> grid_scale{0.1, -0.1};
  size_t num_vis = 10;
  auto padding = 2 * hpg::CFArray::padding;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{
      {3 + padding, 3 + padding, 1, 3},
      {2 + padding, 2 + padding, 1, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  auto g =
    std::get<1>(
      hpg::Gridder::create<1>(
        default_device,
        0,
        0,
        num_vis,
        1,
        &cf,
        grid_size,
        grid_scale,
        {{0}},
        {{0}}));

  std::mt19937 rng(42);
  std::vector<hpg::VisData<1>> vis;
  std::vector<std::map<unsigned, hpg::vis_weight_fp>> ch_maps;
  {
    const std::array<unsigned, 4> cf_size{3 + padding, 3 + padding, 1, 3};
    MyCFArray cf = create_cf(10, {cf_size}, rng);
    g.set_convolution_function(default_host_device, MyCFArray(cf));
    init_visibilities(num_vis, grid_size, grid_scale, cf, rng, vis, ch_maps);
    g.grid_visibilities(default_host_device, std::move(vis), ch_maps, true);

    auto values = g.grid_values();
    EXPECT_TRUE(has_non_zero(values.get()));
    auto weights = g.grid_weights();
    EXPECT_TRUE(has_non_zero(weights.get()));

    g.reset_grid();
    auto values1 = g.grid_values();
    EXPECT_FALSE(has_non_zero(values1.get()));
    auto weights1 = g.grid_weights();
    EXPECT_FALSE(has_non_zero(weights1.get()));
  }
}

// test residual visibility return functionality
TEST(Gridder, ResidualVisibilities) {
  const std::array<unsigned, 4> grid_size{20, 20, 2, 1};
  const std::array<hpg::grid_scale_fp, 2> grid_scale{0.0476591, 0.0476591};
  constexpr unsigned cf_radius = 3;
  constexpr unsigned cf_oversampling = 10;
  constexpr hpg::vis_frequency_fp freq = 3.693e+09;
  constexpr std::complex<hpg::visibility_fp> vis(1.0, -1.0);

  ConeCFArray cf(1, cf_oversampling, cf_radius);
  hpg::Gridder g;
  {
    auto g_or_err =
      hpg::Gridder::create<2>(
        default_device,
        0,
        0,
        1,
        1,
        &cf,
        grid_size,
        grid_scale,
        {{0, -1}, {-1, 0}},
        {{0, -1}, {-1, 0}}
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
        , impl_versions
#endif
        );
    ASSERT_TRUE(hpg::is_value(g_or_err));
    g = hpg::get_value(std::move(g_or_err));
    auto opt_err =
      g.set_convolution_function(default_host_device, ConeCFArray(cf));
    ASSERT_FALSE(opt_err);
  }

  auto visibilities =
    std::vector<hpg::VisData<2>>{
      hpg::VisData<2>({vis, -vis}, freq, 0.0, {0.0, 0.0, 0.1}, {0, 0})};

  {
    // do gridding
    std::vector<std::map<unsigned, hpg::vis_weight_fp>> ch_maps{{{0, 1.0}}};
    auto fv_or_err =
      g.degrid_grid_get_residual_visibilities(
        default_host_device,
        std::vector<hpg::VisData<2>>(visibilities),
        ch_maps);
    ASSERT_TRUE(hpg::is_value(fv_or_err));
    auto fv = hpg::get_value(std::move(fv_or_err));
    // one try before future has a chance to complete
    ASSERT_FALSE(fv.get());
    // ensure that future completes
    g.fence();
    // get result of future
    auto orv = fv.get();
    ASSERT_TRUE(orv);
#if HPG_API >= 17
    // get the residual visibilities
    auto resvis = std::move(orv).value();
#else
    auto resvis = std::move(*orv);
#endif
    EXPECT_TRUE(
      std::mismatch(
        resvis.begin(), resvis.end(),
        visibilities.begin(), visibilities.end())
      == std::make_pair(resvis.end(), visibilities.end()));
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
    hpg::InitArguments args;
    args.cleanup_fftw = true;
    hpg::initialize(args);
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
