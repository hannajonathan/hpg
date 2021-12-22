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
#define HPG_INTERNAL
#include "hpg.hpp"
#include "gtest/gtest.h"

#include <H5Cpp.h>

#include <array>
#include <cassert>
#include <complex>
#include <experimental/array>
#include <iostream>
#include <filesystem>
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
using namespace std::string_literals;

#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
# ifndef HPG_GRIDDING_KERNEL_VERSION
#  define HPG_GRIDDING_KERNEL_VERSION 0
# endif
static constexpr std::array<unsigned, 4>
  impl_versions{HPG_GRIDDING_KERNEL_VERSION, 0, 0, 0};
#else
# undef HPG_DELTA_EXPERIMENTAL_ONLY
#endif

#undef SHOW_COORDINATES

std::filesystem::path exec_dir;
bool write_expected_data_files;

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

  std::array<std::array<long, 2>, 3>
  coords(
    const std::array<unsigned, 4>& grid_size,
    const std::array<hpg::grid_scale_fp, 2>& grid_scale,
    const hpg::vis_frequency_fp& freq,
    const hpg::vis_uvw_t& uvw) const {

    constexpr hpg::vis_frequency_fp c = 299792458.0;
#ifdef SHOW_COORDINATES
    std::cout << "uvw: " << uvw[0]
              << ", " << uvw[1]
              << ", " << uvw[2]
              << std::endl
              << "scale: " << grid_scale[0]
              << ", " << grid_scale[1]
              << std::endl
              << "size: " << grid_size[0]
              << ", " << grid_size[1]
              << std::endl
              << "freq: " << freq
              << std::endl;
#endif
    const std::array<double, 2> position{
      grid_scale[0] * uvw[0] * (freq / c) + grid_size[0] / 2.0,
      grid_scale[1] * uvw[1] * (freq / c) + grid_size[1] / 2.0
    };
#ifdef SHOW_COORDINATES
    std::cout << "position: " << position[0]
              << ", " << position[1] << std::endl;
#endif
    std::array<long, 2>
      grid_coord{std::lrint(position[0]), std::lrint(position[1])};
#ifdef SHOW_COORDINATES
    std::cout << "grid_coord (0): " << grid_coord[0]
              << ", " << grid_coord[1] << std::endl;
#endif
    const std::array<long, 2> fine_offset{
      std::lrint((grid_coord[0] - position[0]) * m_oversampling),
      std::lrint((grid_coord[1] - position[1]) * m_oversampling)
    };
#ifdef SHOW_COORDINATES
    std::cout << "fine_offset: " << fine_offset[0]
              << ", " << fine_offset[1] << std::endl;
#endif
    std::array<long, 2> cf_major;
    grid_coord[0] -= m_radius;
    grid_coord[1] -= m_radius;
#ifdef SHOW_COORDINATES
    std::cout << "grid_coord (1): " << grid_coord[0]
              << ", " << grid_coord[1] << std::endl;
#endif
    std::array<long, 2> cf_minor;
    if (fine_offset[0] >= 0) {
      cf_minor[0] = fine_offset[0];
      cf_major[0] = hpg::CFArray::padding;
    } else {
      cf_minor[0] = m_oversampling + fine_offset[0];
      cf_major[0] = hpg::CFArray::padding - 1;
    }
    if (fine_offset[1] >= 0) {
      cf_minor[1] = fine_offset[1];
      cf_major[1] = hpg::CFArray::padding;
    } else {
      cf_minor[1] = m_oversampling + fine_offset[1];
      cf_major[1] = hpg::CFArray::padding - 1;
    }
#ifdef SHOW_COORDINATES
    std::cout << "cf_major: " << cf_major[0]
              << ", " << cf_major[1] << std::endl;
    std::cout << "cf_minor: " << cf_minor[0]
              << ", " << cf_minor[1] << std::endl;
#endif
    return {grid_coord, cf_major, cf_minor};
  }

  bool
  verify_gridded_vis(
    const hpg::GridValueArray* grid,
    const std::array<unsigned, 4>& grid_size,
    const std::array<hpg::grid_scale_fp, 2>& grid_scale,
    const std::complex<hpg::visibility_fp>& vis,
    const hpg::vis_frequency_fp& freq,
    const hpg::vis_uvw_t& uvw) const {

    auto [grid_coord, cf_major, cf_minor] =
      coords(grid_size, grid_scale, freq, uvw);

    std::complex<hpg::grid_value_fp> gvis(vis);
    bool result = true;
    for (unsigned x = 0; result && x < grid_size[0]; ++x)
      for (unsigned  y = 0; result && y < grid_size[1]; ++y) {
        auto g = (*grid)(x, y, 0, 0);
        std::complex<hpg::grid_value_fp> cf;
        if (grid_coord[0] <= x && x < grid_coord[0] + 2 * m_radius + 1
            && grid_coord[1] <= y && y < grid_coord[1] + 2 * m_radius + 1) {
          int xf =
            (x - grid_coord[0] + cf_major[0]) * int(m_oversampling)
            + cf_minor[0];
          int yf =
            (y - grid_coord[1] + cf_major[1]) * int(m_oversampling)
            + cf_minor[1];
          cf = (*this)(xf, yf, 0);
        }
        cf *= gvis;
        result = near(g, cf);
        if (!result)
          std::cout << "at " << x << "," << y
                    << ": " << cf << ";" << g
                    << std::endl;
      }
    return result;
  }

  std::array<std::array<int, 2>, 2>
  cf_footprint(
    const std::array<unsigned, 4>& grid_size,
    const std::array<hpg::grid_scale_fp, 2>& grid_scale,
    const hpg::vis_frequency_fp& freq,
    const hpg::vis_uvw_t& uvw) const {

    auto grid_coord = std::get<0>(coords(grid_size, grid_scale, freq, uvw));
    return {
      std::array<int, 2>{int(grid_coord[0]), int(grid_coord[1])},
      std::array<int, 2>{
        int(grid_coord[0]) + 2 * m_radius,
        int(grid_coord[1]) + 2 * m_radius}};
  }

  bool
  verify_cf_footprint(
    unsigned mrow,
    const hpg::GridValueArray* grid,
    const std::array<unsigned, 4>& grid_size,
    const std::array<hpg::grid_scale_fp, 2>& grid_scale,
    const std::complex<hpg::visibility_fp>& vis,
    const hpg::vis_frequency_fp& freq,
    const hpg::vis_uvw_t& uvw) const {

    std::array<std::array<int, 2>, 2> measured_footprint{
      std::array<int, 2>{2 * int(grid_size[0]), 2 * int(grid_size[1])},
      std::array<int, 2>{-1, -1}
    };
    for (int x = 0; x < int(grid_size[0]); ++x)
      for (int y = 0; y < int(grid_size[1]); ++y)
        if ((*grid)(x, y, mrow, 0) != hpg::GridValueArray::value_type(0)) {
          measured_footprint[0][0] = std::min(measured_footprint[0][0], x);
          measured_footprint[0][1] = std::min(measured_footprint[0][1], y);
          measured_footprint[1][0] = std::max(measured_footprint[1][0], x);
          measured_footprint[1][1] = std::max(measured_footprint[1][1], y);
        }
    return measured_footprint == cf_footprint(grid_size, grid_scale, freq, uvw);
  }
};

class AnyGridValueArray
  : public hpg::GridValueArray {
public:

  std::array<unsigned, rank> m_extents;
  value_type m_value;

  AnyGridValueArray(const std::array<unsigned, rank>& extents)
    : m_extents(extents)
    , m_value(0) {}

  unsigned
  extent(unsigned dim) const override {
    return m_extents[dim];
  }

  const value_type&
  operator()(unsigned, unsigned, unsigned, unsigned) const override {
    return m_value;
  }

  value_type&
  operator()(unsigned, unsigned, unsigned, unsigned) override {
    return m_value;
  }

  ~AnyGridValueArray() {}

protected:

  void
  unsafe_copy_to(hpg::Device, value_type* dst, hpg::Layout) const override {
    for (size_t i = 0; i < min_buffer_size(); ++i)
      *dst++ = m_value;
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
  std::vector<std::map<unsigned, hpg::vis_weight_fp>>& ch_maps,
  unsigned channels_per_vis = 1) {

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
  std::uniform_real_distribution<hpg::cf_phase_gradient_fp>
    dist_cfgrad(-3.141, 3.141);
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
        {dist_cfchannel(gen), grp},
        {dist_cfgrad(gen), dist_cfgrad(gen)}));
    std::map<unsigned, hpg::vis_weight_fp> vis_ch_map;
    for (unsigned j = 0; j < channels_per_vis; ++j)
      vis_ch_map[dist_gchannel(gen)] = dist_weight(gen);
    ch_maps.push_back(std::move(vis_ch_map));
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
        || array0->extent(1) != array1->extent(1)) {
      std::cerr << "extents differ" << std::endl;
      return false;
    }
    for (unsigned i = 0; i < array0->extent(0); ++i)
      for (unsigned j = 0; j < array0->extent(1); ++j)
        if (!near((*array0)(i, j), (*array1)(i, j))) {
          std::cerr << "values differ at "
                    << i << "," << j
                    << "; " << (*array0)(i, j)
                    << " != " << (*array1)(i, j)
                    << std::endl;
          return false;
        }
    } else {
    if (array0->extent(0) != array1->extent(0)
        || array0->extent(1) != array1->extent(1)
        || array0->extent(2) != array1->extent(2)
        || array0->extent(3) != array1->extent(3)) {
      std::cerr << "extents differ" << std::endl;
      return false;
    }
    for (unsigned i = 0; i < array0->extent(0); ++i)
      for (unsigned j = 0; j < array0->extent(1); ++j)
        for (unsigned k = 0; k < array0->extent(2); ++k)
          for (unsigned m = 0; m < array0->extent(3); ++m)
            if (!near((*array0)(i, j, k, m), (*array1)(i, j, k, m))) {
              std::cerr << "values differ at "
                        << i << "," << j << "," << k << "," << m
                        << "; " << (*array0)(i, j, k, m)
                        << " != " << (*array1)(i, j, k, m)
                        << std::endl;
              return false;
            }
  }
  return true;
}

#ifndef HPG_DELTA_EXPERIMENTAL_ONLY
// test that GridderState  constructor arguments are properly set
TEST(GridderState, ConstructorArgs) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<hpg::grid_scale_fp, 2> grid_scale{0.12, -0.34};
  size_t batch_size = 20;
  auto padding = 2 * hpg::CFArray::padding;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3 + padding, 3 + padding, 3, 3}, {2 + padding, 2 + padding, 2, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  unsigned max_avg_channels_per_vis = 2;
  hpg::GridderState gs0;
  auto gs1_or_err =
    hpg::GridderState::create<1>(
      default_device,
      0,
      0,
      batch_size,
      max_avg_channels_per_vis,
      &cf,
      grid_size,
      grid_scale,
      {{0}, {0}, {0}, {0}},
      {{0}, {0}, {0}, {0}});
  ASSERT_TRUE(hpg::is_value(gs1_or_err));
  auto gs1 = hpg::get_value(std::move(gs1_or_err));

  EXPECT_TRUE(gs0.is_null());
  EXPECT_FALSE(gs1.is_null());
  EXPECT_EQ(gs1.device(), default_device);
  EXPECT_EQ(gs1.grid_size(), grid_size);
  EXPECT_EQ(gs1.grid_scale(), grid_scale);
  EXPECT_EQ(gs1.num_polarizations(), 1);
  EXPECT_EQ(gs1.num_active_tasks(), 1);
  EXPECT_EQ(gs1.num_contexts(), 1);
  EXPECT_EQ(gs1.visibility_batch_size(), batch_size);
  EXPECT_EQ(gs1.max_avg_channels_per_vis(), max_avg_channels_per_vis);
  EXPECT_EQ(
    gs1.convolution_function_region_size(nullptr),
    gs1.convolution_function_region_size(&cf));

  auto gs2_or_err =
    hpg::GridderState::create<1>(
      default_device,
      0,
      0,
      batch_size,
      max_avg_channels_per_vis,
      &cf,
      grid_size,
      grid_scale,
      {{0}, {0}, {0}},
      {{0}, {0}, {0}, {0}});
  EXPECT_TRUE(hpg::is_error(gs2_or_err));
  EXPECT_EQ(
    hpg::get_error(std::move(gs2_or_err))->type(),
    hpg::ErrorType::InvalidNumberMuellerIndexRows);

  auto gs3_or_err =
    hpg::GridderState::create<1>(
      default_device,
      0,
      0,
      batch_size,
      max_avg_channels_per_vis,
      &cf,
      grid_size,
      grid_scale,
      {{0}, {0}, {0}, {0}},
      {{0}, {0}, {0}});
  EXPECT_TRUE(hpg::is_error(gs3_or_err));
  EXPECT_EQ(
    hpg::get_error(std::move(gs3_or_err))->type(),
    hpg::ErrorType::InvalidNumberMuellerIndexRows);
}
#endif // HPG_DELTA_EXPERIMENTAL_ONLY

#ifndef HPG_DELTA_EXPERIMENTAL_ONLY
// test that GridderState copies have correct parameters
TEST(GridderState, Copies) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<hpg::grid_scale_fp, 2> grid_scale{0.12, -0.34};
  size_t batch_size = 20;
  auto padding = 2 * hpg::CFArray::padding;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3 + padding, 3 + padding, 3, 3}, {2 + padding, 2 + padding, 2, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  unsigned max_avg_channels_per_vis = 3;
  auto gs0 =
    hpg::get_value(
      hpg::GridderState::create<1>(
        default_device,
        0,
        0,
        batch_size,
        max_avg_channels_per_vis,
        &cf,
        grid_size,
        grid_scale,
        {{0}, {0}, {0}, {0}},
        {{0}, {0}, {0}, {0}}));
  hpg::GridderState gs1 = gs0;

  EXPECT_FALSE(gs0.is_null());
  EXPECT_EQ(gs1.device(), default_device);
  EXPECT_EQ(gs1.grid_size(), grid_size);
  EXPECT_EQ(gs1.grid_scale(), grid_scale);
  EXPECT_EQ(gs1.num_polarizations(), 1);
  EXPECT_EQ(gs1.visibility_batch_size(), batch_size);
  EXPECT_EQ(gs1.max_avg_channels_per_vis(), max_avg_channels_per_vis);
  EXPECT_EQ(
    gs0.convolution_function_region_size(nullptr),
    gs1.convolution_function_region_size(nullptr));

  hpg::GridderState gs2(gs0);
  EXPECT_FALSE(gs0.is_null());
  EXPECT_EQ(gs2.device(), default_device);
  EXPECT_EQ(gs2.grid_size(), grid_size);
  EXPECT_EQ(gs2.grid_scale(), grid_scale);
  EXPECT_EQ(gs2.num_polarizations(), 1);
  EXPECT_EQ(gs2.visibility_batch_size(), batch_size);
  EXPECT_EQ(gs2.max_avg_channels_per_vis(), max_avg_channels_per_vis);
  EXPECT_EQ(
    gs0.convolution_function_region_size(nullptr),
    gs2.convolution_function_region_size(nullptr));
}
#endif // HPG_DELTA_EXPERIMENTAL_ONLY

#ifndef HPG_DELTA_EXPERIMENTAL_ONLY
// test that GridderState moves have expected outcomes
TEST(GridderState, Moves) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<hpg::grid_scale_fp, 2> grid_scale{0.12, -0.34};
  size_t batch_size = 30;
  auto padding = 2 * hpg::CFArray::padding;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3 + padding, 3 + padding, 3, 3}, {2 + padding, 2 + padding, 2, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  auto gs0 =
    hpg::get_value(
      hpg::GridderState::create<1>(
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
  auto cf_region_size = gs0.convolution_function_region_size(nullptr);
  hpg::GridderState gs1 = std::move(gs0);

  EXPECT_TRUE(gs0.is_null());
  EXPECT_EQ(gs1.device(), default_device);
  EXPECT_EQ(gs1.grid_size(), grid_size);
  EXPECT_EQ(gs1.grid_scale(), grid_scale);
  EXPECT_EQ(gs1.num_polarizations(), 1);
  EXPECT_EQ(gs1.visibility_batch_size(), batch_size);
  EXPECT_EQ(gs1.convolution_function_region_size(nullptr), cf_region_size);

  hpg::GridderState gs2(std::move(gs1));
  EXPECT_TRUE(gs1.is_null());
  EXPECT_EQ(gs2.device(), default_device);
  EXPECT_EQ(gs2.grid_size(), grid_size);
  EXPECT_EQ(gs2.grid_scale(), grid_scale);
  EXPECT_EQ(gs2.num_polarizations(), 1);
  EXPECT_EQ(gs2.visibility_batch_size(), batch_size);
  EXPECT_EQ(gs2.convolution_function_region_size(nullptr), cf_region_size);
}
#endif //HPG_DELTA_EXPERIMENTAL_ONLY

#ifndef HPG_DELTA_EXPERIMENTAL_ONLY
// test that GridderState grid values and weights are properly initialized
TEST(GridderState, InitValues) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<hpg::grid_scale_fp, 2> grid_scale{0.12, -0.34};
  auto padding = 2 * hpg::CFArray::padding;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3 + padding, 3 + padding, 3, 3}, {2 + padding, 2 + padding, 2, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  auto gs =
    hpg::get_value(
      hpg::GridderState::create<1>(
        default_device,
        0,
        0,
        15,
        1,
        &cf,
        grid_size,
        grid_scale,
        {{0}, {0}, {0}, {0}},
        {{0}, {0}, {0}, {0}}));

  auto [gs1, values] = std::move(gs).grid_values();
  for (size_t i = 0; i < 4; ++i)
    EXPECT_EQ(values->extent(i), grid_size[i]);
  EXPECT_FALSE(has_non_zero(values.get()));

  auto [gs2, weights] = std::move(gs1).grid_weights();
  for (size_t i = 2; i < 4; ++i)
    EXPECT_EQ(weights->extent(i - 2), grid_size[i]);
  EXPECT_FALSE(has_non_zero(weights.get()));
}
#endif // HPG_DELTA_EXPERIMENTAL_ONLY

#ifndef HPG_DELTA_EXPERIMENTAL_ONLY
// test that model setting works properly
TEST(GridderState, SetModel) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<hpg::grid_scale_fp, 2> grid_scale{0.12, -0.34};
  auto padding = 2 * hpg::CFArray::padding;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3 + padding, 3 + padding, 3, 3}, {2 + padding, 2 + padding, 2, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  auto gs =
    hpg::get_value(
      hpg::GridderState::create<1>(
        default_device,
        0,
        0,
        15,
        1,
        &cf,
        grid_size,
        grid_scale,
        {{0}, {0}, {0}, {0}},
        {{0}, {0}, {0}, {0}}));

  // easiest way to create a model is to get a copy of the grid
  auto [gs1, values] = std::move(gs).grid_values();
  auto gs2_or_err =
    std::move(gs1).set_model(default_host_device, std::move(*values));
  EXPECT_TRUE(hpg::is_value(gs2_or_err));
  auto gs2 = hpg::get_value(gs2_or_err);

  // create an AnyGridValueArray of the wrong size
  auto bad_grid_size = grid_size;
  bad_grid_size[0]--;
  AnyGridValueArray bad_grid(bad_grid_size);
  auto gs3_or_err =
    std::move(gs2).set_model(default_host_device, std::move(bad_grid));
  ASSERT_TRUE(hpg::is_error(gs3_or_err));
  EXPECT_EQ(
    hpg::get_error(std::move(gs3_or_err))->type(),
    hpg::ErrorType::InvalidModelGridSize);
}
#endif // HPG_DELTA_EXPERIMENTAL_ONLY

#ifndef HPG_DELTA_EXPERIMENTAL_ONLY
// test that GridderState methods have correct copy and move semantics
TEST(GridderState, CopyOrMove) {
  std::array<unsigned, 4> grid_size{15, 14, 1, 3};
  std::array<hpg::grid_scale_fp, 2> grid_scale{0.1, -0.1};
  size_t num_vis = 10;
  auto padding = 2 * hpg::CFArray::padding;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3 + padding, 3 + padding, 1, 3}, {2 + padding, 2 + padding, 1, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  auto gs =
    hpg::get_value(
      hpg::GridderState::create<1>(
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
    MyCFArray cf = create_cf(10, cf_sizes, rng);
    auto gs1 =
      hpg::get_value(
        gs.set_convolution_function(default_host_device, MyCFArray(cf)));
    init_visibilities(num_vis, grid_size, grid_scale, cf, rng, vis, ch_maps);
    auto gs2_or_err =
      gs1.grid_visibilities(default_host_device, decltype(vis)(vis), ch_maps);
    ASSERT_TRUE(hpg::is_value(gs2_or_err));
    auto gs2 = hpg::get_value(std::move(gs2_or_err));

    // gridded visibilities should be in gs2, not gs1
    auto [gs3, values] = std::move(gs1).grid_values();
    for (size_t i = 0; i < 4; ++i)
      EXPECT_EQ(values->extent(i), grid_size[i]);
    EXPECT_FALSE(has_non_zero(values.get()));

    auto gs4 =
      hpg::get_value(
        std::move(gs3)
        .grid_visibilities(default_host_device, std::move(vis), ch_maps));

    // gs2 and gs4 should have same grid values
    auto [gs5, values5] = std::move(gs2).grid_values();
    EXPECT_TRUE(has_non_zero(values5.get()));
    auto [gs6, values6] = std::move(gs4).grid_values();
    EXPECT_TRUE(values_eq(values5.get(), values6.get()));
  }
  {
    auto [gs1, v1] = gs.grid_values();
    EXPECT_FALSE(gs.is_null());
    auto [gs2, v2] = std::move(gs1).grid_values();
    EXPECT_TRUE(gs1.is_null());
    EXPECT_FALSE(gs2.is_null());
  }
  {
    auto [gs1, w1] = gs.grid_weights();
    EXPECT_FALSE(gs.is_null());
    auto [gs2, w2] = std::move(gs1).grid_weights();
    EXPECT_TRUE(gs1.is_null());
    EXPECT_FALSE(gs2.is_null());
  }
  {
    auto rc_fft = gs.apply_grid_fft();
    EXPECT_FALSE(gs.is_null());
    ASSERT_TRUE(hpg::is_value(rc_fft));
    hpg::GridderState gs1 = hpg::get_value(std::move(rc_fft));
    auto err_or_gs2 = std::move(gs1).apply_grid_fft();
    EXPECT_TRUE(gs1.is_null());

    ASSERT_TRUE(hpg::is_value(err_or_gs2));
    EXPECT_FALSE(hpg::get_value(err_or_gs2).is_null());
  }
  {
    auto gs1 = gs.shift_grid(hpg::ShiftDirection::FORWARD);
    EXPECT_FALSE(gs.is_null());
    auto gs2 = std::move(gs1).shift_grid(hpg::ShiftDirection::FORWARD);
    EXPECT_TRUE(gs1.is_null());
    EXPECT_FALSE(gs2.is_null());
  }
}
#endif // HPG_DELTA_EXPERIMENTAL_ONLY

#ifndef HPG_DELTA_EXPERIMENTAL_ONLY
// test that GridderState::reset_grid() correctly resets grid weights and values
// for both copy and move method varieties
TEST(GridderState, Reset) {
  std::array<unsigned, 4> grid_size{16, 15, 1, 3};
  std::array<hpg::grid_scale_fp, 2> grid_scale{0.1, -0.1};
  size_t num_vis = 10;
  auto padding = 2 * hpg::CFArray::padding;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3 + padding, 3 + padding, 1, 3}, {2 + padding, 2 + padding, 1, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  auto gs =
    hpg::get_value(
      hpg::GridderState::create<1>(
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
    auto gs1 =
      hpg::get_value(
        gs.set_convolution_function(default_host_device, MyCFArray(cf)));
    init_visibilities(num_vis, grid_size, grid_scale, cf, rng, vis, ch_maps);

    auto gs2 =
      hpg::get_value(
        std::move(gs1)
        .grid_visibilities(default_host_device, std::move(vis), ch_maps, true));

    auto [gs3, values] = std::move(gs2).grid_values();
    EXPECT_TRUE(has_non_zero(values.get()));
    auto [gs4, weights] = std::move(gs3).grid_weights();
    EXPECT_TRUE(has_non_zero(weights.get()));

    auto gs5 = gs4.reset_grid();
    auto [gs6, values6] = gs4.grid_values();
    EXPECT_TRUE(values_eq(values.get(), values6.get()));
    auto [gs7, values7] = gs5.grid_values();
    EXPECT_FALSE(has_non_zero(values7.get()));

    auto [gs8, weights8] = gs4.grid_weights();
    EXPECT_TRUE(values_eq(weights.get(), weights8.get()));
    auto [gs9, weights9] = gs5.grid_weights();
    EXPECT_FALSE(has_non_zero(weights9.get()));
  }
}
#endif // HPG_DELTA_EXPERIMENTAL_ONLY

#ifndef HPG_DELTA_EXPERIMENTAL_ONLY
// test that GridderState supports multiple calls to grid_visibilities()
// interspersed by calls to set_convolution_function()
TEST(GridderState, Sequences) {
  std::array<unsigned, 4> grid_size{16, 15, 1, 3};
  std::array<hpg::grid_scale_fp, 2> grid_scale{0.1, -0.1};
  size_t num_vis = 10;
  auto padding = 2 * hpg::CFArray::padding;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3 + padding, 3 + padding, 1, 3}, {2 + padding, 2 + padding, 1, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  auto gs =
    hpg::get_value(
      hpg::GridderState::create<1>(
        default_device,
        0,
        1,
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
    MyCFArray cf = create_cf(10, cf_sizes, rng);
    auto gs1 =
      hpg::get_value(
        gs.set_convolution_function(default_host_device, MyCFArray(cf)));
    init_visibilities(num_vis, grid_size, grid_scale, cf, rng, vis, ch_maps);
    auto gs2 =
      hpg::get_value(
        std::move(gs1)
        .grid_visibilities(default_host_device, decltype(vis)(vis), ch_maps));

    auto err_or_gs3 =
      std::move(gs2)
      .grid_visibilities(default_host_device, decltype(vis)(vis), ch_maps);
    ASSERT_TRUE(hpg::is_value(err_or_gs3));

    auto err_or_gs4 =
      hpg::get_value(std::move(err_or_gs3))
      .set_convolution_function(default_host_device, MyCFArray(cf));
    ASSERT_TRUE(hpg::is_value(err_or_gs4));

    auto err_or_gs5 =
      hpg::get_value(
        std::move(err_or_gs4)).grid_visibilities(
          default_host_device,
          decltype(vis)(vis),
          ch_maps);
    ASSERT_TRUE(hpg::is_value(err_or_gs5));

    auto err_or_gs6 =
      hpg::get_value(
        std::move(err_or_gs5)).grid_visibilities(
          default_host_device,
          decltype(vis)(vis),
          ch_maps);
    ASSERT_TRUE(hpg::is_value(err_or_gs6));
  }
}
#endif // HPG_DELTA_EXPERIMENTAL_ONLY

#ifndef HPG_DELTA_EXPERIMENTAL_ONLY
// test error conditions in grid_visibilities()
TEST(GridderState, GriddingError) {
  std::array<unsigned, 4> grid_size{16, 15, 2, 10};
  std::array<hpg::grid_scale_fp, 2> grid_scale{0.1, -0.1};
  size_t num_vis = 10;
  auto padding = 2 * hpg::CFArray::padding;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3 + padding, 3 + padding, 2, 3}, {2 + padding, 2 + padding, 2, 2}};
  std::mt19937 rng(42);
  MyCFArray cf = create_cf(10, cf_sizes, rng);
  unsigned max_avg_channels_per_vis = 2;
  std::vector<hpg::VisData<1>> vis;
  std::vector<std::map<unsigned, hpg::vis_weight_fp>> ch_maps;

  {
    // check for wrong number of polarizations in gridding
    auto gs =
      hpg::get_value(
        hpg::GridderState::create<2>(
          default_device,
          0,
          1,
          num_vis,
          max_avg_channels_per_vis,
          &cf,
          grid_size,
          grid_scale,
          {{0, -1}, {-1, 1}},
          {{0, -1}, {-1, 1}}));

    auto gs1 =
      hpg::get_value(
        std::move(gs)
        .set_convolution_function(default_host_device, MyCFArray(cf)));

    init_visibilities(num_vis, grid_size, grid_scale, cf, rng, vis, ch_maps);
    auto gs2_or_err =
      gs1.grid_visibilities(default_host_device, decltype(vis)(vis), ch_maps);
    ASSERT_TRUE(hpg::is_error(gs2_or_err));
    EXPECT_EQ(
      hpg::get_error(std::move(gs2_or_err))->type(),
      hpg::ErrorType::InvalidNumberPolarizations);
  }
  auto gs =
    hpg::get_value(
      hpg::GridderState::create<1>(
        default_device,
        0,
        1,
        num_vis,
        max_avg_channels_per_vis,
        &cf,
        grid_size,
        grid_scale,
        {{0}, {0}},
        {{0}, {0}}));
  auto gs1 =
    hpg::get_value(
      std::move(gs)
      .set_convolution_function(default_host_device, MyCFArray(cf)));
  {
    // check for update to grid weights without gridding
    init_visibilities(num_vis, grid_size, grid_scale, cf, rng, vis, ch_maps);
    auto gs2_or_err =
      gs1.grid_visibilities_base(
        default_host_device,
        decltype(vis)(vis),
        ch_maps,
        true, // update_grid_weights
        true, // do_degrid
        false, // return_visibilities,
        false); // do_grid
    ASSERT_TRUE(hpg::is_error(gs2_or_err));
    EXPECT_EQ(
      hpg::get_error(std::move(gs2_or_err))->type(),
      hpg::ErrorType::UpdateWeightsWithoutGridding);
  }
  {
    // check for not enough grid channel maps
    init_visibilities(num_vis, grid_size, grid_scale, cf, rng, vis, ch_maps);
    decltype(ch_maps) chm1 = ch_maps;
    chm1.pop_back();
    auto gs2_or_err =
      gs1.grid_visibilities(default_host_device, decltype(vis)(vis), chm1);
    ASSERT_TRUE(hpg::is_error(gs2_or_err));
    EXPECT_EQ(
      hpg::get_error(std::move(gs2_or_err))->type(),
      hpg::ErrorType::GridChannelMapsSize);
  }
  {
    // check for too many grid channel maps
    init_visibilities(num_vis, grid_size, grid_scale, cf, rng, vis, ch_maps);
    decltype(vis) vism1 = vis;
    vism1.pop_back();
    auto gs2_or_err =
      gs1.grid_visibilities(default_host_device, std::move(vism1), ch_maps);
    ASSERT_TRUE(hpg::is_error(gs2_or_err));
    EXPECT_EQ(
      hpg::get_error(std::move(gs2_or_err))->type(),
      hpg::ErrorType::GridChannelMapsSize);
  }
  {
    // check for too many channels in aggregate in the grid channel maps (this
    // test could be improved, as init_visibilities() could produce channel maps
    // with smaller than the requested size because channels are selected
    // randomly...here we just create large enough maps that it is unlikely that
    // the constraint won't be violated)
    init_visibilities(
      num_vis,
      grid_size,
      grid_scale,
      cf,
      rng,
      vis,
      ch_maps,
      2 * max_avg_channels_per_vis);
    auto gs2_or_err =
      gs1.grid_visibilities(default_host_device, decltype(vis)(vis), ch_maps);
    ASSERT_TRUE(hpg::is_error(gs2_or_err));
    EXPECT_EQ(
      hpg::get_error(std::move(gs2_or_err))->type(),
      hpg::ErrorType::ExcessiveVisibilityChannels);
  }
}
#endif // HPG_DELTA_EXPERIMENTAL_ONLY

#ifndef HPG_DELTA_EXPERIMENTAL_ONLY
// test that GridderState correctly serializes CF changes with gridding
TEST(GridderState, Serialization) {
  std::array<unsigned, 4> grid_size{16, 15, 1, 3};
  std::array<hpg::grid_scale_fp, 2> grid_scale{0.1, -0.1};

  auto padding = 2 * hpg::CFArray::padding;
  std::array<std::vector<std::array<unsigned, 4>>, 2>
    cf_sizes{
      std::vector<std::array<unsigned, 4>>{{3 + padding, 3 + padding, 1, 3}},
      std::vector<std::array<unsigned, 4>>{{2 + padding, 2 + padding, 1, 2}}};

  std::mt19937 rng(42);

  std::array<MyCFArray, 2> cfs{
    create_cf(10, cf_sizes[0], rng),
    create_cf(10, cf_sizes[1], rng)};

  std::array<std::vector<hpg::VisData<1>>, 2> vis;
  std::array<std::vector<std::map<unsigned, hpg::vis_weight_fp>>, 2> ch_maps;
  const size_t num_vis = 1000;
  for (size_t i = 0; i < 2; ++i)
    init_visibilities(
      num_vis,
      grid_size,
      grid_scale,
      cfs[i],
      rng,
      vis[i],
      ch_maps[i]);

  // do gridding with the two sets in both orders, and check that the results
  // are identical
  auto test =
    [&](unsigned first, unsigned second) {
      return
        hpg::RvalM<const size_t&, hpg::GridderState>::pure(
          [=](size_t i) {
            return
              hpg::GridderState::create<1>(
                default_device,
                0,
                0,
                num_vis,
                1,
                static_cast<const hpg::CFArray*>(&cfs[i]),
                grid_size,
                grid_scale,
                {{0}},
                {{0}});
          })
        .and_then(
          [=](auto&& gs) mutable {
            return
              std::move(gs)
              .set_convolution_function(
                default_host_device,
                std::move(cfs[first]));
          })
        .and_then(
          [=](auto&& gs) mutable {
            return
              std::move(gs)
              .grid_visibilities(
                default_host_device,
                std::move(vis[first]),
                ch_maps[first],
                true);
          })
        .and_then(
          [=](auto&& gs) mutable {
            return
              std::move(gs)
              .set_convolution_function(
                default_host_device,
                std::move(cfs[second]));
          })
        .and_then(
          [=](auto&& gs) mutable {
            return
              std::move(gs)
              .grid_visibilities(
                default_host_device,
                std::move(vis[second]),
                ch_maps[second],
                true);
          })
        .map(
          [](auto&& gs) {
            auto [gs1, gv] = std::move(gs).grid_values();
            auto [gs2, gw] = std::move(gs1).grid_weights();
            return std::make_tuple(std::move(gv), std::move(gw));
          });
    };
  // run this test twice in order to test serialization with both reallocation
  // and reuse of CF regions
  for (size_t i = 0; i < 2; ++i) {
    auto r01 = test(0, 1)(i);
    auto r10 = test(1, 0)(i);
    ASSERT_TRUE(hpg::is_value(r01));
    ASSERT_TRUE(hpg::is_value(r10));
    auto [v01, w01] = hpg::get_value(std::move(r01));
    auto [v10, w10] = hpg::get_value(std::move(r10));
    EXPECT_TRUE(values_eq(v01.get(), v10.get()));
    EXPECT_TRUE(values_eq(w01.get(), w10.get()));
  }
}
#endif // HPG_DELTA_EXPERIMENTAL_ONLY

#ifndef HPG_DELTA_EXPERIMENTAL_ONLY
// test that exceeding visibility batch size limit produces error
TEST(GridderState, Batching) {
  std::array<unsigned, 4> grid_size{16, 15, 1, 3};
  std::array<hpg::grid_scale_fp, 2> grid_scale{0.1, -0.1};
  auto padding = 2 * hpg::CFArray::padding;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3 + padding, 3 + padding, 1, 3}, {2 + padding, 2 + padding, 1, 2}};
  std::mt19937 rng(42);
  MyCFArray cf = create_cf(10, cf_sizes, rng);
  size_t num_vis = 100;

  auto gs =
    hpg::get_value(
      hpg::GridderState::create<1>(
        default_device,
        0,
        1,
        num_vis - 1,
        1,
        &cf,
        grid_size,
        grid_scale,
        {{0}},
        {{0}}));

  // error if number of visibilities exceeds batch size
  {
    std::vector<hpg::VisData<1>> vis;
    std::vector<std::map<unsigned, hpg::vis_weight_fp>> ch_maps;
    auto gs1 =
      hpg::get_value(
        gs.set_convolution_function(default_host_device, MyCFArray(cf)));
    init_visibilities(num_vis, grid_size, grid_scale, cf, rng, vis, ch_maps);
    auto gs2_or_err =
      gs1.grid_visibilities(default_host_device, decltype(vis)(vis), ch_maps);
    ASSERT_TRUE(hpg::is_error(gs2_or_err));
    EXPECT_EQ(
      hpg::get_error(std::move(gs2_or_err))->type(),
      hpg::ErrorType::ExcessiveNumberVisibilities);
  }
  // no error if number of visibilities does not exceed batch size
  {
    std::vector<hpg::VisData<1>> vis;
    std::vector<std::map<unsigned, hpg::vis_weight_fp>> ch_maps;
    auto gs1 =
      hpg::get_value(
        gs.set_convolution_function(default_host_device, MyCFArray(cf)));
    init_visibilities(
      num_vis - 1,
      grid_size,
      grid_scale,
      cf,
      rng,
      vis,
      ch_maps);
    auto gs2_or_err =
      gs1.grid_visibilities(default_host_device, decltype(vis)(vis), ch_maps);
    EXPECT_FALSE(hpg::is_error(gs2_or_err));
  }
}
#endif // HPG_DELTA_EXPERIMENTAL_ONLY

TEST(GridderState, Gridding) {
  const std::array<unsigned, 4> grid_size{14, 14, 1, 1};
  const std::array<hpg::grid_scale_fp, 2> grid_scale{0.1, 0.1};
  constexpr unsigned cf_radius = 3;
  constexpr unsigned cf_oversampling = 10;
  constexpr hpg::vis_frequency_fp c = 299792458.0;
  constexpr std::complex<hpg::visibility_fp> vis(1.0, -1.0);
  constexpr hpg::vis_weight_fp wgt = 1.0;

  ConeCFArray cf(1, cf_oversampling, cf_radius);

  auto test =
    [=](hpg::vis_uvw_t uvw) {
      return
        hpg::RvalM<void, hpg::GridderState>::pure(
          [=]() {
            return
              hpg::GridderState::create<1>(
                default_device,
                0,
                0,
                1,
                1,
                &cf,
                grid_size,
                grid_scale,
                {{0}},
                {{0}}
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
                , impl_versions
#endif
                );
          })
        .and_then(
          [=](auto&& gs) {
            return
              std::move(gs)
              .set_convolution_function(default_host_device, ConeCFArray(cf));
          })
        .and_then(
          [=](auto&& gs) {
            return
              std::move(gs)
              .grid_visibilities(
                default_host_device,
                std::vector<hpg::VisData<1>>{
                  hpg::VisData<1>({vis}, c, 0.0, uvw, {0, 0})},
                std::vector<std::map<unsigned, hpg::vis_weight_fp>>{{{0, wgt}}},
                true);
          })
        .map(
          [](auto&& gs) {
            auto [gs1, gv] = std::move(gs).grid_values();
            auto [gs2, gw] = std::move(gs1).grid_weights();
            return std::make_tuple(std::move(gv), std::move(gw));
          });
        };
  const std::array<hpg::grid_scale_fp, 2> fine_scale{
    grid_scale[0] * cf_oversampling,
    grid_scale[1] * cf_oversampling
  };
  for (unsigned i = 0; i <= cf_oversampling; ++i)
    for (unsigned j = 0; j <= cf_oversampling; ++j) {
      const hpg::vis_uvw_t uvw{i * fine_scale[0], j * fine_scale[1], 0.0};
      auto err_or_result = test(uvw)();
      ASSERT_TRUE(hpg::is_value(err_or_result));
      auto [v, w] = hpg::get_value(std::move(err_or_result));
      EXPECT_TRUE(
        cf.verify_gridded_vis(v.get(), grid_size, grid_scale, vis, c, uvw));
    }
}

TEST(GridderState, GridOne) {
  const std::array<unsigned, 4> grid_size{16384, 16384, 1, 1};
  const std::array<hpg::grid_scale_fp, 2> grid_scale{0.0476591, 0.0476591};
  constexpr unsigned cf_radius = 45;
  constexpr unsigned cf_oversampling = 20;
  constexpr hpg::vis_frequency_fp freq = 3.693e+09;
  constexpr std::complex<hpg::visibility_fp> vis(1.0, -1.0);
  constexpr hpg::vis_weight_fp wgt = 1.0;

  ConeCFArray cf(1, cf_oversampling, cf_radius);

  auto test =
    [=](hpg::vis_uvw_t uvw) {
      return
        hpg::RvalM<void, hpg::GridderState>::pure(
          [=]() {
            return
              hpg::GridderState::create<1>(
                default_device,
                0,
                0,
                1,
                1,
                &cf,
                grid_size,
                grid_scale,
                {{0}},
                {{0}}
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
                , impl_versions
#endif
                );
          })
        .and_then(
          [=](auto&& gs) {
            return
              std::move(gs)
              .set_convolution_function(default_host_device, ConeCFArray(cf));
          })
        .and_then(
          [=](auto&& gs) {
            return
              std::move(gs)
              .grid_visibilities(
                default_host_device,
                std::vector<hpg::VisData<1>>{
                  hpg::VisData<1>({vis}, freq, 0.0, uvw, {0, 0})},
                std::vector<std::map<unsigned, hpg::vis_weight_fp>>{{{0, wgt}}},
                true);
          })
        .map(
          [](auto&& gs) {
            auto [gs1, gv] = std::move(gs).grid_values();
            auto [gs2, gw] = std::move(gs1).grid_weights();
            return std::make_tuple(std::move(gv), std::move(gw));
          });
        };

  const hpg::vis_uvw_t uvw{2344.1, 638.066, -1826.55};
  auto [ll, ur] = cf.cf_footprint(grid_size, grid_scale, freq, uvw);
  EXPECT_EQ(ll[0], 9523);
  EXPECT_EQ(ll[1], 8522);
  EXPECT_EQ(ur[0], 9613);
  EXPECT_EQ(ur[1], 8612);
  auto err_or_result = test(uvw)();
  ASSERT_TRUE(hpg::is_value(err_or_result));
  auto [g, w] = hpg::get_value(std::move(err_or_result));
  ASSERT_TRUE(
    cf.verify_cf_footprint(0, g.get(), grid_size, grid_scale, vis, freq, uvw));
  H5::Exception::dontPrint();
  const std::filesystem::path gridone_path = exec_dir / "gridone.h5";
  size_t d = 2;
  auto dt = H5::ArrayType(H5::PredType::NATIVE_DOUBLE, 1, &d);
  std::array<size_t, 2>
    dims{size_t(ur[0] - ll[0] + 1), size_t(ur[1] - ll[1] + 1)};
  if (!write_expected_data_files) {
    bool have_gridone;
    try {
      have_gridone = H5::H5File::isHdf5(gridone_path);
    } catch (const H5::Exception&) {
      have_gridone = false;
    }
    EXPECT_TRUE(have_gridone);
    std::vector<std::complex<double>> expected(dims[0] * dims[1]);
    auto file = H5::H5File(gridone_path, H5F_ACC_RDONLY);
    auto dd = file.openDataSet("grid");
    dd.read(expected.data(), dt);
    bool eq = true;
    for (size_t x = ll[0]; eq && x < ur[0]; ++x)
      for (size_t y = ll[1]; eq && y < ur[1]; ++y) {
        eq = (*g)(x, y, 0, 0) == expected[(x - ll[0]) * dims[1] + (y - ll[1])];
        EXPECT_EQ(
          (*g)(x, y, 0, 0),
          expected[(x - ll[0]) * dims[1] + (y - ll[1])]);
      }
  } else {
    std::vector<std::complex<double>> buff(g->min_buffer_size());
    EXPECT_FALSE(
      g->copy_to(default_host_device, buff.data(), hpg::Layout::Right));
    auto file = H5::H5File(gridone_path, H5F_ACC_TRUNC);
    dt.commit(file, "complexdouble");
    auto st_ds = H5::DataSpace(dims.size(), dims.data());
    H5::DSetCreatPropList dcpl;
    dcpl.setDeflate(9);
    dcpl.setChunk(dims.size(), dims.data());
    auto dd = file.createDataSet("grid", dt, st_ds, dcpl);
    std::array<size_t, 2> sz{size_t(grid_size[0]), size_t(grid_size[1])};
    auto mem_ds = H5::DataSpace(sz.size(), sz.data());
    std::array<size_t, 2> count{1, 1};
    std::array<size_t, 2> start{size_t(ll[0]), size_t(ll[1])};
    mem_ds.selectHyperslab(
      H5S_SELECT_SET,
      count.data(),
      start.data(),
      dims.data(),
      dims.data());
    dd.write(buff.data(), dt, mem_ds);
  }
}

TEST(GridderState, GridNone) {
  const std::array<unsigned, 4> grid_size{16384, 16384, 1, 1};
  const std::array<hpg::grid_scale_fp, 2> grid_scale{0.0476591, 0.0476591};
  constexpr unsigned cf_radius = 45;
  constexpr unsigned cf_oversampling = 20;

  ConeCFArray cf(1, cf_oversampling, cf_radius);

  auto test =
    [=](hpg::vis_uvw_t uvw) {
      return
        hpg::RvalM<void, hpg::GridderState>::pure(
          [=]() {
            return
              hpg::GridderState::create<1>(
                default_device,
                0,
                0,
                1,
                1,
                &cf,
                grid_size,
                grid_scale,
                {{0}},
                {{0}}
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
                , impl_versions
#endif
                );
          })
        .and_then(
          [=](auto&& gs) {
            return
              std::move(gs)
              .set_convolution_function(default_host_device, ConeCFArray(cf));
          })
        .and_then(
          [=](auto&& gs) {
            return
              std::move(gs)
              .grid_visibilities(
                default_host_device,
                std::vector<hpg::VisData<1>>{},
                std::vector<std::map<unsigned, hpg::vis_weight_fp>>{},
                true);
          })
        .map(
          [](auto&& gs) {
            auto [gs1, gv] = std::move(gs).grid_values();
            auto [gs2, gw] = std::move(gs1).grid_weights();
            return std::make_tuple(std::move(gv), std::move(gw));
          });
        };
  auto err_or_result = test(hpg::vis_uvw_t{2344.1, 638.066, -1826.55})();
  ASSERT_TRUE(hpg::is_value(err_or_result));
  auto [g, w] = hpg::get_value(std::move(err_or_result));
  EXPECT_FALSE(has_non_zero(g.get()));
  EXPECT_FALSE(has_non_zero(w.get()));
}

TEST(GridderState, ZeroModel) {
  const std::array<unsigned, 4> grid_size{8192, 8192, 2, 1};
  const std::array<hpg::grid_scale_fp, 2> grid_scale{0.0476591, 0.0476591};
  constexpr unsigned cf_radius = 45;
  constexpr unsigned cf_oversampling = 20;
  constexpr hpg::vis_frequency_fp freq = 3.693e+09;
  constexpr std::complex<hpg::visibility_fp> vis(1.0, -1.0);
  constexpr hpg::vis_weight_fp wgt = 1.0;

  ConeCFArray cf(1, cf_oversampling, cf_radius);
  auto gs1_or_err =
    hpg::GridderState::create<2>(
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
  ASSERT_TRUE(hpg::is_value(gs1_or_err));
  auto gs2_or_err =
    hpg::get_value(std::move(gs1_or_err))
    .set_convolution_function(default_host_device, ConeCFArray(cf));
  ASSERT_TRUE(hpg::is_value(gs2_or_err));
  auto gs2 = hpg::get_value(std::move(gs2_or_err));

  // first do gridding without a model
  auto visibilities =
    std::vector<hpg::VisData<2>>{
      hpg::VisData<2>(
        {vis, -vis},
        freq,
        0.0,
        {0.0, 0.0, 0.1},
        {0, 0})};
  std::vector<std::map<unsigned, hpg::vis_weight_fp>> ch_maps{{{0, wgt}}};
  // retain current value (zero grid) of gs2
  auto gs3_or_err =
    gs2.grid_visibilities(default_host_device, visibilities, ch_maps);
  ASSERT_TRUE(hpg::is_value(gs3_or_err));
  auto gs3 = hpg::get_value(std::move(gs3_or_err));

  // now set a zero-valued model
  std::vector<hpg::GridValueArray::value_type>
    model_buffer(grid_size[0] * grid_size[1] * grid_size[2] * grid_size[3]);
  auto model =
    hpg::GridValueArray::copy_from(
      "model_in",
      default_device,
      default_host_device,
      model_buffer.data(),
      grid_size);
  auto gs4_or_err =
    std::move(gs2).set_model(default_host_device, std::move(*model));
  ASSERT_TRUE(hpg::is_value(gs4_or_err));
  // and grid the visibility
  auto gs5_or_err =
    hpg::get_value(std::move(gs4_or_err))
    .grid_visibilities(default_host_device, visibilities, ch_maps);
  ASSERT_TRUE(hpg::is_value(gs5_or_err));
  auto gs5 = hpg::get_value(std::move(gs5_or_err));

  // compare grid values
  auto gv3 = std::get<1>(gs3.grid_values());
  ASSERT_TRUE(has_non_zero(gv3.get()));
  auto gv5 = std::get<1>(gs5.grid_values());
  EXPECT_TRUE(values_eq(gv3.get(), gv5.get()));
}

#ifndef HPG_DELTA_EXPERIMENTAL_ONLY
// test of FFT functionality through grid model
TEST(GridderState, ModelFFT) {
  std::array<unsigned, 4> grid_size{64, 64, 4, 3};
  std::array<hpg::grid_scale_fp, 2> grid_scale{0.12, -0.34};
  auto padding = 2 * hpg::CFArray::padding;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3 + padding, 3 + padding, 3, 3}, {2 + padding, 2 + padding, 2, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  auto gs =
    hpg::get_value(
      hpg::GridderState::create<1>(
        default_device,
        0,
        0,
        15,
        1,
        &cf,
        grid_size,
        grid_scale,
        {{0}, {0}, {0}, {0}},
        {{0}, {0}, {0}, {0}}));

  // initial model should be identically zero
  {
    ASSERT_TRUE(!gs.is_null());
    auto [gs1, model] = std::move(gs).model_values();
    EXPECT_FALSE(has_non_zero(model.get()));
    gs = std::move(gs1);
  }

  // create a zero-value model explicitly
  {
    auto [gs1, values] = std::move(gs).grid_values();
    auto gs2_or_err =
      std::move(gs1).set_model(default_host_device, std::move(*values));
    ASSERT_TRUE(hpg::is_value(gs2_or_err));
    auto [gs3, model] = hpg::get_value(std::move(gs2_or_err)).model_values();
    EXPECT_FALSE(has_non_zero(model.get()));
    gs = std::move(gs3);
  }

  // apply FFT to (identically zero) model
  {
    auto gs1_or_err = std::move(gs).apply_model_fft();
    ASSERT_TRUE(hpg::is_value(gs1_or_err));
    auto gs1 = hpg::get_value(gs1_or_err);
    auto [gs2, model] = std::move(gs1).model_values();
    EXPECT_FALSE(has_non_zero(model.get()));
    gs = std::move(gs2);
  }

  // check FFT of non-zero model yields non-zero values
  std::unique_ptr<hpg::GridValueArray> model0;
  {
    auto [gs1, values] = std::move(gs).grid_values(); // still zero
    auto x0 = grid_size[hpg::GridValueArray::Axis::x] / 2;
    auto y0 = grid_size[hpg::GridValueArray::Axis::y] / 2;
    auto n_mrow = grid_size[hpg::GridValueArray::Axis::mrow];
    auto n_channel = grid_size[hpg::GridValueArray::Axis::channel];
    for (size_t mrow = 0; mrow < n_mrow; ++mrow)
      for (size_t channel = 0; channel < n_channel; ++channel)
        (*values)(x0, y0, mrow, channel) = (mrow + 1) * n_channel + channel;
    auto gs2_or_err =
      std::move(gs1).set_model(default_host_device, std::move(*values));
    ASSERT_TRUE(hpg::is_value(gs2_or_err));
    auto gs3_or_err = hpg::get_value(std::move(gs2_or_err)).apply_model_fft();
    ASSERT_TRUE(hpg::is_value(gs3_or_err));
    auto [gs4, model] = hpg::get_value(std::move(gs3_or_err)).model_values();
    ASSERT_TRUE(has_non_zero(model.get()));
    model0 = std::move(model);
    gs = std::move(gs4);
  }

  // do FFT again, with normalization factor and compare to previous outcome
  {
    // apply normalization factor to model0
    const hpg::grid_value_fp norm = 0.001;
    auto n_x = grid_size[hpg::GridValueArray::Axis::x];
    auto n_y = grid_size[hpg::GridValueArray::Axis::y];
    auto n_mrow = grid_size[hpg::GridValueArray::Axis::mrow];
    auto n_channel = grid_size[hpg::GridValueArray::Axis::channel];
    for (size_t x = 0; x < n_x; ++x)
      for (size_t y = 0; y < n_y; ++y)
        for (size_t mrow = 0; mrow < n_mrow; ++mrow)
          for (size_t channel = 0; channel < n_channel; ++channel)
            (*model0)(x, y, mrow, channel) /= norm;

    auto [gs1, values] = std::move(gs).grid_values(); // still zero
    auto x0 = n_x / 2;
    auto y0 = n_y / 2;
    for (size_t mrow = 0; mrow < n_mrow; ++mrow)
      for (size_t channel = 0; channel < n_channel; ++channel)
        (*values)(x0, y0, mrow, channel) = (mrow + 1) * n_channel + channel;
    auto gs2_or_err =
      std::move(gs1).set_model(default_host_device, std::move(*values));
    ASSERT_TRUE(hpg::is_value(gs2_or_err));
    auto gs3_or_err =
      hpg::get_value(std::move(gs2_or_err)).apply_model_fft(norm);
    ASSERT_TRUE(hpg::is_value(gs3_or_err));
    auto [gs4, model] = hpg::get_value(std::move(gs3_or_err)).model_values();
    EXPECT_TRUE(values_eq(model0.get(), model.get()));
    gs = std::move(gs4);
  }
}
#endif // HPG_DELTA_EXPERIMENTAL_ONLY

// test FFT result
TEST(GridderState, OneFFT) {
  const std::array<unsigned, 4> grid_size{2048, 2048, 1, 1};
  std::array<hpg::grid_scale_fp, 2> grid_scale{0.12, -0.34};
  auto padding = 2 * hpg::CFArray::padding;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3 + padding, 3 + padding, 3, 3}};
  MyCFArrayShape cf(10, cf_sizes);
  auto gs =
    hpg::get_value(
      hpg::GridderState::create<1>(
        default_device,
        0,
        0,
        15,
        1,
        &cf,
        grid_size,
        grid_scale,
        {{0}},
        {{0}}));

  AnyGridValueArray zero_model(grid_size);
  auto gs1 =
    hpg::get_value(
      std::move(gs).set_model(default_host_device, std::move(zero_model)));
  auto [gs2, model] = std::move(gs1).model_values();
  (*model)(1024, 1108, 0, 0) = 0.825;
  auto gs3 =
    hpg::get_value(
      std::move(gs2).set_model(default_host_device, std::move(*model)));
  auto gs4_or_err = std::move(gs3).apply_model_fft();
  ASSERT_TRUE(hpg::is_value(gs4_or_err));
  auto gs4 = hpg::get_value(std::move(gs4_or_err));
  auto gs5 = std::move(gs4).shift_model(hpg::ShiftDirection::FORWARD);
  auto [gs6, modelf] = std::move(gs5).model_values();
  auto val = (*modelf)(1070, 10, 0, 0);
  std::cout << val << std::endl;
  EXPECT_TRUE(val.imag() < 0);
}

// test residual visibility return functionality
TEST(GridderState, ResidualVisibilities) {
  const std::array<unsigned, 4> grid_size{20, 20, 2, 1};
  const std::array<hpg::grid_scale_fp, 2> grid_scale{0.0476591, 0.0476591};
  constexpr unsigned cf_radius = 3;
  constexpr unsigned cf_oversampling = 10;
  constexpr hpg::vis_frequency_fp freq = 3.693e+09;
  constexpr std::complex<hpg::visibility_fp> vis(1.0, -1.0);
  constexpr hpg::vis_weight_fp wgt = 1.0;

  ConeCFArray cf(1, cf_oversampling, cf_radius);
  hpg::GridderState gs;
  {
    auto gs1_or_err =
      hpg::GridderState::create<2>(
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
    ASSERT_TRUE(hpg::is_value(gs1_or_err));
    auto gs2_or_err =
      hpg::get_value(std::move(gs1_or_err))
      .set_convolution_function(default_host_device, ConeCFArray(cf));
    ASSERT_TRUE(hpg::is_value(gs2_or_err));
    gs = hpg::get_value(std::move(gs2_or_err));
  }

  auto visibilities =
    std::vector<hpg::VisData<2>>{
      hpg::VisData<2>({vis, -vis}, freq, 0.0, {0.0, 0.0, 0.1}, {0, 0})};
  std::vector<std::map<unsigned, hpg::vis_weight_fp>> ch_maps{{{0, wgt}}};
  {
    // do gridding
    auto gsfv_or_err =
      gs.degrid_grid_get_residual_visibilities(
        default_host_device,
        std::vector<hpg::VisData<2>>(visibilities),
        ch_maps);
    ASSERT_TRUE(hpg::is_value(gsfv_or_err));
    auto [gs1, fv] = hpg::get_value(std::move(gsfv_or_err));
    // one try before future has a chance to complete
    ASSERT_FALSE(fv.get());
    // ensure that future completes
    auto gs2 = std::move(gs1).fence();
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

TEST(GridderState, EmptyCF) {
  const std::array<unsigned, 4> grid_size{16384, 16384, 1, 1};
  const std::array<hpg::grid_scale_fp, 2> grid_scale{0.0476591, 0.0476591};
  constexpr unsigned cf_oversampling = 20;

  MyCFArray cf(cf_oversampling, {{31, 31, 0, 0}}, {});

  auto test =
    [=](hpg::vis_uvw_t uvw) {
      return
        hpg::RvalM<void, hpg::GridderState>::pure(
          [=]() {
            return
              hpg::GridderState::create<1>(
                default_device,
                0,
                0,
                1,
                1,
                &cf,
                grid_size,
                grid_scale,
                {{0}},
                {{0}}
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
                , impl_versions
#endif
                );
          })
        .and_then(
          [=](auto&& gs) {
            return
              std::move(gs)
              .grid_visibilities(
                default_host_device,
                std::vector<hpg::VisData<1>>{},
                std::vector<std::map<unsigned, hpg::vis_weight_fp>>{},
                true);
          });
        };
  auto err_or_result = test(hpg::vis_uvw_t{2344.1, 638.066, -1826.55})();
  EXPECT_TRUE(hpg::is_value(err_or_result));
}

#ifndef HPG_DELTA_EXPERIMENTAL_ONLY
TEST(GridderState, ShiftCycle) {
  auto padding = 2 * hpg::CFArray::padding;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3 + padding, 3 + padding, 1, 1}};
  std::mt19937 rng(42);
  MyCFArray cf = create_cf(10, cf_sizes, rng);
  size_t num_vis = 10000;
  const std::array<hpg::grid_scale_fp, 2> grid_scale{0.0476591, 0.0476591};

  auto test =
    [&](const std::array<unsigned, 4>& grid_size) {
      return
        hpg::RvalM<void, hpg::GridderState>::pure(
          [=]() {
            return
              hpg::GridderState::create<1>(
                default_device,
                0,
                0,
                num_vis,
                1,
                &cf,
                grid_size,
                grid_scale,
                {{0}},
                {{0}}
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
                , impl_versions
#endif
                ) ;
          })
        .and_then(
          [&](auto&& gs) {
            return
              std::move(gs)
              .set_convolution_function(default_host_device, MyCFArray(cf));
          })
        .and_then(
          [&](auto&& gs) {
            std::vector<hpg::VisData<1>> vis;
            std::vector<std::map<unsigned, hpg::vis_weight_fp>> ch_maps;
            init_visibilities(
              num_vis,
              grid_size,
              grid_scale,
              cf,
              rng,
              vis,
              ch_maps);
            return
              std::move(gs)
              .grid_visibilities(
                default_host_device,
                std::move(vis),
                ch_maps,
                false);
          })
        .map(
          [](auto&& gs) {
            auto [gs0, grid0] = std::move(gs).grid_values();
            auto [gs1, grid1] =
              std::move(gs0)
              .shift_grid(hpg::ShiftDirection::FORWARD)
              .shift_grid(hpg::ShiftDirection::BACKWARD)
              .grid_values();
            return std::make_tuple(std::move(grid0), std::move(grid1));
          });
    };

  for (auto& asz : std::vector<std::array<unsigned, 2>>{
      {20, 20}, {19, 19}, {18, 19}, {19, 18}}) {
    auto grids_or_err = test({asz[0], asz[1], 1, 2})();
    ASSERT_TRUE(hpg::is_value(grids_or_err));
    auto [g0, g1] = hpg::get_value(std::move(grids_or_err));
    EXPECT_TRUE(values_eq(g0.get(), g1.get()));
  }
}
#endif // HPG_DELTA_EXPERIMENTAL_ONLY

#ifdef HPG_DELTA_EXPERIMENTAL_ONLY
TEST(GridderState, CompareExp) {
  const std::array<unsigned, 4> grid_size{16384, 16384, 1, 1};
  const std::array<hpg::grid_scale_fp, 2> grid_scale{0.0476591, 0.0476591};
  constexpr unsigned cf_radius = 45;
  constexpr unsigned cf_oversampling = 20;
  constexpr hpg::vis_frequency_fp freq = 3.693e+09;
  constexpr std::complex<hpg::visibility_fp> vis(1.0, -1.0);
  constexpr hpg::vis_weight_fp wgt = 1.0;

  ConeCFArray cf(1, cf_oversampling, cf_radius);

  auto test =
    [=](const std::array<unsigned, 4>& vsns, hpg::vis_uvw_t uvw) {
      return
        hpg::RvalM<void, hpg::GridderState>::pure(
          [=]() {
            return
              hpg::GridderState::create<1>(
                default_device,
                0,
                0,
                1,
                1,
                &cf,
                grid_size,
                grid_scale,
                {{0}},
                {{0}},
                vsns);
          })
        .and_then(
          [=](auto&& gs) {
            return
              std::move(gs)
              .set_convolution_function(default_host_device, ConeCFArray(cf));
          })
        .and_then(
          [=](auto&& gs) {
            return
              std::move(gs)
              .grid_visibilities(
                default_host_device,
                std::vector<hpg::VisData<1>>{
                  hpg::VisData<1>({vis}, freq, 0.0, uvw, {0, 0})},
                std::vector<std::map<unsigned, hpg::vis_weight_fp>>{{{0, wgt}}},
                true);
          })
        .map(
          [](auto&& gs) {
            auto [gs1, gv] = std::move(gs).grid_values();
            auto [gs2, gw] = std::move(gs1).grid_weights();
            return std::make_tuple(std::move(gv), std::move(gw));
          });
        };

  const hpg::vis_uvw_t uvw{2344.1, 638.066, -1826.55};
  auto err_or_result_default = test({0, 0, 0, 0}, uvw)();
  ASSERT_TRUE(hpg::is_value(err_or_result_default));
  auto [g0, w0] = hpg::get_value(std::move(err_or_result_default));
  auto err_or_result_exp = test(impl_versions, uvw)();
  ASSERT_TRUE(hpg::is_value(err_or_result_exp));
  auto [g1, w1] = hpg::get_value(std::move(err_or_result_exp));
  EXPECT_TRUE(values_eq(g0.get(), g1.get()));
  EXPECT_TRUE(values_eq(w0.get(), w1.get()));
}
#endif // HPG_DELTA_EXPERIMENTAL_ONLY

#ifndef HPG_DELTA_EXPERIMENTAL_ONLY
TEST(GridderState, Contexts) {
  std::array<unsigned, 4> grid_size{16, 15, 1, 3};
  std::array<hpg::grid_scale_fp, 2> grid_scale{0.1, -0.1};

  auto padding = 2 * hpg::CFArray::padding;
  std::array<std::vector<std::array<unsigned, 4>>, 2>
    cf_sizes{
      std::vector<std::array<unsigned, 4>>{{3 + padding, 3 + padding, 1, 3}},
      std::vector<std::array<unsigned, 4>>{{2 + padding, 2 + padding, 1, 2}}};

  std::mt19937 rng(42);

  std::array<MyCFArray, 2> cfs{
    create_cf(10, cf_sizes[0], rng),
    create_cf(10, cf_sizes[1], rng)};

  std::array<std::vector<hpg::VisData<1>>, 2> vis;
  std::array<std::vector<std::map<unsigned, hpg::vis_weight_fp>>, 2> ch_maps;
  const size_t num_vis = 1000;
  for (size_t i = 0; i < 2; ++i)
    init_visibilities(
      num_vis,
      grid_size,
      grid_scale,
      cfs[i],
      rng,
      vis[i],
      ch_maps[i]);

  // check initialization of contexts
  {
    auto g0_or_err =
      hpg::GridderState::create<1>(
        default_device,
        1,
        0,
        num_vis,
        1,
        static_cast<const hpg::CFArray*>(&cfs[0]),
        grid_size,
        grid_scale,
        {{0}},
        {{0}});
    ASSERT_TRUE(hpg::is_value(g0_or_err));
    auto g0 = hpg::get_value(std::move(g0_or_err));
    EXPECT_EQ(g0.num_contexts(), 2);
    if (default_device == hpg::Device::Cuda)
      EXPECT_EQ(g0.num_active_tasks(), 2);
    else
      EXPECT_EQ(g0.num_active_tasks(), 1);
    auto g1_or_err =
      std::move(g0)
      .set_convolution_function(default_host_device, MyCFArray(cfs[0]), 2);
    ASSERT_TRUE(hpg::is_error(g1_or_err));
    EXPECT_EQ(
      hpg::get_error(std::move(g1_or_err))->type(),
      hpg::ErrorType::InvalidGriddingContext);
  }

  // First grid vis[0] using cfs[0], then vis[1] with cfs[1] using a single
  // context
  auto r1 =
    hpg::RvalM<void, hpg::GridderState>::pure(
      [=]() {
        return
          hpg::GridderState::create<1>(
            default_device,
            0,
            0,
            num_vis,
            1,
            static_cast<const hpg::CFArray*>(&cfs[0]),
            grid_size,
            grid_scale,
            {{0}},
            {{0}});
      })
    .and_then(
      [=](auto&& gs) mutable {
        return
          std::move(gs)
          .set_convolution_function(default_host_device, MyCFArray(cfs[0]));
      })
    .and_then(
      [=](auto&& gs) mutable {
        return
          std::move(gs)
          .grid_visibilities(
            default_host_device,
            std::vector<hpg::VisData<1>>(vis[0]),
            ch_maps[0],
            true);
      })
    .and_then(
      [=](auto&& gs) mutable {
        return
          std::move(gs)
          .set_convolution_function(default_host_device, MyCFArray(cfs[1]));
      })
    .and_then(
      [=](auto&& gs) mutable {
        return
          std::move(gs)
          .grid_visibilities(
            default_host_device,
            std::vector<hpg::VisData<1>>(vis[1]),
            ch_maps[1],
            true);
      })
    .map(
      [](auto&& gs) {
        auto [gs1, gv] = std::move(gs).grid_values();
        auto [gs2, gw] = std::move(gs1).grid_weights();
        return std::make_tuple(std::move(gv), std::move(gw));
      })();

  // grid vis[0] with cfs[0] in one context and vis[1] with cfs[1] in the other
  // context, setting both cfs first and then gridding both vis
  auto r2 =
    hpg::RvalM<void, hpg::GridderState>::pure(
      [=]() {
        return
          hpg::GridderState::create<1>(
            default_device,
            1,
            0,
            num_vis,
            1,
            static_cast<const hpg::CFArray*>(&cfs[0]),
            grid_size,
            grid_scale,
            {{0}},
            {{0}});
      })
    .and_then(
      [=](auto&& gs) mutable {
        return
          std::move(gs).set_convolution_function(
            default_host_device,
            MyCFArray(cfs[0]),
            0);
      })
    .and_then(
      [=](auto&& gs) mutable {
        return
          std::move(gs).set_convolution_function(
            default_host_device,
            MyCFArray(cfs[1]),
            1);
      })
    .and_then(
      [=](auto&& gs) mutable {
        return
          std::move(gs)
          .grid_visibilities(
            default_host_device,
            std::vector<hpg::VisData<1>>(vis[0]),
            ch_maps[0],
            true,
            0);
      })
    .and_then(
      [=](auto&& gs) mutable {
        return
          std::move(gs)
          .grid_visibilities(
            default_host_device,
            std::vector<hpg::VisData<1>>(vis[1]),
            ch_maps[1],
            true,
            1);
      })
    .map(
      [](auto&& gs) {
        auto [gs1, gv] = std::move(gs).grid_values();
        auto [gs2, gw] = std::move(gs1).grid_weights();
        return std::make_tuple(std::move(gv), std::move(gw));
      })();

  ASSERT_TRUE(hpg::is_value(r1));
  auto [gv1, gw1] = hpg::get_value(std::move(r1));
  ASSERT_TRUE(hpg::is_value(r2));
  auto [gv2, gw2] = hpg::get_value(std::move(r2));
  EXPECT_TRUE(values_eq(gv1.get(), gv2.get()));
  EXPECT_TRUE(values_eq(gw1.get(), gw2.get()));
}
#endif // !HPG_DELTA_EXPERIMENTAL_ONLY

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

  exec_dir = std::filesystem::path(argv[0]).remove_filename();
  write_expected_data_files = false;
  for (int c = 0; !write_expected_data_files && c < argc; ++c)
    write_expected_data_files = "--write-expected"s == argv[c];

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
