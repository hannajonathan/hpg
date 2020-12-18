#include "hpg.hpp"
#include "gtest/gtest.h"

#include <array>
#include <cassert>
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
  const std::array<float, 2>& grid_scale,
  const MyCFArray& cf,
  Generator& gen,
  std::vector<std::complex<hpg::visibility_fp>>& vis,
  std::vector<unsigned>& grid_cubes,
  std::vector<hpg::vis_cf_index_t>& cf_indexes,
  std::vector<hpg::vis_weight_fp>& weights,
  std::vector<hpg::vis_frequency_fp>& frequencies,
  std::vector<hpg::vis_phase_fp>& phases,
  std::vector<hpg::vis_uvw_t>& coordinates) {

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

  const double inv_lambda = 9.75719;
  const double freq = 299792458.0 * inv_lambda;
  std::uniform_int_distribution<unsigned> dist_gcube(0, grid_size[3] - 1);
  std::uniform_int_distribution<unsigned> dist_gcopol(0, grid_size[2] - 1);
  std::uniform_real_distribution<hpg::visibility_fp> dist_vis(-1.0, 1.0);
  std::uniform_real_distribution<hpg::vis_weight_fp> dist_weight(0.0, 1.0);
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

template <typename T>
bool
near(const T& x, const T& y) {
  return std::abs(x - y) <= 1.0e-6 * std::abs(x);
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

// test that GridderState  constructor arguments are properly set
TEST(GridderState, ConstructorArgs) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<float, 2> grid_scale{0.12, -0.34};
  size_t batch_size = 20;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3, 3, 4, 3}, {2, 2, 4, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  hpg::GridderState gs0;
  auto gs1 =
    hpg::get_value(
      hpg::GridderState::create(
        default_device,
        0,
        batch_size,
        &cf,
        grid_size,
        grid_scale));

  EXPECT_TRUE(gs0.is_null());
  EXPECT_FALSE(gs1.is_null());
  EXPECT_EQ(gs1.device(), default_device);
  EXPECT_EQ(gs1.grid_size(), grid_size);
  EXPECT_EQ(gs1.grid_scale(), grid_scale);
  EXPECT_EQ(gs1.max_added_tasks(), 0);
  EXPECT_EQ(gs1.max_visibility_batch_size(), batch_size);
  EXPECT_EQ(
    gs1.convolution_function_region_size(nullptr),
    gs1.convolution_function_region_size(&cf));
}

// test that GridderState copies have correct parameters
TEST(GridderState, Copies) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<float, 2> grid_scale{0.12, -0.34};
  size_t batch_size = 20;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3, 3, 4, 3}, {2, 2, 4, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  auto gs0 =
    hpg::get_value(
      hpg::GridderState::create(
        default_device,
        0,
        batch_size,
        &cf,
        grid_size,
        grid_scale));
  hpg::GridderState gs1 = gs0;

  EXPECT_FALSE(gs0.is_null());
  EXPECT_EQ(gs1.device(), default_device);
  EXPECT_EQ(gs1.grid_size(), grid_size);
  EXPECT_EQ(gs1.grid_scale(), grid_scale);
  EXPECT_EQ(gs1.max_visibility_batch_size(), batch_size);
  EXPECT_EQ(
    gs0.convolution_function_region_size(nullptr),
    gs1.convolution_function_region_size(nullptr));

  hpg::GridderState gs2(gs0);
  EXPECT_FALSE(gs0.is_null());
  EXPECT_EQ(gs2.device(), default_device);
  EXPECT_EQ(gs2.grid_size(), grid_size);
  EXPECT_EQ(gs2.grid_scale(), grid_scale);
  EXPECT_EQ(gs2.max_visibility_batch_size(), batch_size);
  EXPECT_EQ(
    gs0.convolution_function_region_size(nullptr),
    gs2.convolution_function_region_size(nullptr));
}

// test that GridderState moves have expected outcomes
TEST(GridderState, Moves) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<float, 2> grid_scale{0.12, -0.34};
  size_t batch_size = 30;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3, 3, 4, 3}, {2, 2, 4, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  auto gs0 =
    hpg::get_value(
      hpg::GridderState::create(
        default_device,
        0,
        batch_size,
        &cf,
        grid_size,
        grid_scale));
  auto cf_region_size = gs0.convolution_function_region_size(nullptr);
  hpg::GridderState gs1 = std::move(gs0);

  EXPECT_TRUE(gs0.is_null());
  EXPECT_EQ(gs1.device(), default_device);
  EXPECT_EQ(gs1.grid_size(), grid_size);
  EXPECT_EQ(gs1.grid_scale(), grid_scale);
  EXPECT_EQ(gs1.max_visibility_batch_size(), batch_size);
  EXPECT_EQ(gs1.convolution_function_region_size(nullptr), cf_region_size);

  hpg::GridderState gs2(std::move(gs1));
  EXPECT_TRUE(gs1.is_null());
  EXPECT_EQ(gs2.device(), default_device);
  EXPECT_EQ(gs2.grid_size(), grid_size);
  EXPECT_EQ(gs2.grid_scale(), grid_scale);
  EXPECT_EQ(gs2.max_visibility_batch_size(), batch_size);
  EXPECT_EQ(gs2.convolution_function_region_size(nullptr), cf_region_size);
}

// test that GridderState grid values and weights are properly initialized
TEST(GridderState, InitValues) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<float, 2> grid_scale{0.12, -0.34};
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3, 3, 4, 3}, {2, 2, 4, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  auto gs =
    hpg::get_value(
      hpg::GridderState::create(
        default_device,
        0,
        15,
        &cf,
        grid_size,
        grid_scale));

  auto [gs1, values] = std::move(gs).grid_values();
  for (size_t i = 0; i < 4; ++i)
    EXPECT_EQ(values->extent(i), grid_size[i]);
  EXPECT_FALSE(has_non_zero(values.get()));

  auto [gs2, weights] = std::move(gs1).grid_weights();
  for (size_t i = 2; i < 4; ++i)
    EXPECT_EQ(weights->extent(i - 2), grid_size[i]);
  EXPECT_FALSE(has_non_zero(weights.get()));
}

// test that GridderState methods have correct copy and move semantics
TEST(GridderState, CopyOrMove) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<float, 2> grid_scale{0.1, -0.1};
  size_t num_vis = 10;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3, 3, 4, 3}, {2, 2, 4, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  auto gs =
    hpg::get_value(
      hpg::GridderState::create(
        default_device,
        0,
        num_vis,
        &cf,
        grid_size,
        grid_scale));

  std::mt19937 rng(42);

  std::vector<std::complex<hpg::visibility_fp>> vis;
  std::vector<unsigned> grid_cubes;
  std::vector<hpg::vis_cf_index_t> cf_indexes;
  std::vector<hpg::vis_weight_fp> weights;
  std::vector<hpg::vis_frequency_fp> frequencies;
  std::vector<hpg::vis_phase_fp> phases;
  std::vector<hpg::vis_uvw_t> coordinates;

  {
    const std::vector<std::array<unsigned, 4>>
      cf_sizes{{3, 3, 4, 3}, {2, 2, 4, 2}};
    MyCFArray cf = create_cf(10, cf_sizes, rng);
    auto gs1 =
      hpg::get_value(
        gs.set_convolution_function(default_host_device, MyCFArray(cf)));
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
      coordinates);
    auto gs2 =
      hpg::get_value(
        gs1.grid_visibilities(
          default_host_device,
          decltype(vis)(vis),
          decltype(grid_cubes)(grid_cubes),
          decltype(cf_indexes)(cf_indexes),
          decltype(weights)(weights),
          decltype(frequencies)(frequencies),
          decltype(phases)(phases),
          decltype(coordinates)(coordinates)));

    // gridded visibilities should be in gs2, not gs1
    auto [gs3, values] = std::move(gs1).grid_values();
    for (size_t i = 0; i < 4; ++i)
      EXPECT_EQ(values->extent(i), grid_size[i]);
    EXPECT_FALSE(has_non_zero(values.get()));

    auto gs4 =
      hpg::get_value(
        std::move(gs3).grid_visibilities(
          default_host_device,
          std::move(vis),
          std::move(grid_cubes),
          std::move(cf_indexes),
          std::move(weights),
          std::move(frequencies),
          std::move(phases),
          std::move(coordinates)));

    // gs2 and gs4 should have same grid values
    auto [gs5, values5] = std::move(gs2).grid_values();
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
    auto rc_fft = gs.apply_fft();
    EXPECT_FALSE(gs.is_null());
    ASSERT_TRUE(hpg::is_value(rc_fft));
    hpg::GridderState gs1 = hpg::get_value(std::move(rc_fft));
    auto err_or_gs2 = std::move(gs1).apply_fft();
    EXPECT_TRUE(gs1.is_null());

    ASSERT_TRUE(hpg::is_value(err_or_gs2));
    EXPECT_FALSE(hpg::get_value(err_or_gs2).is_null());
  }
  {
    auto gs1 = gs.shift_grid();
    EXPECT_FALSE(gs.is_null());
    auto gs2 = std::move(gs1).shift_grid();
    EXPECT_TRUE(gs1.is_null());
    EXPECT_FALSE(gs2.is_null());
  }
}

// test that GridderState::set_convolution_function() returns errors for
// erroneous CFArray arguments
TEST(GridderState, CFError) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<float, 2> grid_scale{0.1, -0.1};
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3, 3, 4, 3}, {2, 2, 4, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  auto gs =
    hpg::get_value(
      hpg::GridderState::create(
        default_device,
        0,
        10,
        &cf,
        grid_size,
        grid_scale));

  std::mt19937 rng(42);

  {
    // incorrect copolarization dimension size
    const std::array<unsigned, 4> cf_size{3, 3, 1, 3};
    auto error_or_gs =
      gs.set_convolution_function(
        default_host_device,
        create_cf(10, {cf_size}, rng));
    EXPECT_TRUE(hpg::is_error(error_or_gs));
  }
  {
    // X dimension too large
    const std::array<unsigned, 4> cf_size{8, 3, 4, 3};
    auto error_or_gs =
      gs.set_convolution_function(
        default_host_device,
        create_cf(10, {cf_size}, rng));
    EXPECT_TRUE(hpg::is_error(error_or_gs));
  }
  {
    // Y dimension too large
    const std::array<unsigned, 4> cf_size{3, 8, 4, 3};
    auto error_or_gs =
      gs.set_convolution_function(
        default_host_device,
        create_cf(10, {cf_size}, rng));
    EXPECT_TRUE(hpg::is_error(error_or_gs));
  }
  {
    // error in one of a list of CFs
    const std::vector<std::array<unsigned, 4>>
      cf_sizes{{3, 3, 4, 3}, {3, 3, 1, 3}, {2, 2, 4, 2}};
    auto error_or_gs =
      gs.set_convolution_function(
        default_host_device,
        create_cf(10, cf_sizes, rng));
    EXPECT_TRUE(hpg::is_error(error_or_gs));
  }
}

// test that GridderState::reset_grid() correctly resets grid weights and values
// for both copy and move method varieties
TEST(GridderState, Reset) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<float, 2> grid_scale{0.1, -0.1};
  size_t num_vis = 10;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3, 3, 4, 3}, {2, 2, 4, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  auto gs =
    hpg::get_value(
      hpg::GridderState::create(
        default_device,
        0,
        num_vis,
        &cf,
        grid_size,
        grid_scale));

  std::mt19937 rng(42);

  std::vector<std::complex<hpg::visibility_fp>> vis;
  std::vector<unsigned> grid_cubes;
  std::vector<hpg::vis_cf_index_t> cf_indexes;
  std::vector<hpg::vis_weight_fp> weights;
  std::vector<hpg::vis_frequency_fp> frequencies;
  std::vector<hpg::vis_phase_fp> phases;
  std::vector<hpg::vis_uvw_t> coordinates;

  {
    const std::array<unsigned, 4> cf_size{3, 3, 4, 3};
    MyCFArray cf = create_cf(10, {cf_size}, rng);
    auto gs1 =
      hpg::get_value(
        gs.set_convolution_function(default_host_device, MyCFArray(cf)));
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
      coordinates);
    auto gs2 =
      hpg::get_value(
        std::move(gs1).grid_visibilities(
          default_host_device,
          std::move(vis),
          std::move(grid_cubes),
          std::move(cf_indexes),
          std::move(weights),
          std::move(frequencies),
          std::move(phases),
          std::move(coordinates)));

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

// test that GridderState supports multiple calls to grid_visibilities()
// interspersed by calls to set_convolution_function()
TEST(GridderState, Sequences) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<float, 2> grid_scale{0.1, -0.1};
  size_t num_vis = 10;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3, 3, 4, 3}, {2, 2, 4, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  auto gs =
    hpg::get_value(
      hpg::GridderState::create(
        default_device,
        1,
        num_vis,
        &cf,
        grid_size,
        grid_scale));

  std::mt19937 rng(42);

  std::vector<std::complex<hpg::visibility_fp>> vis;
  std::vector<unsigned> grid_cubes;
  std::vector<hpg::vis_cf_index_t> cf_indexes;
  std::vector<hpg::vis_weight_fp> weights;
  std::vector<hpg::vis_frequency_fp> frequencies;
  std::vector<hpg::vis_phase_fp> phases;
  std::vector<hpg::vis_uvw_t> coordinates;

  {
    const std::vector<std::array<unsigned, 4>>
      cf_sizes{{3, 3, 4, 3}, {2, 2, 4, 2}};
    MyCFArray cf = create_cf(10, cf_sizes, rng);
    auto gs1 =
      hpg::get_value(
        gs.set_convolution_function(default_host_device, MyCFArray(cf)));
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
      coordinates);
    auto gs2 =
      hpg::get_value(
        std::move(gs1).grid_visibilities(
          default_host_device,
          decltype(vis)(vis),
          decltype(grid_cubes)(grid_cubes),
          decltype(cf_indexes)(cf_indexes),
          decltype(weights)(weights),
          decltype(frequencies)(frequencies),
          decltype(phases)(phases),
          decltype(coordinates)(coordinates)));

    auto err_or_gs3 =
      std::move(gs2).grid_visibilities(
        default_host_device,
        decltype(vis)(vis),
        decltype(grid_cubes)(grid_cubes),
        decltype(cf_indexes)(cf_indexes),
        decltype(weights)(weights),
        decltype(frequencies)(frequencies),
        decltype(phases)(phases),
        decltype(coordinates)(coordinates));
    ASSERT_TRUE(hpg::is_value(err_or_gs3));

    auto err_or_gs4 =
      hpg::get_value(std::move(err_or_gs3))
      .set_convolution_function(default_host_device, MyCFArray(cf));
    ASSERT_TRUE(hpg::is_value(err_or_gs4));

    auto err_or_gs5 =
      hpg::get_value(std::move(err_or_gs4)).grid_visibilities(
        default_host_device,
        decltype(vis)(vis),
        decltype(grid_cubes)(grid_cubes),
        decltype(cf_indexes)(cf_indexes),
        decltype(weights)(weights),
        decltype(frequencies)(frequencies),
        decltype(phases)(phases),
        decltype(coordinates)(coordinates));
    ASSERT_TRUE(hpg::is_value(err_or_gs5));

    auto err_or_gs6 =
      hpg::get_value(std::move(err_or_gs5)).grid_visibilities(
        default_host_device,
        decltype(vis)(vis),
        decltype(grid_cubes)(grid_cubes),
        decltype(cf_indexes)(cf_indexes),
        decltype(weights)(weights),
        decltype(frequencies)(frequencies),
        decltype(phases)(phases),
        decltype(coordinates)(coordinates));
    ASSERT_TRUE(hpg::is_value(err_or_gs6));
  }
}

// test that GridderState correctly serializes CF changes with gridding
TEST(GridderState, Serialization) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<float, 2> grid_scale{0.1, -0.1};

  std::array<std::vector<std::array<unsigned, 4>>, 2>
    cf_sizes{std::vector<std::array<unsigned, 4>>{{3, 3, 4, 3}},
             std::vector<std::array<unsigned, 4>>{{2, 2, 4, 2}}};

  std::mt19937 rng(42);

  std::array<MyCFArray, 2> cfs{
    create_cf(10, cf_sizes[0], rng),
    create_cf(10, cf_sizes[1], rng)};

  std::array<std::vector<std::complex<hpg::visibility_fp>>, 2> vis;
  std::array<std::vector<unsigned>, 2> grid_cubes;
  std::array<std::vector<hpg::vis_cf_index_t>, 2> cf_indexes;
  std::array<std::vector<hpg::vis_weight_fp>, 2> weights;
  std::array<std::vector<hpg::vis_frequency_fp>, 2> frequencies;
  std::array<std::vector<hpg::vis_phase_fp>, 2> phases;
  std::array<std::vector<hpg::vis_uvw_t>, 2> coordinates;

  const size_t num_vis = 1000;

  for (size_t i = 0; i < 2; ++i)
    init_visibilities(
      num_vis,
      grid_size,
      grid_scale,
      cfs[i],
      rng,
      vis[i],
      grid_cubes[i],
      cf_indexes[i],
      weights[i],
      frequencies[i],
      phases[i],
      coordinates[i]);

  // do gridding with the two sets in both orders, and check that the results
  // are identical
  auto test =
    [&](unsigned first, unsigned second) {
      return
        hpg::RvalM<const size_t&, hpg::GridderState>::pure(
          [=](size_t i) {
            return
              hpg::GridderState::create(
                default_device,
                0,
                num_vis,
                static_cast<const hpg::CFArray*>(&cfs[i]),
                grid_size,
                grid_scale);
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
                std::move(grid_cubes[first]),
                std::move(cf_indexes[first]),
                std::move(weights[first]),
                std::move(frequencies[first]),
                std::move(phases[first]),
                std::move(coordinates[first]));
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
                std::move(grid_cubes[second]),
                std::move(cf_indexes[second]),
                std::move(weights[second]),
                std::move(frequencies[second]),
                std::move(phases[second]),
                std::move(coordinates[second]));
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

// test that visibility batching for oversize requests to grid_visibilities()
// works
TEST(GridderState, Batching) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<float, 2> grid_scale{0.1, -0.1};
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3, 3, 4, 3}, {2, 2, 4, 2}};
  std::mt19937 rng(42);
  MyCFArray cf = create_cf(10, cf_sizes, rng);
  size_t num_vis = 100;
  auto gs_small =
    hpg::get_value(
      hpg::flatmap(
        hpg::GridderState::create(
          default_device,
          1,
          num_vis / 11,
          static_cast<const hpg::CFArray*>(&cf),
          grid_size,
          grid_scale)
        , [&](auto&& g) {
            return
              std::move(g)
              .set_convolution_function(default_host_device, MyCFArray(cf));
          }));
  auto gs_large =
    hpg::get_value(
      hpg::flatmap(
        hpg::GridderState::create(
          default_device,
          1,
          num_vis,
          static_cast<const hpg::CFArray*>(&cf),
          grid_size,
          grid_scale)
        , [&](auto&& g) {
            return
              std::move(g)
              .set_convolution_function(default_host_device, std::move(cf));
          }));

  std::vector<std::complex<hpg::visibility_fp>> vis;
  std::vector<unsigned> grid_cubes;
  std::vector<hpg::vis_cf_index_t> cf_indexes;
  std::vector<hpg::vis_weight_fp> weights;
  std::vector<hpg::vis_frequency_fp> frequencies;
  std::vector<hpg::vis_phase_fp> phases;
  std::vector<hpg::vis_uvw_t> coordinates;

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
    coordinates);

  auto gv_small =
    std::get<1>(
      hpg::get_value(
        hpg::map(
          std::move(gs_small).grid_visibilities(
            default_host_device,
            decltype(vis)(vis),
            decltype(grid_cubes)(grid_cubes),
            decltype(cf_indexes)(cf_indexes),
            decltype(weights)(weights),
            decltype(frequencies)(frequencies),
            decltype(phases)(phases),
            decltype(coordinates)(coordinates))
          , [](auto&& g) {
              return std::move(g).grid_values();
            })));

  auto gv_large =
    std::get<1>(
      hpg::get_value(
        hpg::map(
          std::move(gs_large).grid_visibilities(
            default_host_device,
            decltype(vis)(vis),
            decltype(grid_cubes)(grid_cubes),
            decltype(cf_indexes)(cf_indexes),
            decltype(weights)(weights),
            decltype(frequencies)(frequencies),
            decltype(phases)(phases),
            decltype(coordinates)(coordinates))
          , [](auto&& g) {
              return std::move(g).grid_values();
            })));
  EXPECT_TRUE(values_eq(gv_small.get(), gv_large.get()));
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
