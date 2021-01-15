#include "hpg.hpp"
#include "gtest/gtest.h"

#include <array>
#include <cassert>
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
  : private MyCFArrayShape, public hpg::CFArray {

  std::vector<std::vector<std::complex<hpg::cf_fp>>> m_values;

  MyCFArray() {}

  MyCFArray(
    unsigned oversampling,
    const std::vector<std::array<unsigned, 4>>& sizes,
    const std::vector<std::vector<std::complex<hpg::cf_fp>>>& values)
    : MyCFArrayShape(oversampling, sizes)
    , m_values(values) {
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
  cf_phase_screens.resize(num_vis);

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
  std::array<float, 2> grid_scale{0.12, -0.34};
  size_t batch_size = 21;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3, 3, 4, 3}, {2, 2, 4, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  hpg::Gridder g0;
  auto g1 =
    std::get<1>(
      hpg::Gridder::create(
        default_device,
        0,
        batch_size,
        &cf,
        grid_size,
        grid_scale));

  EXPECT_TRUE(g0.is_null());
  EXPECT_FALSE(g1.is_null());
  EXPECT_EQ(g1.device(), default_device);
  EXPECT_EQ(g1.grid_size(), grid_size);
  EXPECT_EQ(g1.grid_scale(), grid_scale);
  EXPECT_EQ(g1.max_added_tasks(), 0);
  EXPECT_EQ(g1.max_visibility_batch_size(), batch_size);
  EXPECT_EQ(
    g1.convolution_function_region_size(nullptr),
    g1.convolution_function_region_size(&cf));
}

// test that Gridder copies have correct parameters
TEST(Gridder, Copies) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<float, 2> grid_scale{0.12, -0.34};
  size_t batch_size = 31;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3, 3, 4, 3}, {2, 2, 4, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  auto g0 =
    std::get<1>(
      hpg::Gridder::create(
        default_device,
        0,
        batch_size,
        &cf,
        grid_size,
        grid_scale));
  hpg::Gridder g1 = g0;

  EXPECT_FALSE(g0.is_null());
  EXPECT_EQ(g1.device(), default_device);
  EXPECT_EQ(g1.grid_size(), grid_size);
  EXPECT_EQ(g1.grid_scale(), grid_scale);
  EXPECT_EQ(g1.max_visibility_batch_size(), batch_size);
  EXPECT_EQ(
    g1.convolution_function_region_size(nullptr),
    g0.convolution_function_region_size(nullptr));

  hpg::Gridder g2(g0);
  EXPECT_FALSE(g0.is_null());
  EXPECT_EQ(g2.device(), default_device);
  EXPECT_EQ(g2.grid_size(), grid_size);
  EXPECT_EQ(g2.grid_scale(), grid_scale);
  EXPECT_EQ(g2.max_visibility_batch_size(), batch_size);
  EXPECT_EQ(
    g2.convolution_function_region_size(nullptr),
    g0.convolution_function_region_size(nullptr));
}

// test that Gridder moves have expected outcomes
TEST(Gridder, Moves) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<float, 2> grid_scale{0.12, -0.34};
  size_t batch_size = 11;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3, 3, 4, 3}, {2, 2, 4, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  auto g0 =
    std::get<1>(
      hpg::Gridder::create(
        default_device,
        0,
        batch_size,
        &cf,
        grid_size,
        grid_scale));
  auto cf_region_sz = g0.convolution_function_region_size(nullptr);
  hpg::Gridder g1 = std::move(g0);

  EXPECT_TRUE(g0.is_null());
  EXPECT_EQ(g1.device(), default_device);
  EXPECT_EQ(g1.grid_size(), grid_size);
  EXPECT_EQ(g1.grid_scale(), grid_scale);
  EXPECT_EQ(g1.max_visibility_batch_size(), batch_size);
  EXPECT_EQ(
    g1.convolution_function_region_size(nullptr),
    cf_region_sz);

  hpg::Gridder g2(std::move(g1));
  EXPECT_TRUE(g1.is_null());
  EXPECT_EQ(g2.device(), default_device);
  EXPECT_EQ(g2.grid_size(), grid_size);
  EXPECT_EQ(g2.grid_scale(), grid_scale);
  EXPECT_EQ(g2.max_visibility_batch_size(), batch_size);
  EXPECT_EQ(
    g2.convolution_function_region_size(nullptr),
    cf_region_sz);
}

// test that Gridder grid values and weights are properly initialized
TEST(Gridder, InitValues) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<float, 2> grid_scale{0.12, -0.34};
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3, 3, 4, 3}, {2, 2, 4, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  auto g =
    std::get<1>(
      hpg::Gridder::create(default_device, 0, 10, &cf, grid_size, grid_scale));

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
  std::array<float, 2> grid_scale{0.1, -0.1};
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{{3, 3, 4, 3}, {2, 2, 4, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  auto g =
    std::get<1>(
      hpg::Gridder::create(default_device, 0, 22, &cf, grid_size, grid_scale));

  std::mt19937 rng(42);

  {
    const std::vector<std::array<unsigned, 4>>
      cf_sizes{{3, 3, 4, 3}, {2, 2, 4, 2}};
    MyCFArray cf = create_cf(10, cf_sizes, rng);
    auto oerr = g.set_convolution_function(default_host_device, MyCFArray(cf));
    EXPECT_FALSE(bool(oerr));
    // do it again
    auto oerr1 = g.set_convolution_function(default_host_device, std::move(cf));
    EXPECT_FALSE(bool(oerr1));
  }
  {
    // incorrect copolarization dimension size
    const std::array<unsigned, 4> cf_size{3, 3, 1, 3};
    auto oerr =
      hpg::Gridder(g).set_convolution_function(
        default_host_device,
        create_cf(10, {cf_size}, rng));
    EXPECT_TRUE(bool(oerr));
  }
  {
    // X dimension too large
    const std::array<unsigned, 4> cf_size{8, 3, 4, 3};
    auto oerr =
      hpg::Gridder(g).set_convolution_function(
        default_host_device,
        create_cf(10, {cf_size}, rng));
    EXPECT_TRUE(bool(oerr));
  }
  {
    // Y dimension too large
    const std::array<unsigned, 4> cf_size{3, 8, 4, 3};
    auto oerr =
      hpg::Gridder(g).set_convolution_function(
        default_host_device,
        create_cf(10, {cf_size}, rng));
    EXPECT_TRUE(bool(oerr));
  }
  {
    // error in one of a list of CFs
    const std::vector<std::array<unsigned, 4>>
      cf_sizes{{3, 3, 4, 3}, {3, 3, 1, 3}, {2, 2, 4, 2}};
    auto oerr =
      hpg::Gridder(g).set_convolution_function(
        default_host_device,
        create_cf(10, cf_sizes, rng));
    EXPECT_TRUE(bool(oerr));
  }
}

// test that Gridder::reset_grid() correctly resets grid weights and values
TEST(Gridder, Reset) {
  std::array<unsigned, 4> grid_size{16, 15, 4, 3};
  std::array<float, 2> grid_scale{0.1, -0.1};
  size_t num_vis = 10;
  auto padding = 2 * hpg::CFArray::padding;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{
      {3 + padding, 3 + padding, 4, 3},
      {2 + padding, 2 + padding, 4, 2}};
  MyCFArrayShape cf(10, cf_sizes);
  auto g =
    std::get<1>(
      hpg::Gridder::create(
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
  std::vector<hpg::cf_phase_screen_t> cf_phase_screens;

  {
    const std::array<unsigned, 4> cf_size{3 + padding, 3 + padding, 4, 3};
    MyCFArray cf = create_cf(10, {cf_size}, rng);
    g.set_convolution_function(default_host_device, MyCFArray(cf));
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
    g.grid_visibilities(
      default_host_device,
      std::move(vis),
      std::move(grid_cubes),
      std::move(cf_indexes),
      std::move(weights),
      std::move(frequencies),
      std::move(phases),
      std::move(coordinates),
      std::move(cf_phase_screens));

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
