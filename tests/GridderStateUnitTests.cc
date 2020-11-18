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

struct MyCFArray final
  : public hpg::CFArray {

  unsigned m_oversampling;
  std::array<unsigned, 4> m_extent;
  std::vector<std::complex<hpg::cf_fp>> m_values;

  MyCFArray() {}

  MyCFArray(
    unsigned oversampling,
    const std::array<unsigned, 4>& size,
    const std::vector<std::complex<hpg::cf_fp>>& values)
    : m_oversampling(oversampling)
    , m_values(values) {

    m_extent[0] = size[0] * oversampling;
    m_extent[1] = size[1] * oversampling;
    m_extent[2] = size[2];
    m_extent[3] = size[3];
  }

  unsigned
  oversampling() const override {
    return m_oversampling;
  }

  unsigned
  extent(unsigned dim) const override {
    return m_extent[dim];
  }

  std::complex<hpg::cf_fp>
  operator()(unsigned x, unsigned y, unsigned stokes, unsigned cube)
    const override {
    return
      m_values[
        ((x * m_extent[1] + y) * m_extent[2] + stokes) * m_extent[3]
        + cube];
  }
};

template <typename Generator>
MyCFArray
create_cf(
  unsigned oversampling,
  const std::array<unsigned, 4>& size,
  Generator& gen) {

  const unsigned num_values =
    oversampling * size[0] * oversampling * size[1] * size[2] * size[3];
  std::vector<std::complex<hpg::cf_fp>> values;
  values.reserve(num_values);
  std::uniform_real_distribution<hpg::cf_fp> dist(-1.0, 1.0);
  for (auto i = 0; i < num_values; ++i)
    values.emplace_back(dist(gen), dist(gen));
  return MyCFArray(oversampling, size, values);
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
  std::vector<unsigned>& cf_cubes,
  std::vector<hpg::vis_weight_fp>& weights,
  std::vector<hpg::vis_frequency_fp>& frequencies,
  std::vector<hpg::vis_phase_fp>& phases,
  std::vector<hpg::vis_uvw_t>& coordinates) {

  vis.clear();
  vis.reserve(num_vis);
  grid_cubes.clear();
  grid_cubes.reserve(num_vis);
  cf_cubes.clear();
  cf_cubes.reserve(num_vis);
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
  std::uniform_int_distribution<unsigned> dist_gsto(0, grid_size[2] - 1);
  std::uniform_int_distribution<unsigned> dist_cfcube(0, cf.extent(3) - 1);
  std::uniform_real_distribution<hpg::visibility_fp> dist_vis(-1.0, 1.0);
  std::uniform_real_distribution<hpg::vis_weight_fp> dist_weight(0.0, 1.0);
  double ulim =
    ((cf.oversampling() * (grid_size[0] - 2)) / 2 - (cf.extent(0)) / 2)
    / (grid_scale[0] * cf.oversampling() * inv_lambda);
  double vlim =
    ((cf.oversampling() * (grid_size[1] - 2)) / 2 - (cf.extent(1)) / 2)
    / (grid_scale[1] * cf.oversampling() * inv_lambda);
  std::uniform_real_distribution<hpg::vis_uvw_fp> dist_u(-ulim, ulim);
  std::uniform_real_distribution<hpg::vis_uvw_fp> dist_v(-vlim, vlim);
  for (auto i = 0; i < num_vis; ++i) {
    vis.emplace_back(dist_vis(gen), dist_vis(gen));
    grid_cubes.push_back(dist_gcube(gen));
    cf_cubes.push_back(dist_cfcube(gen));
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
        if (array->operator()(i, j) != typename T::scalar_type(0))
          return true;
  } else {
    for (unsigned i = 0; i < array->extent(0); ++i)
      for (unsigned j = 0; j < array->extent(1); ++j)
        for (unsigned k = 0; k < array->extent(2); ++k)
          for (unsigned m = 0; m < array->extent(3); ++m)
            if (array->operator()(i, j, k, m) != typename T::scalar_type(0))
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
        if (array0->operator()(i, j) != array1->operator()(i, j))
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
            if (array0->operator()(i, j, k, m)
                != array1->operator()(i, j, k, m))
              return false;
  }
  return true;
}

// test that GridderState  constructor arguments are properly set
TEST(GridderState, ConstructorArgs) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<float, 2> grid_scale{0.12, -0.34};
  hpg::GridderState gs0;
  hpg::GridderState gs1(default_device, 0, grid_size, grid_scale);

  EXPECT_TRUE(gs0.is_null());
  EXPECT_FALSE(gs1.is_null());
  EXPECT_EQ(gs1.device(), default_device);
  EXPECT_EQ(gs1.grid_size(), grid_size);
  EXPECT_EQ(gs1.grid_scale(), grid_scale);
  EXPECT_EQ(gs1.max_added_tasks(), 0);
}

// test that GridderState copies have correct parameters
TEST(GridderState, Copies) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<float, 2> grid_scale{0.12, -0.34};
  hpg::GridderState gs0(default_device, 0, grid_size, grid_scale);
  hpg::GridderState gs1 = gs0;

  EXPECT_FALSE(gs0.is_null());
  EXPECT_EQ(gs1.device(), default_device);
  EXPECT_EQ(gs1.grid_size(), grid_size);
  EXPECT_EQ(gs1.grid_scale(), grid_scale);

  hpg::GridderState gs2(gs0);
  EXPECT_FALSE(gs0.is_null());
  EXPECT_EQ(gs2.device(), default_device);
  EXPECT_EQ(gs2.grid_size(), grid_size);
  EXPECT_EQ(gs2.grid_scale(), grid_scale);
}

// test that GridderState moves have expected outcomes
TEST(GridderState, Moves) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<float, 2> grid_scale{0.12, -0.34};
  hpg::GridderState gs0(default_device, 0, grid_size, grid_scale);
  hpg::GridderState gs1 = std::move(gs0);

  EXPECT_TRUE(gs0.is_null());
  EXPECT_EQ(gs1.device(), default_device);
  EXPECT_EQ(gs1.grid_size(), grid_size);
  EXPECT_EQ(gs1.grid_scale(), grid_scale);

  hpg::GridderState gs2(std::move(gs1));
  EXPECT_TRUE(gs1.is_null());
  EXPECT_EQ(gs2.device(), default_device);
  EXPECT_EQ(gs2.grid_size(), grid_size);
  EXPECT_EQ(gs2.grid_scale(), grid_scale);
}

// test that GridderState grid values and weights are properly initialized
TEST(GridderState, InitValues) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<float, 2> grid_scale{0.12, -0.34};
  hpg::GridderState gs(default_device, 0, grid_size, grid_scale);

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
  hpg::GridderState gs(default_device, 0, grid_size, grid_scale);

  std::mt19937 rng(42);

  std::vector<std::complex<hpg::visibility_fp>> vis;
  std::vector<unsigned> grid_cubes;
  std::vector<unsigned> cf_cubes;
  std::vector<hpg::vis_weight_fp> weights;
  std::vector<hpg::vis_frequency_fp> frequencies;
  std::vector<hpg::vis_phase_fp> phases;
  std::vector<hpg::vis_uvw_t> coordinates;

  {
    const std::array<unsigned, 4> cf_size{3, 3, 4, 3};
    MyCFArray cf = create_cf(10, cf_size, rng);
    auto gs1 =
      std::get<1>(gs.set_convolution_function(default_host_device, cf));
    init_visibilities(
      10,
      grid_size,
      grid_scale,
      cf,
      rng,
      vis,
      grid_cubes,
      cf_cubes,
      weights,
      frequencies,
      phases,
      coordinates);
    auto gs2 =
      gs1.grid_visibilities(
        default_host_device,
        vis,
        grid_cubes,
        cf_cubes,
        weights,
        frequencies,
        phases,
        coordinates);

    // gridded visibilities should be in gs2, not gs1
    auto [gs3, values] = std::move(gs1).grid_values();
    for (size_t i = 0; i < 4; ++i)
      EXPECT_EQ(values->extent(i), grid_size[i]);
    EXPECT_FALSE(has_non_zero(values.get()));

    auto gs4 =
      std::move(gs3).grid_visibilities(
        default_host_device,
        vis,
        grid_cubes,
        cf_cubes,
        weights,
        frequencies,
        phases,
        coordinates);

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
#if HPG_API >= 17
    ASSERT_TRUE(std::holds_alternative<hpg::GridderState>(rc_fft));
#else
    ASSERT_FALSE(std::get<0>(rc_fft));
#endif
    hpg::GridderState& gs1 = std::get<1>(rc_fft);
    auto [opterr, gs2] = std::move(gs1).apply_fft();
    EXPECT_TRUE(gs1.is_null());
    EXPECT_FALSE(gs2.is_null());
  }
  {
    auto gs1 = gs.rotate_grid();
    EXPECT_FALSE(gs.is_null());
    auto gs2 = std::move(gs1).rotate_grid();
    EXPECT_TRUE(gs1.is_null());
    EXPECT_FALSE(gs2.is_null());
  }
}

// test that GridderState::set_convolution_function() returns errors for
// erroneous CFArray arguments
TEST(GridderState, CFError) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<float, 2> grid_scale{0.1, -0.1};
  hpg::GridderState gs(default_device, 0, grid_size, grid_scale);

  std::mt19937 rng(42);

  std::vector<std::complex<hpg::visibility_fp>> vis;
  std::vector<unsigned> grid_cubes;
  std::vector<unsigned> cf_cubes;
  std::vector<hpg::vis_weight_fp> weights;
  std::vector<hpg::vis_frequency_fp> frequencies;
  std::vector<hpg::vis_phase_fp> phases;
  std::vector<hpg::vis_uvw_t> coordinates;

  {
    // incorrect Stokes dimension size
    const std::array<unsigned, 4> cf_size{3, 3, 1, 3};
    MyCFArray cf = create_cf(10, cf_size, rng);
    auto error_or_gs = gs.set_convolution_function(default_host_device, cf);
#if HPG_API >= 17
    EXPECT_TRUE(std::holds_alternative<hpg::Error>(error_or_gs));
#else
    EXPECT_TRUE(std::get<0>(error_or_gs));
#endif
    auto [error, gs1] =
      hpg::GridderState(gs).set_convolution_function(default_host_device, cf);
    EXPECT_TRUE(error);
  }
  {
    // X dimension too large
    const std::array<unsigned, 4> cf_size{8, 3, 4, 3};
    MyCFArray cf = create_cf(10, cf_size, rng);
    auto error_or_gs = gs.set_convolution_function(default_host_device, cf);
#if HPG_API >= 17
    EXPECT_TRUE(std::holds_alternative<hpg::Error>(error_or_gs));
#else
    EXPECT_TRUE(std::get<0>(error_or_gs));
#endif
    auto [error, gs1] =
      hpg::GridderState(gs).set_convolution_function(default_host_device, cf);
    EXPECT_TRUE(error);
  }
  {
    // Y dimension too large
    const std::array<unsigned, 4> cf_size{3, 8, 4, 3};
    MyCFArray cf = create_cf(10, cf_size, rng);
    auto error_or_gs = gs.set_convolution_function(default_host_device, cf);
#if HPG_API >= 17
    EXPECT_TRUE(std::holds_alternative<hpg::Error>(error_or_gs));
#else
    EXPECT_TRUE(std::get<0>(error_or_gs));
#endif
    auto [error, gs1] =
      hpg::GridderState(gs).set_convolution_function(default_host_device, cf);
    EXPECT_TRUE(error);
  }
}

// test that GridderState::reset_grid() correctly resets grid weights and values
// for both copy and move method varieties
TEST(GridderState, Reset) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<float, 2> grid_scale{0.1, -0.1};
  hpg::GridderState gs(default_device, 0, grid_size, grid_scale);

  std::mt19937 rng(42);

  std::vector<std::complex<hpg::visibility_fp>> vis;
  std::vector<unsigned> grid_cubes;
  std::vector<unsigned> cf_cubes;
  std::vector<hpg::vis_weight_fp> weights;
  std::vector<hpg::vis_frequency_fp> frequencies;
  std::vector<hpg::vis_phase_fp> phases;
  std::vector<hpg::vis_uvw_t> coordinates;

  {
    const std::array<unsigned, 4> cf_size{3, 3, 4, 3};
    MyCFArray cf = create_cf(10, cf_size, rng);
    auto gs1 =
      std::get<1>(gs.set_convolution_function(default_host_device, cf));
    init_visibilities(
      10,
      grid_size,
      grid_scale,
      cf,
      rng,
      vis,
      grid_cubes,
      cf_cubes,
      weights,
      frequencies,
      phases,
      coordinates);
    auto gs2 =
      std::move(gs1).grid_visibilities(
        default_host_device,
        vis,
        grid_cubes,
        cf_cubes,
        weights,
        frequencies,
        phases,
        coordinates);

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