#define HPG_INTERNAL
#include "hpg.hpp"
#include "gtest/gtest.h"

#include <array>
#include <chrono>
#include <complex>
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
#endif

using namespace std::complex_literals;

struct ConeCFArray final
  : public hpg::CFArray {

  unsigned m_nmueller;
  int m_oversampling;
  std::vector<unsigned> m_radius;
  std::vector<unsigned> m_oversampled_radius;

  ConeCFArray() {}

  ConeCFArray(
    unsigned nmueller,
    unsigned oversampling,
    std::vector<unsigned> radius)
    : m_nmueller(nmueller)
    , m_oversampling(oversampling)
    , m_radius(radius) {

    for (auto& r : radius)
      m_oversampled_radius.push_back((r /*+ padding*/) * m_oversampling);
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
    return m_radius.size();
  }

  std::array<unsigned, 4>
  extents(unsigned grp) const override {
    unsigned w = 2 * m_oversampled_radius[grp] + 1;
    return {w, w, m_nmueller, 1};
  }

  std::complex<float>
  operator()(unsigned x, unsigned y, unsigned mueller, unsigned, unsigned grp)
    const override {

    std::complex<float> p(
      (-m_oversampled_radius[grp] + int(x)) + 0.5f,
      (-m_oversampled_radius[grp] + int(y)) + 0.5f);
    return
      std::polar(
        (mueller + 1) * std::max(m_oversampled_radius[grp] - std::abs(p), 0.0f),
        std::arg(std::abs(p.real()) + 1.0if * std::abs(p.imag())));
  }
};

struct LargeCFArray final
  : public hpg::CFArray {

  static const unsigned m_oversampling = 20;
  static constexpr const std::array<unsigned,32> m_sizes{
    17, 17, 17, 17, 18, 19, 20, 21,
    22, 23, 25, 26, 28, 31, 36, 40,
    50, 50, 50, 50, 50, 50, 50, 50,
    50, 50, 50, 50, 50, 50, 50, 50};
  std::vector<std::array<unsigned, 4>> m_extents;
  std::vector<std::vector<std::complex<hpg::cf_fp>>> m_values;

  LargeCFArray() {}

  LargeCFArray(unsigned n_mueller) {

    for (auto& sz : m_sizes) {
      m_extents.push_back(
        {(2 * (sz + padding) + 1) * m_oversampling,
         (2 * (sz + padding) + 1) * m_oversampling,
         n_mueller,
         1});
      size_t len =
        m_extents.back()[0] * m_extents.back()[1]
        * m_extents.back()[2] * m_extents.back()[3];
      m_values.emplace_back(len);
    }
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
  operator()(
    unsigned x,
    unsigned y,
    unsigned mueller,
    unsigned cube,
    unsigned grp)
    const override {
    auto& vals = m_values[grp];
    auto& ext = m_extents[grp];
    return vals[((x * ext[1] + y) * ext[2] + mueller) * ext[3] + cube];
  }
};

template <typename Generator>
void
init_visibilities(
  unsigned num_vis,
  const std::array<unsigned, 4>& grid_size,
  const std::array<float, 2>& grid_scale,
  const hpg::CFArray* cf,
  Generator& gen,
  std::vector<hpg::VisData<1>>& vis) {

  vis.clear();
  vis.reserve(num_vis);

  const double inv_lambda = 9.75719;
  const double freq = 299792458.0 * inv_lambda;
  std::uniform_int_distribution<unsigned> dist_gcube(0, grid_size[3] - 1);
  std::uniform_int_distribution<unsigned> dist_gcopol(0, grid_size[2] - 1);
  std::uniform_real_distribution<hpg::visibility_fp> dist_vis(-1.0, 1.0);
  std::uniform_real_distribution<hpg::vis_weight_fp> dist_weight(0.0, 1.0);
  std::uniform_real_distribution<hpg::cf_phase_gradient_fp>
    dist_cfgrad(-3.141, 3.141);
  std::uniform_int_distribution<unsigned> dist_cfgrp(0, cf->num_groups() - 1);
  auto x0 = (cf->oversampling() * (grid_size[0] - 2)) / 2;
  auto y0 = (cf->oversampling() * (grid_size[1] - 2)) / 2;
  double uscale = grid_scale[0] * cf->oversampling() * inv_lambda;
  double vscale = grid_scale[1] * cf->oversampling() * inv_lambda;
  for (auto i = 0; i < num_vis; ++i) {
    auto grp = dist_cfgrp(gen);
    auto cfextents = cf->extents(grp);
    std::uniform_int_distribution<unsigned> dist_cfcube(0, cfextents[3] - 1);
    double ulim = (x0 - (cfextents[0]) / 2) / uscale;
    double vlim = (y0 - (cfextents[1]) / 2) / vscale;
    std::uniform_real_distribution<hpg::vis_uvw_fp> dist_u(-ulim, ulim);
    std::uniform_real_distribution<hpg::vis_uvw_fp> dist_v(-vlim, vlim);
    vis.push_back(
      hpg::VisData<1>(
        {std::complex<hpg::visibility_fp>(dist_vis(gen), dist_vis(gen))},
        {dist_weight(gen)},
        freq,
        0.0,
        hpg::vis_uvw_t({dist_u(gen), dist_v(gen), 0.0}),
        dist_gcube(gen),
        {dist_cfcube(gen), grp},
        {dist_cfgrad(gen), dist_cfgrad(gen)}));
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

// test array returned from hpg::CFArray::copy_to()
TEST(DeviceCFArray, Create) {
  const unsigned oversampling = 20;
  ConeCFArray cf(2, oversampling, {10, 20, 30});
  std::vector<std::vector<ConeCFArray::value_type>> arrays;
  std::optional<std::string> vsn;
  for (unsigned grp = 0; grp < cf.num_groups(); ++grp) {
    auto sz_or_err = cf.min_buffer_size(default_device, grp);
    ASSERT_TRUE(hpg::is_value(sz_or_err));
    arrays.emplace_back(hpg::get_value(sz_or_err));
    auto vsn_or_err =
      cf.copy_to(
        default_device,
        default_host_device,
        grp,
        arrays.back().data());
    ASSERT_TRUE(hpg::is_value(vsn_or_err));
    if (vsn)
      EXPECT_EQ(vsn, hpg::get_value(vsn_or_err));
    else
      vsn = hpg::get_value(vsn_or_err);
  }
  std::vector<
    std::tuple<std::array<unsigned, 4>, std::vector<ConeCFArray::value_type>>>
    sized_arrays;
  for (unsigned grp = 0; grp < arrays.size(); ++grp)
    sized_arrays.emplace_back(cf.extents(grp), std::move(arrays[grp]));
  auto devcf_or_err =
    hpg::DeviceCFArray::create(
      vsn.value(),
      oversampling,
      std::move(sized_arrays));
  ASSERT_TRUE(hpg::is_value(devcf_or_err));
  auto devcf = hpg::get_value(std::move(devcf_or_err));
  EXPECT_EQ(devcf->device(), default_device);
  EXPECT_EQ(devcf->num_groups(), cf.num_groups());
  EXPECT_EQ(devcf->oversampling(), cf.oversampling());
  for (unsigned grp = 0; grp < arrays.size(); ++grp)
    ASSERT_TRUE(devcf->extents(grp) == cf.extents(grp));
  for (unsigned grp = 0; grp < arrays.size(); ++grp) {
    auto extents = cf.extents(grp);
    auto radius = cf.m_oversampled_radius[grp];
    for (unsigned m = 0; m < extents[2]; ++m)
      for (unsigned y = 0; y < radius; ++y)
        for (unsigned x = 0; x < radius; ++x)
          EXPECT_EQ((*devcf)(x, y, m, 0, grp), cf(x, y, m, 0, grp));
  }
}

// tests of layout versioning
TEST(DeviceCFArray, LayoutVersion) {
  const unsigned oversampling = 20;
  ConeCFArray cf(1, oversampling, {10});
  std::vector<ConeCFArray::value_type>
    array(hpg::get_value(cf.min_buffer_size(default_device, 0)));
  auto vsn_or_err =
    cf.copy_to(default_device, default_host_device, 0, array.data());
  ASSERT_TRUE(hpg::is_value(vsn_or_err));
  std::string vsn = hpg::get_value(vsn_or_err);

  std::vector<
    std::tuple<std::array<unsigned, 4>, std::vector<ConeCFArray::value_type>>>
    sized_arrays;
  sized_arrays.emplace_back(cf.extents(0), std::move(array));
  auto devcf_or_err =
    hpg::DeviceCFArray::create(
      hpg::cf_layout_unspecified_version,
      oversampling,
      std::move(sized_arrays));
  ASSERT_TRUE(hpg::is_error(devcf_or_err));
  auto err = hpg::get_error(devcf_or_err);
  EXPECT_EQ(err.type(), hpg::ErrorType::InvalidCFLayout);
  // It would be nice to have more tests of the layout version functionality,
  // but without having access to the string format, and being limited to the
  // enabled devices makes this difficult.
}

// tests of gridding using a DeviceCFArray
TEST(DeviceCFArray, Gridding) {
  // CF definition
  const unsigned oversampling = 20;
  ConeCFArray cf(1, oversampling, {10}); // TODO: more Mueller indexes

  // create DeviceCFArray
  std::vector<ConeCFArray::value_type>
    array(hpg::get_value(cf.min_buffer_size(default_device, 0)));
  auto vsn_or_err =
    cf.copy_to(default_device, default_host_device, 0, array.data());
  ASSERT_TRUE(hpg::is_value(vsn_or_err));
  std::string vsn = hpg::get_value(vsn_or_err);
  std::vector<
    std::tuple<std::array<unsigned, 4>, std::vector<ConeCFArray::value_type>>>
    sized_arrays;
  sized_arrays.emplace_back(cf.extents(0), std::move(array));
  auto devcf =
    hpg::get_value(
      hpg::DeviceCFArray::create(vsn, oversampling, std::move(sized_arrays)));

  // grid definition
  std::array<unsigned, 4> grid_size{50, 50, 1, 1};
  std::array<float, 2> grid_scale{0.1, -0.1};

  // visibilities
  unsigned num_vis = 1000;
  std::mt19937 rng(42);
  std::vector<hpg::VisData<1>> vis;

  init_visibilities(num_vis, grid_size, grid_scale, &cf, rng, vis);

  // cf GridderState
  auto gs_cf =
    hpg::get_value(
      hpg::flatmap(
        hpg::GridderState::create<1>(
          default_device,
          0,
          num_vis,
          &cf,
          grid_size,
          grid_scale,
          {{0}},
          {{0}}),
        [&cf](auto&& gs) {
          return
            std::move(gs)
            .set_convolution_function(default_host_device, std::move(cf));
        }));

  // devcf GridderState
  auto gs_devcf =
    hpg::get_value(
      hpg::flatmap(
        hpg::GridderState::create<1>(
          default_device,
          0,
          num_vis,
          devcf.get(),
          grid_size,
          grid_scale,
          {{0}},
          {{0}}),
        [&devcf](auto&& gs) {
          return
            std::move(gs)
            .set_convolution_function(
              default_host_device,
              std::move(*devcf));
        }));

  // function to grid visibilities and return gridded values
  auto gridding =
    hpg::RvalM<const hpg::GridderState&, hpg::GridderState>::pure(
      [&](const hpg::GridderState& gs) {
        return
          gs.grid_visibilities_only(default_host_device, decltype(vis)(vis));
      })
    .map(
      [](auto&& gs) {
        return std::get<1>(std::move(gs).grid_values());
      });

  auto grid_cf = hpg::get_value(gridding(gs_cf));
  ASSERT_TRUE(has_non_zero(grid_cf.get()));
  auto grid_devcf_or_err = gridding(gs_devcf);
  ASSERT_TRUE(hpg::is_value(grid_devcf_or_err));
  auto grid_devcf = hpg::get_value(std::move(grid_devcf_or_err));
  EXPECT_TRUE(values_eq(grid_cf.get(), grid_devcf.get()));
}

TEST(DeviceCFArray, Efficiency) {
  // create two versions of the following CFArray: one, the original; and two,
  // the DeviceCFArray equivalent
  LargeCFArray cf(2);
  // create DeviceCFArrays of cf
  std::vector<
    std::tuple<std::array<unsigned, 4>, std::vector<hpg::CFArray::value_type>>>
    sized_arrays;
  std::string vsn;
  for (unsigned grp = 0; grp < cf.num_groups(); ++grp) {
    // allocate storage for cf values in group "grp"
    std::vector<hpg::CFArray::value_type>
      array(hpg::get_value(cf.min_buffer_size(default_device, grp)));
    // copy cf values to array
    auto vsn_or_err =
      cf.copy_to(default_device, default_host_device, grp, array.data());
    ASSERT_TRUE(hpg::is_value(vsn_or_err));
    vsn = hpg::get_value(vsn_or_err); // this should be the same for all groups
    // save value array for this group along with the group's extents
    sized_arrays.emplace_back(cf.extents(grp), std::move(array));
  }

  // define test as a function of CFArray, to do timing of
  // set_convolution_function() with given CFArray
  auto time_set_cf =
    [](std::vector<std::unique_ptr<hpg::CFArray>>& cfs) {
      return
        hpg::RvalM<void, hpg::GridderState>::pure(
          [&]() {
            // create the GridderState instance
            return
              hpg::GridderState::create<1>(
                default_device,
                0,
                10,
                cfs[0].get(),
                {1000, 1000, 1, 1},
                {0.1, 0.1},
                {{0}},
                {{0}});
          })
        .map(
          [](auto&& gs) {
            // start a timer
            auto result = std::move(gs).fence();
            return // (start-time, GridderState) tuple
              std::make_tuple(
                std::chrono::steady_clock::now(),
                std::move(result));
          })
        .and_then_loop(
          static_cast<unsigned>(cfs.size()),
          [&](unsigned i, auto&& t0_gs) {
            return
              hpg::map(
                // set CF
                std::get<1>(std::move(t0_gs)).set_convolution_function(
                  default_host_device,
                  std::move(*cfs[i])),
                // insert start time into returned tuple
                [&](auto&& gs) {
                  return
                    std::make_tuple( // (start-time, GridderState) tuple
                      std::get<0>(std::move(t0_gs)),
                      std::move(gs));
                });
          })
        .map(
          [](auto&& t0_gs) {
            // fence to complete CF transfer
            std::get<1>(std::move(t0_gs)).fence();
            // compute elapsed time
            std::chrono::duration<double> elapsed =
              std::chrono::steady_clock::now() - std::get<0>(std::move(t0_gs));
            return elapsed.count();
          });
    };

  // run test with a vector of each version of cf, but allocate the vectors one
  // type at a time to conserve memory
  const unsigned num_copies = 10;
  double t_cf;
  {
    std::vector<std::unique_ptr<hpg::CFArray>> cfs;
    for (unsigned i = 0; i < num_copies; ++i) {
      cfs.emplace_back(new LargeCFArray(cf));
    }
    auto tcf_or_err = time_set_cf(cfs)();
    ASSERT_TRUE(hpg::is_value(tcf_or_err));
    t_cf = hpg::get_value(tcf_or_err);
  }
  double t_devcf;
  {
    std::vector<std::unique_ptr<hpg::CFArray>> devcfs;
    for (unsigned i = 0; i < num_copies; ++i) {
      auto devcf_or_err =
        hpg::DeviceCFArray::create(
          vsn,
          cf.oversampling(),
          decltype(sized_arrays)(sized_arrays));
      ASSERT_TRUE(hpg::is_value(devcf_or_err));
      devcfs.push_back(hpg::get_value(std::move(devcf_or_err)));
    }
    auto tdevcf_or_err = time_set_cf(devcfs)();
    ASSERT_TRUE(hpg::is_value(tdevcf_or_err));
    t_devcf = hpg::get_value(tdevcf_or_err);
  }
  std::cout << "cf " << t_cf << "; devcf " << t_devcf << std::endl;
  EXPECT_LT(t_devcf, t_cf);
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
