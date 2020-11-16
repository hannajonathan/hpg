#include "hpg.hpp"
#include "gtest/gtest.h"

#ifdef HPG_ENABLE_SERIAL

TEST(GridderStateSerial, ConstructorArgs) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<float, 2> grid_scale{0.12, -0.34};
  hpg::GridderState gs0;
  hpg::GridderState gs1(hpg::Device::Serial, 0, grid_size, grid_scale);

  EXPECT_TRUE(gs0.is_null());
  EXPECT_FALSE(gs1.is_null());
  EXPECT_EQ(gs1.device(), hpg::Device::Serial);
  EXPECT_EQ(gs1.grid_size(), grid_size);
  EXPECT_EQ(gs1.grid_scale(), grid_scale);
  EXPECT_EQ(gs1.max_added_tasks(), 0);
}

TEST(GridderStateSerial, Copies) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<float, 2> grid_scale{0.12, -0.34};
  hpg::GridderState gs0(hpg::Device::Serial, 0, grid_size, grid_scale);
  hpg::GridderState gs1 = gs0;
  hpg::GridderState gs2(gs0);

  EXPECT_EQ(gs1.device(), hpg::Device::Serial);
  EXPECT_EQ(gs1.grid_size(), grid_size);
  EXPECT_EQ(gs1.grid_scale(), grid_scale);

  EXPECT_EQ(gs2.device(), hpg::Device::Serial);
  EXPECT_EQ(gs2.grid_size(), grid_size);
  EXPECT_EQ(gs2.grid_scale(), grid_scale);
}

TEST(GridderStateSerial, Moves) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<float, 2> grid_scale{0.12, -0.34};
  hpg::GridderState gs0(hpg::Device::Serial, 0, grid_size, grid_scale);
  hpg::GridderState gs1 = std::move(gs0);

  EXPECT_TRUE(gs0.is_null());
  EXPECT_EQ(gs1.device(), hpg::Device::Serial);
  EXPECT_EQ(gs1.grid_size(), grid_size);
  EXPECT_EQ(gs1.grid_scale(), grid_scale);

  hpg::GridderState gs2(std::move(gs1));
  EXPECT_TRUE(gs1.is_null());
  EXPECT_EQ(gs2.device(), hpg::Device::Serial);
  EXPECT_EQ(gs2.grid_size(), grid_size);
  EXPECT_EQ(gs2.grid_scale(), grid_scale);
}

TEST(GridderStateSerial, InitValues) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<float, 2> grid_scale{0.12, -0.34};
  hpg::GridderState gs(hpg::Device::Serial, 0, grid_size, grid_scale);

  auto [gs1, values] = std::move(gs).grid_values();
  for (size_t i = 0; i < 4; ++i)
    EXPECT_EQ(values->extent(i), grid_size[i]);
  for (unsigned x = 0; x < grid_size[0]; ++x)
    for (unsigned y = 0; y < grid_size[1]; ++y)
      for (unsigned sto = 0; sto < grid_size[2]; ++sto)
        for (unsigned cube = 0; cube < grid_size[3]; ++cube)
          EXPECT_EQ(
            values->operator()(x, y, sto, cube),
            std::complex<hpg::grid_value_fp>(0));

  auto [gs2, weights] = std::move(gs1).grid_weights();
  for (size_t i = 2; i < 4; ++i)
    EXPECT_EQ(weights->extent(i - 2), grid_size[i]);
  for (unsigned sto = 0; sto < grid_size[2]; ++sto)
    for (unsigned cube = 0; cube < grid_size[3]; ++cube)
      EXPECT_EQ(weights->operator()(sto, cube), hpg::grid_value_fp(0));
}

TEST(GridderStateSerial, CopyOrMove) {
  std::array<unsigned, 4> grid_size{6, 5, 4, 3};
  std::array<float, 2> grid_scale{0.12, -0.34};
  hpg::GridderState gs(hpg::Device::Serial, 0, grid_size, grid_scale);

  // TODO: set_convolution_function
  // TODO: grid_visibilities
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
    auto fft_rc = gs.apply_fft();
    EXPECT_FALSE(gs.is_null());
    ASSERT_TRUE(std::holds_alternative<hpg::GridderState>(fft_rc));
    hpg::GridderState& gs1 = std::get<hpg::GridderState>(fft_rc);
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

#endif // HPG_ENABLE_SERIAL

int
main(int argc, char **argv) {
  hpg::ScopeGuard hpg;
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
