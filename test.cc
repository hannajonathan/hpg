#include "hpg.hpp"

#include <iostream>

struct MyCF2
  : public hpg::CF2 {

};

int
main(int argc, char* argv[]) {

  hpg::initialize();
#ifdef HPG_ENABLE_SERIAL
  std::cout << "Serial" << std::endl;
  {
    auto st0 =
      hpg::GridderState(hpg::Device::Serial, {1000, 2000, 3}, {0.1, -0.1});
    auto st1 = st0.fence();
    auto st2 = std::move(st0).fence();
  }
  {
    auto g0 = hpg::Gridder(hpg::Device::Serial, {1000, 2000, 3}, {0.1, -0.1});
    g0.fence();
    g0.fence();
  }
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_CUDA
  std::cout << "Cuda" << std::endl;
  {
    auto st0 =
      hpg::GridderState(hpg::Device::Cuda, {1000, 2000, 3}, {0.1, -0.1});
    auto st1 = st0.fence();
    auto st2 = std::move(st0).fence();
  }
  {
    auto g0 = hpg::Gridder(hpg::Device::Cuda, {1000, 2000, 3}, {0.1, -0.1});
    g0.fence();
    g0.fence();
  }
#endif // HPG_ENABLE_CUDA
  hpg::finalize();
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
