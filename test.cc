#include "hpg.hpp"

#include <iostream>

struct MyCF2
  : public hpg::CF2 {

};

int
main(int argc, char* argv[]) {

  hpg::initialize();
#ifdef HPG_ENABLE_SERIAL
  {
    std::cout << "Serial" << std::endl;
    auto h0 =
      hpg::Gridder::init(hpg::Device::Serial, {1000, 2000, 3}, {0.1, -0.1});
    auto h1 = hpg::Gridder::fence(h0);
    auto h2 = hpg::Gridder::fence(std::move(h0));
  }
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_CUDA
  {
    std::cout << "Cuda" << std::endl;
    auto h0 = hpg::Gridder(hpg::Device::Cuda, {1000, 2000, 3}, {0.1, -0.1});
    h0.fence();
    h0.fence();
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
