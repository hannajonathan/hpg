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
#include "hpg_impl.hpp"

#include <cfenv>
#include <optional>
#include <string>
#include <tuple>

using namespace hpg;

static bool hpg_initialized = false;
static bool hpg_cleanup_fftw = false;

bool
runtime::impl::is_initialized() noexcept {
  return hpg_initialized;
}

bool
runtime::impl::initialize(const InitArguments& args) {
  bool result = true;
  K::InitArguments kargs;
  kargs.num_threads = args.num_threads;
  kargs.num_numa = args.num_numa;
  kargs.device_id = args.device_id;
  kargs.ndevices = args.ndevices;
  kargs.skip_device = ((args.skip_device >= 0) ? args.skip_device : 9999);
  kargs.disable_warnings = args.disable_warnings;
  K::initialize(kargs);
#ifdef HPG_ENABLE_OPENMP
  auto rc = fftw_init_threads();
  result = rc != 0;
#endif
#if defined(HPG_ENABLE_CUDA)                                    \
  && (defined(HPG_ENABLE_OPENMP) || defined(HPG_ENABLE_SERIAL))
  if (std::fegetround() != FE_TONEAREST)
    std::cerr << "hpg::initialize() WARNING:"
              << " Host rounding mode not set to FE_TONEAREST " << std::endl
              << "  To avoid potential inconsistency in gridding on "
              << "  host vs gridding on device,"
              << "  set rounding mode to FE_TONEAREST" << std::endl;
#endif
  hpg_initialized = result;
  hpg_cleanup_fftw = args.cleanup_fftw;
  return result;
}

void
runtime::impl::finalize() {
  K::finalize();
  if (hpg_cleanup_fftw) {
#ifdef HPG_ENABLE_SERIAL
    fftw_cleanup();
#endif
#ifdef HPG_ENABLE_OPENMP
    fftw_cleanup_threads();
#endif
  }
}

std::optional<std::tuple<unsigned, std::optional<Device>>>
runtime::impl::parsed_cf_layout_version(const std::string& layout) {
  auto dash = layout.find('-');
  std::optional<int> vn;
  if (dash != std::string::npos) {
    try {
      vn = std::stoi(layout.substr(0, dash));
      if (vn.value() < 0)
        vn.reset();
    } catch (...) {}
  }
  if (vn) {
    std::string dev = layout.substr(dash + 1);
#ifdef HPG_ENABLE_SERIAL
    if (dev == DeviceT<Device::Serial>::name)
      return
        std::make_tuple(
          unsigned(vn.value()),
          std::optional<Device>(Device::Serial));
#endif
#ifdef HPG_ENABLE_OPENMP
    if (dev == DeviceT<Device::OpenMP>::name)
      return
        std::make_tuple(
          unsigned(vn.value()),
          std::optional<Device>(Device::OpenMP));
#endif
#ifdef HPG_ENABLE_CUDA
    if (dev == DeviceT<Device::Cuda>::name)
      return
        std::make_tuple(
          unsigned(vn.value()),
          std::optional<Device>(Device::Cuda));
#endif
    return std::make_tuple(unsigned(vn.value()), std::nullopt);
  }
  return std::nullopt;
}

std::string
runtime::impl::construct_cf_layout_version(unsigned vn, Device device) {
  std::ostringstream oss;
  oss << vn << "-";
  switch (device) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial:
    oss << DeviceT<Device::Serial>::name;
    break;
#endif
#ifdef HPG_ENABLE_OPENMP
  case Device::OpenMP:
    oss << DeviceT<Device::OpenMP>::name;
    break;
#endif
#ifdef HPG_ENABLE_CUDA
  case Device::Cuda:
    oss << DeviceT<Device::Cuda>::name;
    break;
#endif
  default:
    assert(false);
    break;
  }
  return oss.str();
}

rval_t<size_t>
runtime::impl::min_cf_buffer_size(
  Device device,
  const CFArray& cf,
  unsigned grp) {

  if (devices().count(device) == 0)
    return rval<size_t>(DisabledDeviceError());

  size_t alloc_sz;

  switch (device) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial: {
    using kokkos_device = DeviceT<Device::Serial>::kokkos_device;
    auto layout = CFLayout<kokkos_device>::dimensions(&cf, grp);
    alloc_sz =
      core::cf_view<typename kokkos_device::array_layout, K::HostSpace>
      ::required_allocation_size(
        layout.dimension[0],
        layout.dimension[1],
        layout.dimension[2],
        layout.dimension[3],
        layout.dimension[4],
        layout.dimension[5]);
    break;
  }
#endif
#ifdef HPG_ENABLE_OPENMP
  case Device::OpenMP: {
    using kokkos_device = DeviceT<Device::OpenMP>::kokkos_device;
    auto layout = CFLayout<kokkos_device>::dimensions(&cf, grp);
    alloc_sz =
      core::cf_view<typename kokkos_device::array_layout, K::HostSpace>
      ::required_allocation_size(
        layout.dimension[0],
        layout.dimension[1],
        layout.dimension[2],
        layout.dimension[3],
        layout.dimension[4],
        layout.dimension[5]);
    break;
  }
#endif
#ifdef HPG_ENABLE_CUDA
  case Device::Cuda: {
    using kokkos_device = DeviceT<Device::Cuda>::kokkos_device;
    auto layout = CFLayout<kokkos_device>::dimensions(&cf, grp);
    alloc_sz =
      core::cf_view<typename kokkos_device::array_layout, K::HostSpace>
      ::required_allocation_size(
        layout.dimension[0],
        layout.dimension[1],
        layout.dimension[2],
        layout.dimension[3],
        layout.dimension[4],
        layout.dimension[5]);
    break;
  }
#endif
  default:
    assert(false);
    break;
  }
  return
    rval<size_t>((alloc_sz + (sizeof(core::cf_t) - 1)) / sizeof(core::cf_t));
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
