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

#include <optional>
#include <string>
#include <tuple>

using namespace hpg;

std::optional<std::tuple<unsigned, std::optional<Device>>>
hpg::impl::parsed_cf_layout_version(const std::string& layout) {
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
    if (dev == core::DeviceT<Device::Serial>::name)
      return
        std::make_tuple(
          unsigned(vn.value()),
          std::optional<Device>(Device::Serial));
#endif
#ifdef HPG_ENABLE_OPENMP
    if (dev == core::DeviceT<Device::OpenMP>::name)
      return
        std::make_tuple(
          unsigned(vn.value()),
          std::optional<Device>(Device::OpenMP));
#endif
#ifdef HPG_ENABLE_CUDA
    if (dev == core::DeviceT<Device::Cuda>::name)
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
hpg::impl::construct_cf_layout_version(unsigned vn, Device device) {
  std::ostringstream oss;
  oss << vn << "-";
  switch (device) {
#ifdef HPG_ENABLE_SERIAL
  case Device::Serial:
    oss << core::DeviceT<Device::Serial>::name;
    break;
#endif
#ifdef HPG_ENABLE_OPENMP
  case Device::OpenMP:
    oss << core::DeviceT<Device::OpenMP>::name;
    break;
#endif
#ifdef HPG_ENABLE_CUDA
  case Device::Cuda:
    oss << core::DeviceT<Device::Cuda>::name;
    break;
#endif
  default:
    assert(false);
    break;
  }
  return oss.str();
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
