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
#include "hpg_error.hpp"

using namespace hpg;

Error::Error(const std::string& msg, ErrorType err)
  : m_type(err)
  , m_msg(msg) {}

const std::string&
Error::message() const {
  return m_msg;
}

ErrorType
Error::type() const {
  return m_type;
}

Error::~Error() {}

InvalidNumberMuellerIndexRowsError::InvalidNumberMuellerIndexRowsError()
  : Error(
    "Number of rows of Mueller indexes does not match grid",
    ErrorType::InvalidNumberMuellerIndexRows) {}

DisabledDeviceError::DisabledDeviceError()
  : Error("Requested device is not enabled", ErrorType::DisabledDevice) {}

DisabledHostDeviceError::DisabledHostDeviceError()
  : Error(
    "Requested host device is not enabled",
    ErrorType::DisabledHostDevice) {}

InvalidNumberPolarizationsError::InvalidNumberPolarizationsError()
  : Error(
    "Number of visibility polarizations does not match Mueller matrix",
    ErrorType::InvalidNumberPolarizations) {}

ExcessiveNumberVisibilitiesError::ExcessiveNumberVisibilitiesError()
  : Error(
    "Number of visibilities exceeds maximum batch size",
    ErrorType::ExcessiveNumberVisibilities) {}

UpdateWeightsWithoutGriddingError::UpdateWeightsWithoutGriddingError()
  : Error(
    "Unable to update grid weights during degridding only",
    ErrorType::UpdateWeightsWithoutGridding) {}

ExcessiveVisibilityChannelsError::ExcessiveVisibilityChannelsError()
  : Error(
    "Total number of grid channel indexes for visibilities exceeds maximum",
    ErrorType::ExcessiveVisibilityChannels) {}

GridChannelMapsSizeError::GridChannelMapsSizeError()
  : Error(
    "Size of grid channel maps vector does not equal "
    "size of visibilites vector",
    ErrorType::GridChannelMapsSize) {}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
