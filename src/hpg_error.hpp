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
#pragma once
#include "hpg_export.h"

#include <memory>
#include <string>

namespace hpg {

/** error types
 */
enum class HPG_EXPORT ErrorType {
  DisabledDevice,
  DisabledHostDevice,
  OutOfBoundsCFIndex,
  InvalidNumberMuellerIndexRows,
  InvalidNumberPolarizations,
  InvalidCFLayout,
  InvalidModelGridSize,
  ExcessiveNumberVisibilities,
  ExcessiveVisibilityChannels,
  UpdateWeightsWithoutGridding,
  CFSupportExceedsGrid,
  GridChannelMapsSize,
  Other
};

/** error class
 */
class HPG_EXPORT Error {
private:

  ErrorType m_type;

  std::string m_msg;

public:

  /** error constructor */
  Error(const std::string& msg, ErrorType err = ErrorType::Other);

  Error() {}

  /** error description */
  const std::string&
  message() const;

  /** error type */
  ErrorType
  type() const;

  /** destructor */
  virtual ~Error();
};

/** invalid number of Mueller index rows error
 *
 * Number of rows of Mueller indexes does not equal grid "mrow" axis size */
struct InvalidNumberMuellerIndexRowsError
  : public Error {

  InvalidNumberMuellerIndexRowsError();
};

/** disabled device error
 *
 * Device is not enabled in HPG configuration.
 */
struct DisabledDeviceError
  : public Error {

  DisabledDeviceError();
};

/** disabled host device error
 *
 * Host device is not enabled by HPG configuration.
 */
struct DisabledHostDeviceError
  : public Error {

  DisabledHostDeviceError();
};

/** invalid number of polarizations error
 *
 * Number of polarizations in visibility data does not equal number of columns
 * of Mueller indexes */
struct InvalidNumberPolarizationsError
  : public Error {

  InvalidNumberPolarizationsError();
};

/** excessive number of visibilities error
 *
 * Number of visibilities exceeds configured maximum batch size of
 * GridderState
 */
struct ExcessiveNumberVisibilitiesError
  : public Error {

  ExcessiveNumberVisibilitiesError();
};

/** update weights without gridding error
 *
 * Grid weights cannot be updated without doing gridding
 */
struct UpdateWeightsWithoutGriddingError
  : public Error {

  UpdateWeightsWithoutGriddingError();
};

/** excessive number of channels in mapping error
 *
 * Total number of grid channels for visibilities exceeds configured maximum
 */
struct ExcessiveVisibilityChannelsError
  : public Error {

  ExcessiveVisibilityChannelsError();
};

struct GridChannelMapsSizeError
  : public Error {

  GridChannelMapsSizeError();
};

}  // end namespace hpg

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
