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
#include <cstring>

namespace hpg {

/** string type in HPG API
 *
 * Workaround for a problem occurring when client code links to both HPG and
 * CASA libraries...
 */
struct HPG_EXPORT string {
  char val[120];

  string() {
    val[0] = '\0';
  }

  string(const char* s) {
    std::strncpy(val, s, sizeof(val));
    val[sizeof(val) - 1] = '\0';
  }

  bool
  operator==(const string& rhs) const {
    return std::strcmp(val, rhs.val) == 0;
  }
};

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
  UpdateWeightsWithoutGridding,
  CFSupportExceedsGrid,
  Other
};

/** error class
 */
class HPG_EXPORT Error {
private:

  ErrorType m_type;

  hpg::string m_msg;

public:

  /** error constructor */
  Error(const hpg::string& msg, ErrorType err = ErrorType::Other);

  Error() {}

  /** error description */
  const hpg::string&
  message() const;

  /** error type */
  ErrorType
  type() const;

  /** destructor */
  virtual ~Error();
};
}  // end namespace hpg

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
