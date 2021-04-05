// Copyright 2021 Associated Universities, Inc. Washington DC, USA.
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

namespace hpg {

  /** HPG version string */
  HPG_EXPORT const char* version();

  /** HPG major version number */
  HPG_EXPORT unsigned version_major();

  /** HPG minor version number */
  HPG_EXPORT unsigned version_minor();

  /** HPG patch version number */
  HPG_EXPORT unsigned version_patch();

  /** HPG tweak version number
   *
   * @todo always 0? remove this
   */
  HPG_EXPORT unsigned version_tweak();

}  // end namespace hpg

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
