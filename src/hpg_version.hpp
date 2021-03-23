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
