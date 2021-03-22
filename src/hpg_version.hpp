#pragma once
#include "hpg_export.h"

namespace hpg {

  HPG_EXPORT const char* version();
  HPG_EXPORT unsigned version_major();
  HPG_EXPORT unsigned version_minor();
  HPG_EXPORT unsigned version_patch();
  HPG_EXPORT unsigned version_tweak();

}  // end namespace hpg

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
