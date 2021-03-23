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

  /** error description */
  const std::string&
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
