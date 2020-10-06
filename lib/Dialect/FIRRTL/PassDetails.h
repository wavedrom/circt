//===- PassDetails.h - FIRRTL pass class details ----------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_FIRRTL_PASSDETAILS_H
#define DIALECT_FIRRTL_PASSDETAILS_H

#include "circt/Dialect/FIRRTL/Ops.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace firrtl {

#define GEN_PASS_CLASSES
#include "circt/Dialect/FIRRTL/FIRRTLPasses.h.inc"

} // namespace firrtl
} // namespace circt

#endif // DIALECT_FIRRTL_PASSDETAILS_H
