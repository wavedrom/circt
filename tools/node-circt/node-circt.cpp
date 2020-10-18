#include <node_api.h>

#include "circt/Dialect/RTL/Dialect.h"
#include "llvm/Support/InitLLVM.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Builders.h"
// #include "mlir-c/IR.h"

// #include "circt/Dialect/FIRRTL/Dialect.h"
// #include "circt/Dialect/FIRRTL/Ops.h"
// #include "circt/Dialect/FIRRTL/Passes.h"
// #include "circt/Dialect/RTL/Dialect.h"
// #include "circt/Dialect/SV/Dialect.h"
// #include "circt/EmitVerilog.h"
// #include "circt/FIRParser.h"
// #include "mlir/Dialect/StandardOps/IR/Ops.h"
// #include "mlir/IR/Module.h"
// #include "mlir/Parser.h"
// #include "mlir/Pass/Pass.h"
// #include "mlir/Pass/PassManager.h"
// #include "mlir/Support/FileUtilities.h"
// #include "mlir/Transforms/Passes.h"
// #include "llvm/Support/CommandLine.h"
// #include "llvm/Support/InitLLVM.h"
// #include "llvm/Support/ToolOutputFile.h"

using namespace circt;
// using namespace firrtl;
using namespace mlir;
// using llvm::SMLoc;
// using llvm::SourceMgr;

namespace circt {

  napi_value Method(napi_env env, napi_callback_info args) {
    napi_value greeting;
    napi_status status;

    status = napi_create_string_utf8(env, "world", NAPI_AUTO_LENGTH, &greeting);
    if (status != napi_ok) return nullptr;
    return greeting;
  }

  napi_value newContext(napi_env env, napi_value exports) {
    // OpBuilder builder();
    MLIRContext context;
    return nullptr;
  }

  napi_value init(napi_env env, napi_value exports) {
    napi_status status;
    napi_value fn;

    status = napi_create_function(env, nullptr, 0, Method, nullptr, &fn);
    if (status != napi_ok) return nullptr;

    status = napi_set_named_property(env, exports, "hello", fn);
    if (status != napi_ok) return nullptr;
    return exports;
  }



  NAPI_MODULE(NODE_GYP_MODULE_NAME, init)

} // namespace circt
