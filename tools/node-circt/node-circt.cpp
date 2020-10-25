#define NAPI_VERSION 1
#include <node_api.h>

#include "llvm/Support/CommandLine.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/PassManager.h"
#include "circt/Dialect/RTL/Dialect.h"

// using namespace llvm;
// using namespace circt;
// using namespace mlir;

napi_value Hello(napi_env env, napi_callback_info args) {
  napi_value greeting;
  napi_status status;

  status = napi_create_string_utf8(env, "world", NAPI_AUTO_LENGTH, &greeting);
  if (status != napi_ok) return nullptr;
  return greeting;
}

napi_value getNewContext(napi_env env, napi_callback_info args) {
  napi_value res;
  napi_status status;

  mlir::MLIRContext context;
  context.loadDialect<circt::rtl::RTLDialect>();

  status = napi_create_external(env, &context, 0, 0, &res);
  if (status != napi_ok) return nullptr;

  return res;
}

napi_value init(napi_env env, napi_value exports) {
  napi_status status;
  napi_value fn;

  status = napi_create_function(env, nullptr, 0, Hello, nullptr, &fn);
  if (status != napi_ok) return nullptr;

  status = napi_set_named_property(env, exports, "hello", fn);
  if (status != napi_ok) return nullptr;

  status = napi_create_function(env, nullptr, 0, getNewContext, nullptr, &fn);
  if (status != napi_ok) return nullptr;

  status = napi_set_named_property(env, exports, "getNewContext", fn);
  if (status != napi_ok) return nullptr;

  return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, init)
