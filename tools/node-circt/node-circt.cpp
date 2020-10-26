#include <napi.h>

// #include "llvm/Support/CommandLine.h"
// #include "mlir/Support/LLVM.h"
#include "mlir/IR/Module.h"
// #include "mlir/Pass/PassManager.h"
#include "circt/Dialect/RTL/Dialect.h"

Napi::Object getNewContext(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::Object cxt = Napi::Object::New(env);

  mlir::MLIRContext context(false);
  context.loadDialect<circt::rtl::RTLDialect>();
  cxt.Set("mlir", Napi::External<mlir::MLIRContext>::New(env, &context));

  return cxt;
}

Napi::String toStringMLIR(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  Napi::Value arg0 = info[0];
  if (arg0.Type() != napi_object) {
    Napi::Error::New(env, "An object argument was expected.")
        .ThrowAsJavaScriptException();
    return Napi::String::New(env, "?");
  }
  Napi::Object obj = arg0.As<Napi::Object>();
  const Napi::Value propMlir = obj.Get("mlir");
  if (propMlir.Type() != napi_external) {
    Napi::Error::New(env, "An external property of argument was expected.")
        .ThrowAsJavaScriptException();
    return Napi::String::New(env, "??");
  }
  Napi::External<mlir::MLIRContext> extContext =
      propMlir.As<Napi::External<mlir::MLIRContext>>();
  mlir::MLIRContext *context = extContext.Data();

  return Napi::String::New(env, "world");
}

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
  exports.Set("toStringMLIR", Napi::Function::New(env, toStringMLIR));
  exports.Set("getNewContext", Napi::Function::New(env, getNewContext));
  return exports;
}

NODE_API_MODULE(NODE_GYP_MODULE_NAME, InitAll)
