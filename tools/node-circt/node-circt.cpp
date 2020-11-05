#include <iostream>
#include <napi.h>

#include "llvm/Support/raw_ostream.h"
// #include "llvm/Support/CommandLine.h"
#include "mlir/Support/LLVM.h"
// #include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
// #include "mlir/Pass/PassManager.h"
#include "circt/Dialect/RTL/Dialect.h"
#include "circt/Dialect/RTL/Ops.h"

Napi::Object getNewContext(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::Object obj = Napi::Object::New(env);

  auto *context = new mlir::MLIRContext(/*loadAllDialects=*/false);

  context->loadDialect<circt::rtl::RTLDialect>();

  auto *builder = new mlir::OpBuilder(context);

  auto *theModule = new mlir::OwningModuleRef(mlir::ModuleOp::create(
    builder->getUnknownLoc()
  ));

  obj["context"] = Napi::External<mlir::MLIRContext>::New(env, context);
  obj["module"] = Napi::External<mlir::OwningModuleRef>::New(env, theModule);
  obj["builder"] = Napi::External<mlir::OpBuilder>::New(env, builder);

  return obj;
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
  Napi::Value valModule2 = obj["module"];
  Napi::External<mlir::OwningModuleRef> extModule2 =
      valModule2.As<Napi::External<mlir::OwningModuleRef>>();

  mlir::OwningModuleRef *theModule2 = extModule2.Data();

  const char *res = "world"; // builder? module?

  theModule2->get().print(llvm::errs());

  // char* res;
  // llvm::raw_ostream os;

  return Napi::String::New(env, res);
}

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
  exports["getNewContext"] = Napi::Function::New(env, getNewContext);
  exports["toStringMLIR"] = Napi::Function::New(env, toStringMLIR);
  return exports;
}

NODE_API_MODULE(NODE_GYP_MODULE_NAME, InitAll)
