//===- LowerTypes.cpp - Lower FIRRTL types to ground types ----------------===//
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/Ops.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/FIRRTL/Types.h"
#include "circt/Dialect/FIRRTL/Visitors.h"
#include "circt/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"
using namespace circt;
using namespace firrtl;

static Type getPortType(FlatBundleFieldEntry field) {
  return field.isOutput && field.type.isa<FIRRTLType>()
             ? FlipType::get(field.type.dyn_cast<FIRRTLType>())
             : field.type;
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct FIRRTLTypesLowering
    : public LowerFIRRTLTypesBase<FIRRTLTypesLowering>,
      public FIRRTLVisitor<FIRRTLTypesLowering, LogicalResult> {

  using ValueField = std::pair<Value, std::string>;
  using FIRRTLVisitor<FIRRTLTypesLowering, LogicalResult>::visitDecl;
  using FIRRTLVisitor<FIRRTLTypesLowering, LogicalResult>::visitExpr;
  using FIRRTLVisitor<FIRRTLTypesLowering, LogicalResult>::visitStmt;

  void runOnOperation() override;

  // Lowering module block arguments.
  void lowerArg(BlockArgument arg, BundleType type);
  void lowerArg(BlockArgument arg, Type type);

  // Lowering operations.
  LogicalResult visitDecl(InstanceOp op);
  LogicalResult visitExpr(SubfieldOp op);
  LogicalResult visitStmt(ConnectOp op);

  // Skip over any other operations.
  LogicalResult visitUnhandledOp(Operation *op) { return success(); }
  LogicalResult visitInvalidOp(Operation *op) { return success(); }

  // Helpers to manage state.
  Value addArgument(Type type, unsigned oldArgNumber,
                    StringRef nameSuffix = "");

  void setBundleLowering(Value oldValue, StringRef flatField, Value newValue);
  Optional<Value> getBundleLowering(Value oldValue, StringRef flatField);
  void getAllBundleLowerings(Value oldValue, SmallVectorImpl<Value> &results);

  void setInstanceLowering(Value oldValue, Value newValue);
  Value getInstanceLowering(Value oldValue);

  void setSubfieldLowering(Value subfield, Value rootBundle,
                           std::string suffix);
  Optional<ValueField> getSubfieldLowering(Value subfield);

  void removeArg(unsigned argNumber);
  void removeOp(Operation *op);
  void cleanup();

private:
  // The builder is set and maintained in the main loop.
  OpBuilder *builder;

  // State to keep track of arguments and operations to clean up at the end.
  SmallVector<unsigned, 8> argsToRemove;
  SmallVector<Operation *, 16> opsToRemove;

  // State to keep a mapping from original bundle values to flattened names and
  // flattened values.
  circt::DenseMap<Value, llvm::StringMap<Value>> loweredBundleValues;

  // State to keep a mapping from original instances to new instances with their
  // bundles flattened.
  circt::DenseMap<Value, Value> loweredInstances;

  // State to keep a necessary info when traversing potentially nested subfields
  // of bundles: the root bundle and the fieldname suffix computed so far.
  circt::DenseMap<Value, ValueField> loweredSubfieldInfo;
};
} // end anonymous namespace

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createLowerFIRRTLTypesPass() {
  return std::make_unique<FIRRTLTypesLowering>();
}

// This is the main entrypoint for the lowering pass.
void FIRRTLTypesLowering::runOnOperation() {
  auto module = getOperation();
  auto *body = module.getBodyBlock();

  OpBuilder theBuilder(&getContext());
  builder = &theBuilder;

  // Lower the module block arguments.
  SmallVector<BlockArgument, 8> args(body->args_begin(), body->args_end());
  for (auto arg : args)
    mlir::TypeSwitch<Type>(arg.getType())
        .Case<BundleType>([&](BundleType type) { lowerArg(arg, type); })
        .Default<>([&](Type type) { lowerArg(arg, type); });

  // Lower the operations.
  for (auto &op : body->getOperations()) {
    builder->setInsertionPoint(&op);
    if (failed(dispatchVisitor(&op)))
      return;
  }

  cleanup();
}

//===----------------------------------------------------------------------===//
// Lowering module block arguments.
//===----------------------------------------------------------------------===//

// Lower arguments with bundle type by flattening them.
void FIRRTLTypesLowering::lowerArg(BlockArgument arg, BundleType type) {
  // Flatten the bundle types.
  SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
  flattenBundleTypes(type, "", false, fieldTypes);

  for (auto field : fieldTypes) {
    // Create flattened block arguments.
    auto type = getPortType(field);
    auto newValue = addArgument(type, arg.getArgNumber(), field.suffix);

    // Map the flattened suffix for the original argument to the new value.
    setBundleLowering(arg, field.suffix, newValue);
  }
}

// Lower any other arguments by copying them to keep the relative order.
void FIRRTLTypesLowering::lowerArg(BlockArgument arg, Type type) {
  auto newValue = addArgument(type, arg.getArgNumber());
  arg.replaceAllUsesWith(newValue);
}

//===----------------------------------------------------------------------===//
// Lowering operations.
//===----------------------------------------------------------------------===//

// Lower instance operations in the same way as module block arguments. Bundles
// are flattened, and other arguments are copied to keep the relative order. By
// ensuring both lowerings are the same, we can process every module in the
// circuit in parallel, and every instance will have the correct ports.
LogicalResult FIRRTLTypesLowering::visitDecl(InstanceOp op) {
  auto originalType = op.result().getType().dyn_cast<BundleType>();
  if (!originalType) {
    op.emitError() << "instance result was not bundle type.";
    return failure();
  }

  // Create a new, flat bundle type for the new result
  SmallVector<BundleType::BundleElement, 8> bundleElements;
  for (auto element : originalType.getElements()) {
    if (element.second.isa<BundleType>()) {
      // Flatten nested bundle types the usual way.
      SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
      flattenBundleTypes(element.second, element.first, false, fieldTypes);

      for (auto field : fieldTypes) {
        // Store the flat type for the new bundle type.
        auto flatName = builder->getIdentifier(field.suffix);
        auto flatType = getPortType(field).cast<FIRRTLType>();
        auto newElement = BundleType::BundleElement(flatName, flatType);
        bundleElements.push_back(newElement);
      }
    } else {
      // Store the original type for the new bundle type.
      bundleElements.push_back(element);
    }
  }

  // Get the new bundle type and create a new instance.
  auto newType = BundleType::get(bundleElements, &getContext());

  auto newInstance = builder->create<InstanceOp>(
      op.getLoc(), newType, op.moduleNameAttr(), op.nameAttr());

  // Store the mapping from the old instance to the new one.
  setInstanceLowering(op, newInstance);

  removeOp(op);

  return success();
}

// Lowering subfield operations has to deal with three different cases:
//   1. the input value is from a module block argument
//   2. the input value is from another subfield operation's result
//   3. the input value is from an instance
LogicalResult FIRRTLTypesLowering::visitExpr(SubfieldOp op) {
  Value input = op.input();
  Value result = op.result();
  StringRef fieldname = op.fieldname();

  Value rootBundle;
  SmallString<16> suffix;
  if (input.isa<BlockArgument>()) {
    // When the input is from a block argument, map the result to the root
    // bundle block argument, and begin creating a suffix string.
    rootBundle = input;
    suffix.push_back(kFlatBundleFieldSeparator);
    suffix += fieldname;
  } else if (input.isa<OpResult>()) {
    Operation *owner = input.cast<OpResult>().getOwner();
    if (isa<SubfieldOp>(owner)) {
      // When the input is from a subfield, look up the lowered info from the
      // parent.
      auto subfieldLowering = getSubfieldLowering(input);
      if (!subfieldLowering.hasValue()) {
        op.emitError() << "didn't find subfield lowering for input";
        return failure();
      }

      // Map the result to the root bundle block argument, and append the
      // suffix to what was already mapped.
      auto subfieldInfo = subfieldLowering.getValue();
      rootBundle = subfieldInfo.first;
      suffix.assign(subfieldInfo.second);
      suffix.push_back(kFlatBundleFieldSeparator);
      suffix.append(fieldname);
    } else if (isa<InstanceOp>(owner)) {
      // When the input is from an instance, look up the new instance.
      Optional<Value> newInstance = getInstanceLowering(input);
      if (!newInstance.hasValue()) {
        op.emitError() << "couldn't get lowered instance.";
      }

      // Check if this subfield was originally a bundle type.
      BundleType oldType = input.getType().cast<BundleType>();
      FIRRTLType elementType = oldType.getElementType(fieldname);
      if (!elementType) {
        op.emitError() << "no element found for " << fieldname << " in "
                       << oldType << ".";
        return failure();
      }

      if (elementType.isa<BundleType>()) {
        // If it was a bundle, flatten it the usual way.
        SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
        flattenBundleTypes(elementType, fieldname, false, fieldTypes);

        for (auto field : fieldTypes) {
          // Create a new subfield for the flattened field.
          auto newSubfield =
              builder->create<SubfieldOp>(op.getLoc(), getPortType(field),
                                          newInstance.getValue(), field.suffix);

          // Use the original result and the sub-field name (without the prefix)
          // to remember the new subfield.
          auto fieldName = StringRef(field.suffix).drop_front(fieldname.size());
          setBundleLowering(result, fieldName, newSubfield);
        }
      }
    }
  }

  Optional<Value> newValue = getBundleLowering(rootBundle, suffix);
  if (newValue.hasValue()) {
    // If the lowering for the current result exists, we have finished
    // flattening. Replace the current result with the flattened value.
    result.replaceAllUsesWith(newValue.getValue());
  } else {
    // Otherwise, remember the root bundle and the suffix so far.
    setSubfieldLowering(result, rootBundle, std::string(suffix));
  }

  removeOp(op);

  return success();
}

// Lowering connects only has to deal with one special case: connecting two
// bundles. This situation should only arise when both of the arguments are a
// bundle that was:
//   a) originally a block argument
//   b) originally an instance's port
//
// When two such bundles are connected, none of the subfield visits have a
// chance to lower them, so we must ensure they have the same number of
// flattened values and flatten out this connect into multiple connects.
LogicalResult FIRRTLTypesLowering::visitStmt(ConnectOp op) {
  Value lhs = op.lhs();
  Value rhs = op.rhs();

  // If we aren't connecting two bundles, there is nothing to do.
  if (!(lhs.getType().isa<BundleType>() && rhs.getType().isa<BundleType>()))
    return success();

  // Get the lowered values for each side.
  SmallVector<Value, 8> lhsValues;
  getAllBundleLowerings(lhs, lhsValues);

  SmallVector<Value, 8> rhsValues;
  getAllBundleLowerings(rhs, rhsValues);

  // Check that we got out the same number of values from each bundle.
  if (lhsValues.size() != rhsValues.size()) {
    op.emitError() << "lhs value expands to " << lhsValues.size()
                   << " values, but rhs value expands to " << rhsValues.size();
    return failure();
  }

  for (auto tuple : llvm::zip_first(lhsValues, rhsValues)) {
    Value newLhs = std::get<0>(tuple);
    Value newRhs = std::get<1>(tuple);
    builder->create<ConnectOp>(op.getLoc(), newLhs, newRhs);
  }

  removeOp(op);

  return success();
}

//===----------------------------------------------------------------------===//
// Helpers to manage state.
//===----------------------------------------------------------------------===//

static DictionaryAttr getArgAttrs(StringAttr nameAttr, StringRef suffix,
                                  OpBuilder *builder) {
  SmallString<16> newName(nameAttr.getValue());
  newName += suffix;

  StringAttr newNameAttr = builder->getStringAttr(newName);
  Identifier identifier = builder->getIdentifier(kFIRRTLName);
  NamedAttribute attr = NamedAttribute(identifier, newNameAttr);

  return builder->getDictionaryAttr(attr);
}

// Creates and returns a new block argument of the specified type to the module.
// This also maintains the name attribute for the new argument, possibly with a
// new suffix appended.
Value FIRRTLTypesLowering::addArgument(Type type, unsigned oldArgNumber,
                                       StringRef nameSuffix) {
  FModuleOp module = getOperation();
  Block *body = module.getBodyBlock();

  // Append the new argument.
  auto newValue = body->addArgument(type);

  // Keep the module's type up-to-date.
  module.setType(builder->getFunctionType(body->getArgumentTypes(), {}));

  // Copy over the name attribute for the new argument.
  StringAttr nameAttr = getFIRRTLNameAttr(module.getArgAttrs(oldArgNumber));
  if (nameAttr) {
    auto newArgAttrs = getArgAttrs(nameAttr, nameSuffix, builder);
    module.setArgAttrs(newValue.getArgNumber(), newArgAttrs);
  }

  // Remember to delete the original block argument.
  removeArg(oldArgNumber);

  return newValue;
}

// Store the mapping from a bundle typed value to a mapping from its field names
// to flat values.
void FIRRTLTypesLowering::setBundleLowering(Value oldValue, StringRef flatField,
                                            Value newValue) {
  loweredBundleValues[oldValue][flatField] = newValue;
}

// For a mapped bundle typed value and a flat subfield name, retrieve and return
// the flat value if it exists.
Optional<Value> FIRRTLTypesLowering::getBundleLowering(Value oldValue,
                                                       StringRef flatField) {
  if (oldValue && loweredBundleValues.count(oldValue) &&
      loweredBundleValues[oldValue].count(flatField)) {
    return Optional<Value>(loweredBundleValues[oldValue][flatField]);
  } else {
    return None;
  }
}

// For a mapped bundle typed value, retrieve and return the flat values for each
// field.
void FIRRTLTypesLowering::getAllBundleLowerings(
    Value value, SmallVectorImpl<Value> &results) {
  if (loweredBundleValues.count(value)) {
    BundleType bundleType = value.getType().cast<BundleType>();
    for (auto element : bundleType.getElements()) {
      SmallString<16> loweredName;
      loweredName.push_back(kFlatBundleFieldSeparator);
      loweredName.append(element.first);

      if (loweredBundleValues[value].count(loweredName)) {
        results.push_back(loweredBundleValues[value][loweredName]);
      } else {
        getOperation().emitError()
            << "expected to find " << loweredName << " for value.";
      }
    }
  }
}

// Store the mapping from an original instance value to a new instance value
// that has its bundles flattened.
void FIRRTLTypesLowering::setInstanceLowering(Value oldValue, Value newValue) {
  loweredInstances[oldValue] = newValue;
}

// For a mapped instance value, retrieve and return the new instance with
// bundles flattened.
Value FIRRTLTypesLowering::getInstanceLowering(Value oldValue) {
  return loweredInstances[oldValue];
}

// Store the mapping from a subfield to its root bundle as well as the currently
// constructed suffix. Eventually, the full suffix will be built, which will map
// to a flat field name.
void FIRRTLTypesLowering::setSubfieldLowering(Value subfield, Value rootBundle,
                                              std::string suffix) {
  loweredSubfieldInfo[subfield] = std::make_pair(rootBundle, suffix);
}

// For a mapped subfield value, retrieve and return the root bundle as well as
// the suffix that had been constructed so far.
Optional<FIRRTLTypesLowering::ValueField>
FIRRTLTypesLowering::getSubfieldLowering(Value subfield) {
  if (loweredSubfieldInfo.count(subfield)) {
    return Optional<ValueField>(loweredSubfieldInfo[subfield]);
  } else {
    return None;
  }
}

// Remember an argument number to erase during cleanup.
void FIRRTLTypesLowering::removeArg(unsigned argNumber) {
  argsToRemove.push_back(argNumber);
}

// Remember an operation to erase during cleanup.
void FIRRTLTypesLowering::removeOp(Operation *op) { opsToRemove.push_back(op); }

// Handle deferred removals of operations and block arguments when done. Also
// clean up state.
void FIRRTLTypesLowering::cleanup() {
  while (!opsToRemove.empty())
    opsToRemove.pop_back_val()->erase();

  getOperation().eraseArguments(argsToRemove);
  argsToRemove.clear();

  loweredBundleValues.clear();
  loweredInstances.clear();
  loweredSubfieldInfo.clear();
}
