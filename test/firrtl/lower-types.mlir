// RUN: circt-opt -pass-pipeline='firrtl.circuit(lower-firrtl-types)' %s | FileCheck %s

firrtl.circuit "TopLevel" {

  // CHECK-LABEL: firrtl.module @Simple(%source_valid: !firrtl.uint<1>, %source_ready: !firrtl.flip<uint<1>>, %source_data: !firrtl.uint<64>, %sink_valid: !firrtl.flip<uint<1>>, %sink_ready: !firrtl.uint<1>, %sink_data: !firrtl.flip<uint<64>>)
  firrtl.module @Simple(%source: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>,
                        %sink: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) {
    // CHECK: firrtl.when %source_valid {
    // CHECK:   firrtl.connect %sink_data, %source_data : !firrtl.flip<uint<64>>, !firrtl.uint<64>
    // CHECK:   firrtl.connect %sink_valid, %source_valid : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    // CHECK:   firrtl.connect %source_ready, %sink_ready : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    // CHECK: }
    %0 = firrtl.subfield %source("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
    %1 = firrtl.subfield %source("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.flip<uint<1>>
    %2 = firrtl.subfield %source("data") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<64>
    %3 = firrtl.subfield %sink("valid") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.flip<uint<1>>
    %4 = firrtl.subfield %sink("ready") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.uint<1>
    %5 = firrtl.subfield %sink("data") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.flip<uint<64>>
    firrtl.when %0 {
      firrtl.connect %5, %2 : !firrtl.flip<uint<64>>, !firrtl.uint<64>
      firrtl.connect %3, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
      firrtl.connect %1, %4 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    }
  }

  // CHECK-LABEL: firrtl.module @Recursive(%arg_foo_bar_baz: !firrtl.uint<1>, %arg_foo_qux: !firrtl.sint<64>, %out1: !firrtl.flip<uint<1>>, %out2: !firrtl.flip<sint<64>>)
  firrtl.module @Recursive(%arg: !firrtl.bundle<foo: bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>>,
                           %out1: !firrtl.flip<uint<1>>, %out2: !firrtl.flip<sint<64>>) {
    // CHECK: firrtl.connect %out1, %arg_foo_bar_baz : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    // CHECK: firrtl.connect %out2, %arg_foo_qux : !firrtl.flip<sint<64>>, !firrtl.sint<64>
    %0 = firrtl.subfield %arg("foo") : (!firrtl.bundle<foo: bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>>) -> !firrtl.bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>
    %1 = firrtl.subfield %0("bar") : (!firrtl.bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>) -> !firrtl.bundle<baz: uint<1>>
    %2 = firrtl.subfield %1("baz") : (!firrtl.bundle<baz: uint<1>>) -> !firrtl.uint<1>
    %3 = firrtl.subfield %0("qux") : (!firrtl.bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>) -> !firrtl.sint<64>
    firrtl.connect %out1, %2 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    firrtl.connect %out2, %3 : !firrtl.flip<sint<64>>, !firrtl.sint<64>
  }

  // CHECK-LABEL: firrtl.module @Uniquification(%a_b: !firrtl.uint<1>, %a_b_0: !firrtl.uint<1> {firrtl.name = "a_b"})
  firrtl.module @Uniquification(%a: !firrtl.bundle<b: uint<1>>, %a_b: !firrtl.uint<1>) {
  }

  // CHECK-LABEL: firrtl.module @TopLevel(%source_valid: !firrtl.uint<1>, %source_ready: !firrtl.flip<uint<1>>, %source_data: !firrtl.uint<64>, %sink_valid: !firrtl.flip<uint<1>>, %sink_ready: !firrtl.uint<1>, %sink_data: !firrtl.flip<uint<64>>)
  firrtl.module @TopLevel(%source: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>,
                          %sink: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) {
    // CHECK: %0 = firrtl.instance @Simple {name = ""} : !firrtl.bundle<source_valid: flip<uint<1>>, source_ready: uint<1>, source_data: flip<uint<64>>, sink_valid: uint<1>, sink_ready: flip<uint<1>>, sink_data: uint<64>>
    // CHECK: %1 = firrtl.subfield %0("source_valid") : (!firrtl.bundle<source_valid: flip<uint<1>>, source_ready: uint<1>, source_data: flip<uint<64>>, sink_valid: uint<1>, sink_ready: flip<uint<1>>, sink_data: uint<64>>) -> !firrtl.flip<uint<1>>
    // CHECK: %2 = firrtl.subfield %0("source_ready") : (!firrtl.bundle<source_valid: flip<uint<1>>, source_ready: uint<1>, source_data: flip<uint<64>>, sink_valid: uint<1>, sink_ready: flip<uint<1>>, sink_data: uint<64>>) -> !firrtl.uint<1>
    // CHECK: %3 = firrtl.subfield %0("source_data") : (!firrtl.bundle<source_valid: flip<uint<1>>, source_ready: uint<1>, source_data: flip<uint<64>>, sink_valid: uint<1>, sink_ready: flip<uint<1>>, sink_data: uint<64>>) -> !firrtl.flip<uint<64>>
    // CHECK: firrtl.connect %1, %source_valid : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    // CHECK: firrtl.connect %2, %source_ready : !firrtl.uint<1>, !firrtl.flip<uint<1>>
    // CHECK: firrtl.connect %3, %source_data : !firrtl.flip<uint<64>>, !firrtl.uint<64>
    // CHECK: %4 = firrtl.subfield %0("sink_valid") : (!firrtl.bundle<source_valid: flip<uint<1>>, source_ready: uint<1>, source_data: flip<uint<64>>, sink_valid: uint<1>, sink_ready: flip<uint<1>>, sink_data: uint<64>>) -> !firrtl.uint<1>
    // CHECK: %5 = firrtl.subfield %0("sink_ready") : (!firrtl.bundle<source_valid: flip<uint<1>>, source_ready: uint<1>, source_data: flip<uint<64>>, sink_valid: uint<1>, sink_ready: flip<uint<1>>, sink_data: uint<64>>) -> !firrtl.flip<uint<1>>
    // CHECK: %6 = firrtl.subfield %0("sink_data") : (!firrtl.bundle<source_valid: flip<uint<1>>, source_ready: uint<1>, source_data: flip<uint<64>>, sink_valid: uint<1>, sink_ready: flip<uint<1>>, sink_data: uint<64>>) -> !firrtl.uint<64>
    // CHECK: firrtl.connect %sink_valid, %4 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    // CHECK: firrtl.connect %sink_ready, %5 : !firrtl.uint<1>, !firrtl.flip<uint<1>>
    // CHECK: firrtl.connect %sink_data, %6 : !firrtl.flip<uint<64>>, !firrtl.uint<64>
    %0 = firrtl.instance @Simple {name = ""} : !firrtl.bundle<source: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, sink: bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>>
    %1 = firrtl.subfield %0("source") : (!firrtl.bundle<source: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, sink: bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>>) -> !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>
    firrtl.connect %1, %source : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>
    %2 = firrtl.subfield %0("sink") : (!firrtl.bundle<source: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, sink: bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>>) -> !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>
    firrtl.connect %sink, %2 : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>
  }
}
