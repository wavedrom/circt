// RUN: circt-opt --pass-pipeline='handshake.func(canonicalize,lower-handshake-to-firrtl)' %s | circt-opt --pass-pipeline='firrtl.circuit(lower-firrtl-types,lower-firrtl-to-rtl)' | circt-translate --emit-verilog | FileCheck %s

module {
  handshake.func @simple_addi(%arg0: index, %arg1: index, %arg2: none, ...) -> (index, none) {
    %0 = "handshake.merge"(%arg0) : (index) -> index
    %1 = "handshake.merge"(%arg1) : (index) -> index
    %2 = addi %0, %1 : index
    handshake.return %2, %arg2 : index, none
  }
}

// CHECK: // Standard header to adapt well known macros to our needs.\n";
// CHECK: `ifdef RANDOMIZE_GARBAGE_ASSIGN
// CHECK: `define RANDOMIZE
// CHECK: `endif
// CHECK: `ifdef RANDOMIZE_INVALID_ASSIGN
// CHECK: `define RANDOMIZE
// CHECK: `endif
// CHECK: `ifdef RANDOMIZE_REG_INIT
// CHECK: `define RANDOMIZE
// CHECK: `endif
// CHECK: `ifdef RANDOMIZE_MEM_INIT
// CHECK: `define RANDOMIZE
// CHECK: `endif
// CHECK: `ifndef RANDOM
// CHECK: `define RANDOM $random
// CHECK: `endif
// CHECK: // Users can define 'PRINTF_COND' to add an extra gate to prints.
// CHECK: `ifdef PRINTF_COND
// CHECK: `define PRINTF_COND_ (`PRINTF_COND)
// CHECK: `else
// CHECK: `define PRINTF_COND_ 1
// CHECK: `endif
// CHECK: // Users can define 'STOP_COND' to add an extra gate to stop conditions.
// CHECK: `ifdef STOP_COND
// CHECK: `define STOP_COND_ (`STOP_COND)
// CHECK: `else
// CHECK: `define STOP_COND_ 1
// CHECK: `endif
// CHECK-EMPTY:
// CHECK: // Users can define INIT_RANDOM as general code that gets injected into the
// CHECK: // initializer block for modules with registers.
// CHECK: `ifndef INIT_RANDOM
// CHECK: `define INIT_RANDOM
// CHECK: `endif
// CHECK-EMPTY:
// CHECK: // If using random initialization, you can also define RANDOMIZE_DELAY to
// CHECK: // customize the delay used, otherwise 0.002 is used.
// CHECK: `ifndef RANDOMIZE_DELAY
// CHECK: `define RANDOMIZE_DELAY 0.002
// CHECK: `endif
// CHECK-EMPTY:
// CHECK: // Define INIT_RANDOM_PROLOG_ for use in our modules below.
// CHECK: `ifdef RANDOMIZE
// CHECK:   `ifndef VERILATOR
// CHECK:     `define INIT_RANDOM_PROLOG_ `INIT_RANDOM #`RANDOMIZE_DELAY begin end
// CHECK:   `else
// CHECK:     `define INIT_RANDOM_PROLOG_ `INIT_RANDOM
// CHECK:   `endif
// CHECK: `else
// CHECK:   `define INIT_RANDOM_PROLOG_
// CHECK: `endif
// CHECK: module std.addi_2ins_1outs(
// CHECK:   input         arg0_valid,
// CHECK:   output        arg0_ready,
// CHECK:   input  [63:0] arg0_data,
// CHECK:   input         arg1_valid,
// CHECK:   output        arg1_ready,
// CHECK:   input  [63:0] arg1_data,
// CHECK:   output        arg2_valid,
// CHECK:   input         arg2_ready,
// CHECK:   output [63:0] arg2_data);
// CHECK-EMPTY:
// CHECK:   assign arg2_data = arg0_data + arg1_data;	// <stdin>:6:12, :7:12, :8:12, :9:12, :10:12, :11:7
// CHECK:   wire _T = arg0_valid & arg1_valid;	// <stdin>:12:12, :13:12, :14:12
// CHECK:   assign arg2_valid = _T;	// <stdin>:15:12, :16:12, :17:7
// CHECK:   wire _T_0 = arg2_ready & _T;	// <stdin>:18:13, :19:13
// CHECK:   assign arg0_ready = _T_0;	// <stdin>:20:13, :21:13, :22:7
// CHECK:   assign arg1_ready = _T_0;	// <stdin>:23:13, :24:13, :25:7
// CHECK: endmodule
// CHECK-EMPTY:
// CHECK: module simple_addi(
// CHECK:   input         arg0_valid,
// CHECK:   output        arg0_ready,
// CHECK:   input  [63:0] arg0_data,
// CHECK:   input         arg1_valid,
// CHECK:   output        arg1_ready,
// CHECK:   input  [63:0] arg1_data,
// CHECK:   input         arg2_valid,
// CHECK:   output        arg2_ready, arg3_valid,
// CHECK:   input         arg3_ready,
// CHECK:   output [63:0] arg3_data,
// CHECK:   output        arg4_valid,
// CHECK:   input         arg4_ready, clock, reset);
// CHECK-EMPTY:
// CHECK:   wire        _T_arg0_valid;	// <stdin>:28:12
// CHECK:   wire        _T_arg0_ready;	// <stdin>:28:12
// CHECK:   wire [63:0] _T_arg0_data;	// <stdin>:28:12
// CHECK:   wire        _T_arg1_valid;	// <stdin>:28:12
// CHECK:   wire        _T_arg1_ready;	// <stdin>:28:12
// CHECK:   wire [63:0] _T_arg1_data;	// <stdin>:28:12
// CHECK:   wire        _T_arg2_valid;	// <stdin>:28:12
// CHECK:   wire        _T_arg2_ready;	// <stdin>:28:12
// CHECK:   wire [63:0] _T_arg2_data;	// <stdin>:28:12
// CHECK-EMPTY:
// CHECK:   std.addi_2ins_1outs _T (	// <stdin>:28:12
// CHECK:     .arg0_valid(_T_arg0_valid),
// CHECK:     .arg0_ready(_T_arg0_ready),
// CHECK:     .arg0_data(_T_arg0_data),
// CHECK:     .arg1_valid(_T_arg1_valid),
// CHECK:     .arg1_ready(_T_arg1_ready),
// CHECK:     .arg1_data(_T_arg1_data),
// CHECK:     .arg2_valid(_T_arg2_valid),
// CHECK:     .arg2_ready(_T_arg2_ready),
// CHECK:     .arg2_data(_T_arg2_data)
// CHECK:   );
// CHECK:   assign _T_arg0_valid = arg0_valid;	// <stdin>:29:12, :32:12, :33:12, :34:12, :35:7
// CHECK:   assign _T_arg0_ready = arg0_ready;	// <stdin>:30:12, :36:12, :37:12, :38:12, :39:7
// CHECK:   assign _T_arg0_data = arg0_data;	// <stdin>:31:12, :40:13, :41:13, :42:13, :43:7
// CHECK:   assign _T_arg1_valid = arg1_valid;	// <stdin>:44:13, :47:13, :48:13, :49:13, :50:7
// CHECK:   assign _T_arg1_ready = arg1_ready;	// <stdin>:45:13, :51:13, :52:13, :53:13, :54:7
// CHECK:   assign _T_arg1_data = arg1_data;	// <stdin>:46:13, :55:13, :56:13, :57:13, :58:7
// CHECK:   assign arg3_valid = _T_arg2_valid;	// <stdin>:59:13, :62:13, :63:13, :64:13, :65:7
// CHECK:   assign arg3_ready = _T_arg2_ready;	// <stdin>:60:13, :66:13, :67:13, :68:13, :69:7
// CHECK:   assign arg3_data = _T_arg2_data;	// <stdin>:61:13, :70:13, :71:13, :72:13, :73:7
// CHECK:   assign arg4_valid = arg2_valid;	// <stdin>:74:13, :75:13, :76:13, :77:7
// CHECK:   assign arg4_ready = arg2_ready;	// <stdin>:78:13, :79:13, :80:13, :81:7
// CHECK: endmodule
