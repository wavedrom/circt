{
    "targets": [{
        "target_name": "circt",
        'make_global_settings': [
            ['CXX', '/usr/bin/clang++'],
            ['CC', '/usr/bin/clang'],
            ['LINK', '/usr/bin/clang++'],
            ['AR', '/usr/bin/llvm-ar'],
            ['NM', '/usr/bin/llvm-nm']
        ],
        "sources": [
            "node-circt.cpp"
        ],
        "include_dirs": [
            "../../build/include",
            "../../llvm/llvm/include",
            "../../llvm/mlir/include",
            "../../llvm/build/include",
            "../../llvm/build/tools/mlir/include",
            "../../include"
        ],
        "libraries": [
            '/home/drom/work/github/llvm/circt/llvm/build/lib/*.a',
            '/home/drom/work/github/llvm/circt/build/lib/*.a',
            # '/home/drom/work/github/llvm/circt/build/lib/libCIRCTLLHDSimEngine.a',
            # '/home/drom/work/github/llvm/circt/build/lib/libCIRCTLLHDSimState.a',
            # '/home/drom/work/github/llvm/circt/build/lib/libMLIRFIRRTL.a',
            # '/home/drom/work/github/llvm/circt/build/lib/libMLIRFIRRTLToLLHD.a',
            # '/home/drom/work/github/llvm/circt/build/lib/libMLIRHandshakeOps.a',
            # '/home/drom/work/github/llvm/circt/build/lib/libMLIRHandshakeToFIRRTL.a',
            # '/home/drom/work/github/llvm/circt/build/lib/libMLIRLLHD.a',
            # '/home/drom/work/github/llvm/circt/build/lib/libMLIRLLHDTargetVerilog.a',
            # '/home/drom/work/github/llvm/circt/build/lib/libMLIRLLHDToLLVM.a',
            # '/home/drom/work/github/llvm/circt/build/lib/libMLIRLLHDTransforms.a',
            # '/home/drom/work/github/llvm/circt/build/lib/libMLIRRTL.a',
            # '/home/drom/work/github/llvm/circt/build/lib/libMLIRStandardToHandshake.a',
            # '/home/drom/work/github/llvm/circt/build/lib/libMLIRStandardToStaticLogic.a',
            # '/home/drom/work/github/llvm/circt/build/lib/libMLIRStaticLogicOps.a',
            # '/home/drom/work/github/llvm/circt/build/lib/libMLIRSV.a'
            # '../../llvm/build/tools/mlir/lib/'
        ]
    }]
}
