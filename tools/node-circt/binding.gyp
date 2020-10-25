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
        # For some reason, this has an extra ../ compared to the paths above in include_dirs
        "libraries": [
            "../../../build/lib/*.a",
            "../../../llvm/build/lib/*.a",
            "../../../llvm/build/lib/libLLVMSupport.a"
        ]
    }]
}
