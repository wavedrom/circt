

cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# set (CMAKE_CXX_STANDARD 11)
project (node_circt)
# find_package(MLIR REQUIRED CONFIG)



# file(GLOB SOURCE_FILES "*.cpp")
add_definitions(-DNAPI_EXPERIMENTAL) # -DSRC_NAPI_INL_H_)



add_definitions(-DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir)

# add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})


# set_target_properties(node-circt PROPERTIES PREFIX "" SUFFIX ".node")
# target_link_libraries(${PROJECT_NAME} PRIVATE
#   ${CMAKE_JS_LIB}
#   MLIRParser
#   MLIRSupport
#   MLIRIR
#   MLIROptLib
#   MLIRStandard
#   MLIRTransforms
#   MLIRTranslation
# )
# 
set(LLVM_LINK_COMPONENTS
  Support
)

# add_llvm_tool(node-circt
#   node-circt.cpp
# )

# llvm_update_compile_flags(node-circt)

include(./node_modules/node-cmake/NodeJS.cmake)
nodejs_init()

add_nodejs_module(node_circt node-circt.cpp)

set_property(TARGET node_circt PROPERTY CXX_STANDARD 14)

# target_link_libraries(node_circt PRIVATE
#   ${CMAKE_JS_LIB}
#   MLIRSupport
#   MLIRIR
#   MLIROptLib
#   MLIRStandard
#   MLIRTransforms
#   MLIRTranslation
# )

target_link_libraries(node_circt PRIVATE
  ${CMAKE_JS_LIB}
  /src/llvm/build/lib/libMLIRSupport.a
  /src/llvm/build/lib/libMLIRIR.a
  /src/llvm/build/lib/libMLIROptLib.a
  /src/llvm/build/lib/libMLIRStandard.a
  /src/llvm/build/lib/libMLIRTransforms.a
  /src/llvm/build/lib/libMLIRTranslation.a
)

message(STATUS ${pwd})

target_include_directories(node_circt PRIVATE
"../../build/include"
"../../llvm/llvm/include"
"../../llvm/mlir/include"
"../../llvm/build/include"
"../../llvm/build/tools/mlir/include"
"../../include"
)

# include_directories(
#   /home/drom/.nvm/versions/node/v14.13.0/include/node
# )
