# cmake_minimum_required(VERSION 3.18)
# set (CMAKE_CXX_STANDARD 11)
# project (node-circt)
# find_package(MLIR REQUIRED CONFIG)


# file(GLOB SOURCE_FILES "*.cpp")
add_definitions(-DNAPI_EXPERIMENTAL) # -DSRC_NAPI_INL_H_)
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

add_llvm_tool(node-circt
  node-circt.cpp
)

# llvm_update_compile_flags(node-circt)

target_link_libraries(node-circt PRIVATE
  ${CMAKE_JS_LIB}
  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRStandard
  MLIRTransforms
  MLIRTranslation
)

include_directories(
  /home/drom/.nvm/versions/node/v14.13.0/include/node
)
