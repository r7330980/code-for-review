cmake_minimum_required(VERSION 3.16)
project(src-analyzer)

cmake_policy(SET CMP0069 NEW)
include(CheckIPOSupported)
check_ipo_supported()

# find_package(LLVM 10 REQUIRED CONFIG)
find_package(LLVM 12 REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
message("${LLVM_LIBRARY_DIRS}")
set(Clang_DIR "/lib/cmake/clang-12")

set(CMAKE_CXX_FLAGS " -g -O0 --std=c++17 ")
# set(CMAKE_CXX_FLAGS " -g -O3 --std=c++17 ")
# set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)


find_package(Clang REQUIRED CONFIG PATHS "/lib/cmake/clang-12/ClangConfig.cmake")
# find_package(Clang 12 REQUIRED CONFIG )#PATHS "/lib/cmake/clang-10/ClangConfig.cmake")
message("${Clang_LIBRARY_DIRS}")

add_definitions(${Clang_DEFINITIONS})

include_directories(
  ${Clang_INCLUDE_DIRS}
)

include_directories(include)

# include("/usr/lib/llvm-12/cmake/AddLLVM.cmake")

find_package(Boost COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(src-analyzer driver.cc utils.cc
  analysis/ir_builder.cc
  analysis/gennm_ir.cc  
  gennm_ir.pb.cc
  pb_printer.cc
)

target_link_libraries(src-analyzer clangTooling)
target_link_libraries(src-analyzer protobuf)
target_link_libraries(src-analyzer ${Boost_LIBRARIES})
# add LTO
# set_target_properties(propagator PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)

message(STATUS "${Clang_LIBRARIES}")