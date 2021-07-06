cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(Pytorch_ONNX_ex)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_PREFIX_PATH ~/.local/lib/python3.6/site-packages/torch/share/cmake/Torch)
set(Torch_DIR ~/.local/lib/python3.6/site-packages/torch/include/torch/csrc/api/include/torch)
set(CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)

add_executable(Pytorch_ONNX_ex rlcustom_operator.cpp)
target_link_libraries(Pytorch_ONNX_ex "${TORCH_LIBRARIES}")

#message("Supported features = ${CMAKE_CXX_COMPILE_FEATURES}")
