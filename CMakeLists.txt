# ======================================================================== #
# Copyright 2020 Ingo Wald                                                 #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

project(sphere-test)

cmake_minimum_required(VERSION 3.1.3)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(owl_dir ${CMAKE_CURRENT_SOURCE_DIR}/submodules/owl)
add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)
include_directories(${OWL_INCLUDES})
include(embed_ptx)

include_directories(${owl_dir}/samples/common)
# SET(OpenCV_DIR /home/arun/Desktop/Workspace/installation/OpenCV-3.4.4/share/OpenCV)
SET(OpenCV_DIR /home/arun/Desktop/Workspace/opencv_4.5/installation/OpenCV-4.5/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(nlohmann_json 3.2.0 REQUIRED)

embed_ptx(
  OUTPUT_TARGET
    sphere-test-ptx
  EMBEDDED_SYMBOL_NAMES
    embedded_deviceCode
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    deviceCode.cu
)
add_executable(sphere-test
  Renderer.cpp
  viewer.cpp
  )
target_link_libraries(sphere-test
  PRIVATE
  sphere-test-ptx
  owl_viewer
  ${OpenCV_LIBS}
  Eigen3::Eigen
  nlohmann_json::nlohmann_json
  )
