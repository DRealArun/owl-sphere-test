# Installation
An Nvidia graphics card is necessary to run this software. In addition to that the following software packages will also be necessary.
- Install Nvidia driver, Cuda and Optix 7. Please follow this blog post [blog post](https://ingowald.blog/installing-the-latest-nvidia-driver-cuda-and-optix-on-linux-ubuntu-18-04/) to install these packages.
- Install OWL (A Node Graph "Wrapper" Library for OptiX 7). Please follow the Building OWL instructions provided on OWL github repository [OWL github repository](https://github.com/owl-project/owl/tree/ce012a9c575044cc7a118dffdf11f16712949869#using-owl-through-cmake)
- Eigen3 3.3
```
sudo apt-get update
sudo apt-get install libeigen3-dev
```
- nlohmann_json 3.2.0. Use this command, 
```
sudo apt-get update
sudo apt-get install nlohmann-json3-dev
```
- OpenCV 4.5.1 with GPU support. Build and install this using the instructions provided in this [OpenCV blog post](https://learnopencv.com/opencv-dnn-with-gpu-support/)


# Build
- Clone the repository
```
git clone --recurse-submodules https://github.com/DRealArun/owl-sphere-test

```
- Make data and models directories inside the github repository
```
cd owl-sphere-test
mkdir data
mkdir models
```
- Copy the blender simulation dataset to the data directory (Ensure that the depth maps, images and .json file are in the data directory)

- Download the monocular depth estimation models (links below) into the models directory that we created,
    - MiDaS v2.1 Small : https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx
    - MiDaS v2.1 Large : https://github.com/isl-org/MiDaS/releases/download/v2_1/model-f6b98070.onnx

- Build by running the following commands
```
mkdir build
cd build
cmake ..
make
```

# Running the application
- Update the ``OpenCV_DIR`` in the ``CMakeLists.txt`` to point to your OpenCV installation (Folder that contains the ``OpenCVConfig.cmake`` and other ``.cmake`` files).

- Enable openexr support at runtime by running the following command in the terminal,
```
export OPENCV_IO_ENABLE_OPENEXR=1 # This sets the enviroment variable used by opencv
```

- To run the application on simulation blender data using groundtruth depth maps, run the following command.

```
./sphere-test
```

- To run the application on simulation blender data using MiDaS v2.1 Small model, run the following command.

```
./sphere-test --infer-depth-s
```

- To run the application on simulation blender data using MiDaS v2.1 Small model, run the following command.

```
./sphere-test --infer-depth-l
```

