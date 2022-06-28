// ======================================================================== //
// Copyright 2020-2020 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <fstream>
// #include "mesh.h"
#include "Renderer.h"
#include "deviceCode.h"
#include <owl/owl.h>
#include <iomanip>
#include <string>
#include <array>
#include <map>
#include <random>
#include "projection_utils.h"
#include <unistd.h>
#include <stdlib.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <nlohmann/json.hpp>

extern "C" char embedded_deviceCode[];
using json = nlohmann::json;

namespace dvr
{

  bool Renderer::heatMapEnabled = false;
  float Renderer::heatMapScale = 1e-5f;
  int Renderer::spp = 1;

  cv::Mat Renderer::load_image(std::string imagepath, int numChannels)
  {
    int readflag;
    if (numChannels == 3)
    {
      readflag = cv::IMREAD_COLOR;
    }
    else if (numChannels == 1)
    {
      readflag = cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH;
    }
    else
    {
      std::cout << "Number of channels: ", numChannels, " unsupported";
      exit(EXIT_FAILURE);
    }
    cv::Mat img = cv::imread(imagepath, readflag);
    if (img.empty())
    {
      std::cout << "Could not read the image: " << imagepath << std::endl;
      exit(EXIT_FAILURE);
    }
    // std::cout << "Data type" << cvtype2str(img.type()) << "\t" << "Dim" << img.channels();
    // for(int row = 0; row < img.rows; ++row) {
    //   for(int col = 0; col < img.cols; ++col) {
    //       std::cout << (float) img.data[row*img.cols + col] << "\t";
    //   }
    //   std::cout << std::endl;
    // }
    int down_width = 256;
    int down_height = 256;
    cv::Mat resizeImg;
    resize(img, resizeImg, cv::Size(down_width, down_height), CV_INTER_AREA);
    if (numChannels != 3)
    {
      std::vector<cv::Mat> channels(3);
      cv::split(resizeImg, channels);
      cv::Mat img1d = channels[1];
      cv::Mat img1dGray;
      img1d.convertTo(img1dGray, CV_8UC1);
      // cv::imshow("Depth", img1dGray);
      // int k = cv::waitKey(0);
      return img1d;
    }
    else
    {
      // cv::imshow("Image", resizeImg);
      // int k = cv::waitKey(0);
      return resizeImg;
    }
  }

  OWLVarDecl rayGenVars[] = {
      {nullptr /* sentinel to mark end of list */}};

  OWLVarDecl triangleGeomVars[] = {
      {"indexBuffer", OWL_BUFPTR, OWL_OFFSETOF(TriangleGeom, indexBuffer)},
      {"vertexBuffer", OWL_BUFPTR, OWL_OFFSETOF(TriangleGeom, vertexBuffer)},
      {"slopes", OWL_BUFPTR, OWL_OFFSETOF(TriangleGeom, slopes)},
      {nullptr /* sentinel to mark end of list */}};

  OWLVarDecl launchParamsVars[] = {
      {"fbPointer", OWL_RAW_POINTER, OWL_OFFSETOF(LaunchParams, fbPointer)},
      {"accumBuffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, accumBuffer)},
      {"accumID", OWL_INT, OWL_OFFSETOF(LaunchParams, accumID)},
#ifdef DUMP_FRAMES
      // to allow dumping rgba and depth for some unrelated compositing work....
      {"fbDepth", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, fbDepth)},
#endif
      {"world", OWL_GROUP, OWL_OFFSETOF(LaunchParams, world)},
      {"domain.lower", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, domain.lower)},
      {"domain.upper", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, domain.upper)},
      {"particles", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, particles)},
      {"numParticles", OWL_UINT, OWL_OFFSETOF(LaunchParams, numParticles)},
      {"radius", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, radius)},
      // render settings
      {"render.dt", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, render.dt)},
      {"render.spp", OWL_INT, OWL_OFFSETOF(LaunchParams, render.spp)},
      {"render.heatMapEnabled", OWL_INT, OWL_OFFSETOF(LaunchParams, render.heatMapEnabled)},
      {"render.heatMapScale", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, render.heatMapScale)},
      // camera settings
      {"camera.org", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.org)},
      {"camera.dir_00", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.dir_00)},
      {"camera.dir_du", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.dir_du)},
      {"camera.dir_dv", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.dir_dv)},
      // Model, if in rendering mode
      {"model.group", OWL_GROUP, OWL_OFFSETOF(LaunchParams, model.group)},
      {"model.indexBuffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, model.indexBuffer)},
      {"model.vertexBuffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, model.vertexBuffer)},
      {nullptr /* sentinel to mark end of list */}};

  Renderer::Renderer()
      : xfDomain({0.f, 1.f})
  {
#if 1
    std::default_random_engine rd;
    std::uniform_real_distribution<float> pos(-1.0f, 1.0f);

    float radius = .01f;
    // particles.resize(500);
    box3f domain;
    // Eigen::Matrix2Xf g = generateGridCoordinates(10, 20);
    // for (int i = 0; i < 10; ++i) {
    //     for (int j = 0; j < 20; ++j) {
    //         std::cout << "(" << g(0, (i*20)+j) << "," << g(1, (i*20)+j) << ") ";
    //     }
    //     std::cout << "\n";
    // }
    std::ifstream people_file("/home/arun/Desktop/data/Mock_scene_setup/Mock_scene_annotation.json", std::ifstream::binary);
    json people;
    people_file >> people;
    std::cout << people["camera_2"];

    frameId = 1;
    camId = 1;
    std::ostringstream imgPath, depPath;
    imgPath << "/home/arun/Desktop/data/Mock_scene_setup/RGB_camera_" << frameId << "_" << std::setfill('0') << std::setw(4) << camId << ".jpg";
    depPath << "/home/arun/Desktop/data/Mock_scene_setup/Depth_camera_" << frameId << "_" << std::setfill('0') << std::setw(4) << camId << ".exr";

    cv::Mat img = Renderer::load_image(imgPath.str(), 3);
    cv::Mat dep = Renderer::load_image(depPath.str(), 1);

    // cv::Mat img = Renderer::load_image("/home/arun/Desktop/data/Mock_scene_setup/RGB_camera_1_0001.jpg", 3);
    // cv::Mat dep = Renderer::load_image("/home/arun/Desktop/data/Mock_scene_setup/Depth_camera_1_0001.exr", 1);

    Eigen::Matrix4f K, offset;
    std::array<std::array<float, 3>, 3> Int;
    std::map<std::string, float> Ori;
    std::map<std::string, float> Loc;
    for (json::iterator it = people["camera_1"].begin(); it != people["camera_1"].end(); ++it)
    {
      std::cout << "\n"
                << it.key() << " : " << it.value() << "\n";
      if (it.key() == "K")
      {
        Int = it.value().get<std::array<std::array<float, 3>, 3>>();
      }
      if (it.key() == "orientation")
      {
        Ori = it.value().get<std::map<std::string, float>>();
      }
      if (it.key() == "location")
      {
        Loc = it.value().get<std::map<std::string, float>>();
      }
    }
    std::cout << "Location " << Loc["x"] << " : " << Loc["y"] << " : " << Loc["z"] << std::endl;
    std::cout << "Orientation " << Ori["w"] << " : " << Ori["x"] << " : " << Ori["y"] << " : " << Ori["z"] << std::endl;
    // This works
    // for (auto i: Int){
    //   for (auto j: i) {
    //     std::cout <<  j << " ";
    //   }
    //   std::cout << std::endl;
    // }
    // K << 2666.666748046875 / 1920.0 * 256, 0.0, 960.0 / 1920.0 * 256, 0.0,
    //     0.0, 2666.666748046875 / 1080.0 * 256, 540.0 / 1080.0 * 256, 0.0,
    //     0.0, 0.0, 1.0, 0.0,
    //     0.0, 0.0, 0.0, 1.0;

    K << Int[0][0] / 1920.0 * 256, Int[0][1] / 1920.0 * 256, Int[0][2] / 1920.0 * 256, 0.0,
        Int[1][0] / 1080.0 * 256, Int[1][1] / 1080.0 * 256, Int[1][2] / 1080.0 * 256, 0.0,
        Int[2][0], Int[2][1], Int[2][2], 0.0,
        0.0, 0.0, 0.0, 1.0;

    // offset << 2, 0, -1, 0,
    //           0, 2, -1, 0,
    //           0, 0, 1, 0,
    //           0, 0, 0, 1;
    Eigen::Matrix4Xf pts = projectPix2Camera(dep, K, 0, 10);
    // std::cout << "Camera coords size: " << pts.rows() << " " << pts.cols()  << std::endl;
    // std::cout << "Row min and max: " << pts.row(3).minCoeff() << " " << pts.row(3).maxCoeff()  << std::endl;
    // std::cout << "x min and max: " << pts.row(0).minCoeff() << " " << pts.row(0).maxCoeff()  << std::endl;
    // std::cout << "y min and max: " << pts.row(1).minCoeff() << " " << pts.row(1).maxCoeff()  << std::endl;
    // std::cout << "z min and max: " << pts.row(2).minCoeff() << " " << pts.row(2).maxCoeff()  << std::endl;
    // Since max and min are 1 then we can ignore the 4th component

    // for (int i=0; i<particles.size(); ++i) {
    //     float x=pos(rd);
    //     float y=pos(rd);
    //     float z=pos(rd);
    //     particles[i] = {x,y,z};
    //     domain.extend(particles[i]-vec3f(radius));
    //     domain.extend(particles[i]+vec3f(radius));
    // }

    // Particles must be between -1 and 1
    particles.resize(pts.cols());
    for (int i = 0; i < pts.cols(); ++i)
    {
      float x = pts(0, i);
      float y = pts(1, i);
      float z = pts(2, i);
      particles[i] = {x, y, z};
      domain.extend(particles[i] - vec3f(radius));
      domain.extend(particles[i] + vec3f(radius));
    }
#else
    // write state machine here that can determine if only matrix update is needed or
    // image update is also needed
    // Use arrow keys to get user input. Use this to calculate the position using step update
    // Use this to determine if complete matrix update is needed.

    // std::ifstream in("/Users/stefan/vowl/data/atm_2019_07_01_07_00.tab.out");
    // float radius = 10000.312f;
    // uint64_t size;
    // in.read((char*)&size,sizeof(size));
    // particles.resize(size);
    // std::vector<vec4f> temp(particles.size());
    // in.read((char*)temp.data(),temp.size()*sizeof(vec4f));
    // box3f domain;
    // for (int i=0; i<particles.size(); ++i) {
    //     particles[i] = vec3f(temp[i]);std::cout << particles[i] << '\n';
    //     domain.extend(particles[i]-vec3f(radius));
    //     domain.extend(particles[i]+vec3f(radius));
    // }
#endif

    std::cout << domain << '\n';
    std::cout << particles.size() << '\n';

    modelBounds.extend(domain);

    owl = owlContextCreate(nullptr, 1);
    module = owlModuleCreate(owl, embedded_deviceCode);
    rayGen = owlRayGenCreate(owl, module, "renderFrame",
                             sizeof(RayGen), rayGenVars, -1);
    lp = owlParamsCreate(owl, sizeof(LaunchParams), launchParamsVars, -1);

    // owlParamsSet3i(lp,"volume.dims",
    //                512,
    //                512,
    //                512);
    // owlParamsSet3f(lp,"render.gradientDelta",
    //                1.f/512,
    //                1.f/512,
    //                1.f/512);

#ifdef DUMP_FRAMES
    fbDepth = owlManagedMemoryBufferCreate(owl, OWL_FLOAT, 1, nullptr);
    fbSize = vec2i(1);
    owlParamsSetBuffer(lp, "fbDepth", fbDepth);
#endif

    particlesBuf = owlDeviceBufferCreate(owl,
                                         OWL_USER_TYPE(Particle),
                                         0, nullptr);

    OWLVarDecl particleGeomVars[] = {
        {"world", OWL_GROUP, OWL_OFFSETOF(ParticleGeom, world)},
        {"domain.lower", OWL_FLOAT3, OWL_OFFSETOF(ParticleGeom, domain.lower)},
        {"domain.upper", OWL_FLOAT3, OWL_OFFSETOF(ParticleGeom, domain.upper)},
        {"particles", OWL_BUFPTR, OWL_OFFSETOF(ParticleGeom, particles)},
        {"numParticles", OWL_UINT, OWL_OFFSETOF(ParticleGeom, numParticles)},
        {"radius", OWL_FLOAT, OWL_OFFSETOF(ParticleGeom, radius)},
        {/* sentinel to mark end of list */}};

    geomType = owlGeomTypeCreate(owl,
                                 OWL_GEOMETRY_USER,
                                 sizeof(ParticleGeom),
                                 particleGeomVars, -1);

    owlGeomTypeSetBoundsProg(geomType, module, "Particles");
    owlGeomTypeSetIntersectProg(geomType, 0, module, "Particles");
    owlGeomTypeSetClosestHit(geomType, 0, module, "Particles");

    owlBuildPrograms(owl);

    geom = owlGeomCreate(owl, geomType);

    blasGroup = owlUserGeomGroupCreate(owl, 1, &geom, OPTIX_BUILD_FLAG_ALLOW_UPDATE);
    // tlasGroup = owlInstanceGroupCreate(owl, 1);
    tlasGroup = owlInstanceGroupCreate(owl, 1,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       OWL_MATRIX_FORMAT_OWL,
                                       OPTIX_BUILD_FLAG_ALLOW_UPDATE);
    owlInstanceGroupSetChild(tlasGroup, 0, blasGroup);

    // TODO: restructure; just for testing
    owlBufferResize(particlesBuf, particles.size());
    owlBufferUpload(particlesBuf, particles.data());

    owlParamsSetGroup(lp, "world", tlasGroup);
    owlParamsSetBuffer(lp, "particles", particlesBuf);

    owlGeomSetGroup(geom, "world", tlasGroup);
    owlGeomSetBuffer(geom, "particles", particlesBuf);

    owlBuildPrograms(owl);
    owlBuildPipeline(owl);
    owlBuildSBT(owl);

    owlParamsSet3f(lp, "domain.lower",
                   domain.lower.x,
                   domain.lower.y,
                   domain.lower.z);
    owlParamsSet3f(lp, "domain.upper",
                   domain.upper.x,
                   domain.upper.y,
                   domain.upper.z);
    owlParamsSet1ui(lp, "numParticles", particles.size());
    owlParamsSet1f(lp, "radius", radius);

    owlGeomSetPrimCount(geom, particles.size());

    owlGeomSet3f(geom, "domain.lower",
                 domain.lower.x,
                 domain.lower.y,
                 domain.lower.z);
    owlGeomSet3f(geom, "domain.upper",
                 domain.upper.x,
                 domain.upper.y,
                 domain.upper.z);
    owlGeomSet1ui(geom, "numParticles", particles.size());
    owlGeomSet1f(geom, "radius", radius);

    owlGroupBuildAccel(blasGroup);
    owlGroupBuildAccel(tlasGroup);
    owlBuildSBT(owl);
  }

  void Renderer::set_dt(float dt)
  {
    owlParamsSet1f(lp, "render.dt", dt);
  }

  void Renderer::setCamera(const vec3f &org,
                           const vec3f &dir_00,
                           const vec3f &dir_du,
                           const vec3f &dir_dv)
  {
    owlParamsSet3f(lp, "camera.org", org.x, org.y, org.z);
    owlParamsSet3f(lp, "camera.dir_00", dir_00.x, dir_00.y, dir_00.z);
    owlParamsSet3f(lp, "camera.dir_du", dir_du.x, dir_du.y, dir_du.z);
    owlParamsSet3f(lp, "camera.dir_dv", dir_dv.x, dir_dv.y, dir_dv.z);

    std::cout << "Camera: " << org << " : " << dir_00 << " : " << dir_du << " : " << dir_dv << std::endl;
  }

  void Renderer::render(const vec2i &fbSize,
                        uint32_t *fbPointer)
  {
    unsigned int microsecond = 1000000;
    usleep(1 * microsecond);
    if (fbSize != this->fbSize)
    {
#ifdef DUMP_FRAMES
      owlBufferResize(fbDepth, fbSize.x * fbSize.y);
#endif
      if (!accumBuffer)
        accumBuffer = owlDeviceBufferCreate(owl, OWL_FLOAT4, 1, nullptr);
      owlBufferResize(accumBuffer, fbSize.x * fbSize.y);
      owlParamsSetBuffer(lp, "accumBuffer", accumBuffer);
      this->fbSize = fbSize;
    }
    owlParamsSetPointer(lp, "fbPointer", fbPointer);

    owlParamsSet1i(lp, "accumID", accumID);
    accumID++;
    owlParamsSet1i(lp, "render.spp", max(spp, 1));
    owlParamsSet1i(lp, "render.heatMapEnabled", heatMapEnabled);
    owlParamsSet1f(lp, "render.heatMapScale", heatMapScale);

    // owlGroupBuildAccel(blasGroup);
    // owlGroupBuildAccel(tlasGroup);

    // frameId+=1;
    camId += 1;

    std::ostringstream imgPath, depPath;
    imgPath << "/home/arun/Desktop/data/Mock_scene_setup/RGB_camera_" << frameId << "_" << std::setfill('0') << std::setw(4) << camId << ".jpg";
    depPath << "/home/arun/Desktop/data/Mock_scene_setup/Depth_camera_" << frameId << "_" << std::setfill('0') << std::setw(4) << camId << ".exr";

    cv::Mat img = Renderer::load_image(imgPath.str(), 3);
    cv::Mat dep = Renderer::load_image(depPath.str(), 1);

    Eigen::Matrix4f K, offset;
    K << 2666.666748046875 / 1920.0 * 256, 0.0, 960.0 / 1920.0 * 256, 0.0,
        0.0, 2666.666748046875 / 1080.0 * 256, 540.0 / 1080.0 * 256, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0;
    Eigen::Matrix4Xf pts = projectPix2Camera(dep, K, 0, 10);
    float radius = .01f;
    box3f domain;
    particles.resize(pts.cols());
    for (int i = 0; i < pts.cols(); ++i)
    {
      float x = pts(0, i);
      float y = pts(1, i);
      float z = pts(2, i);
      particles[i] = {x, y, z};
      domain.extend(particles[i] - vec3f(radius));
      domain.extend(particles[i] + vec3f(radius));
    }

    owlBufferResize(particlesBuf, particles.size());
    owlBufferUpload(particlesBuf, particles.data());
    // owlGroupRefitAccel(blasGroup);
    // owlGroupRefitAccel(tlasGroup);
    owlGroupBuildAccel(blasGroup);
    owlGroupBuildAccel(tlasGroup);
    owlBuildSBT(owl);
    owlLaunch2D(rayGen, fbSize.x, fbSize.y, lp);
  }

}
