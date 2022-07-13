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
#include <sstream>
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

extern "C" char embedded_deviceCode[];

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

  std::string Renderer::get_nearest_camera(const vec3f &org) {
    std::map<std::string, float> Ori;
    std::map<std::string, float> Loc;
    double dist = INFINITY;
    std::string cId;
    double norm;
    for (json::iterator it = annoData.begin(); it != annoData.end(); ++it)
    {
        std::map<std::string, float> _Loc = it.value()["location"].get<std::map<std::string, float>>();
        Eigen::Vector3d loc_vec = {_Loc["x"]-org[0], _Loc["y"]-org[1], _Loc["z"]-org[2]};
        norm = loc_vec.squaredNorm();
        if (norm <= dist) {
          cId = it.key();
          dist = norm;
          Loc = _Loc;
        }
    }
    std::cout << "Selected Camera : " << cId << " with norm of " << dist << std::endl;
    return cId;
  }

  void Renderer::get_cam_specs(int cId, Eigen::Matrix4f& k, Eigen::Matrix4f& p, float& fovy) {
    std::array<std::array<float, 3>, 3> Int;
    std::map<std::string, float> Ori;
    std::map<std::string, float> Loc;
    std::ostringstream camName;
    camName << "camera_" << cId;
    for (json::iterator it = annoData[camName.str()].begin(); it != annoData[camName.str()].end(); ++it)
    {
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
      if (it.key() == "angle_y") 
      {
        fovy = it.value().get<float>();
      }
    }
    Eigen::Quaternionf q;
    q.x() = Ori["x"];
    q.y() = Ori["y"];
    q.z() = Ori["z"];
    q.w() = Ori["w"];
    p = Eigen :: Matrix4f :: Identity ();
    p.topLeftCorner(3,3) = q.normalized().toRotationMatrix();
    p.topRightCorner(3,1) <<  Loc["x"] , Loc["y"] , Loc["z"];

    Eigen::Matrix4f K, offset;
    int res = 1;
    K << Int[0][0] / 1920.0 * res, Int[0][1] / 1920.0 * res, Int[0][2] / 1920.0 * res, 0.0,
        Int[1][0] / 1080.0 * res, Int[1][1] / 1080.0 * res, Int[1][2] / 1080.0 * res, 0.0,
        Int[2][0], Int[2][1], Int[2][2], 0.0,
        0.0, 0.0, 0.0, 1.0;
    offset << 2, 0, -1, 0,
              0, 2, -1, 0,
              0, 0, 1, 0,
              0, 0, 0, 1;
    k = offset*K;
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
      {"colors", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, colors)},
      {nullptr /* sentinel to mark end of list */}};

  Renderer::Renderer()
      : xfDomain({0.f, 1.f})
  {
#if 1
    std::default_random_engine rd;
    std::uniform_real_distribution<float> pos(-1.0f, 1.0f);

    float radius = 0.01f;
    box3f domain;
    std::ifstream annoFile("/home/arun/Desktop/data/Mock_scene_setup/Mock_scene_annotation.json", std::ifstream::binary);
    annoFile >> annoData;
    std::cout << annoData["camera_2"];

    frameId = 1;
    camId = 1;
    std::ostringstream imgPath, depPath;
    imgPath << "/home/arun/Desktop/data/Mock_scene_setup/RGB_camera_" << camId << "_" << std::setfill('0') << std::setw(4) << frameId << ".jpg";
    depPath << "/home/arun/Desktop/data/Mock_scene_setup/Depth_camera_" << camId << "_" << std::setfill('0') << std::setw(4) << frameId << ".exr";
    cv::Mat img = Renderer::load_image(imgPath.str(), 3);
    cv::Mat dep = Renderer::load_image(depPath.str(), 1);
    Eigen::Matrix4f K;
    Eigen::Matrix4f SrcPoseMat;
    get_cam_specs(camId, K, SrcPoseMat, Camfovy);
    Eigen::Vector3f camLoc= SrcPoseMat.topRightCorner(3,1);
    Camfovy = Camfovy*180.0/M_PI;
    initCamRotMat = SrcPoseMat.topLeftCorner(3,3);
    initCamLoc = vec3f(camLoc(0), camLoc(1), camLoc(2));
    std::cout << "Source Pose matrix: " << SrcPoseMat << std::endl;
    std::cout << "Intrinsic matrix: " << K << std::endl;
    std::cout << "Initial location of camera: " << initCamLoc << std::endl;
    std::cout << "Initial fov-y of camera: " << Camfovy << std::endl;
    Eigen::MatrixXi mask;

    Eigen::Matrix4Xf pts = projectCam2World(projectPix2Camera(dep, K, 0.01, 10, mask), SrcPoseMat);
    // Eigen::Matrix4Xf pts = projectPix2Camera(dep, K, 0.01, 10);
    // Eigen::Matrix4Xf pts = projectPix2Camera2(dep, K, 5, 10);

    particles.resize(pts.cols());
    colors.resize(pts.cols());
    float maxx = -INFINITY;
    float maxy = -INFINITY;
    float maxz = -INFINITY;
    float minx = INFINITY;
    float miny = INFINITY;
    float minz = INFINITY;
    int im_row, im_col, temp;
    int chns = img.channels();
    const int nCols = img.cols;
    uint8_t* pixPtr = (uint8_t*)img.data;
    for (int i = 0; i < pts.cols(); ++i)
    {
      im_col = i % dep.cols;
      im_row = i / dep.cols;
      float x = pts(0, i);
      float y = pts(1, i);
      float z = pts(2, i);
      particles[i] = {x, y, z};
      temp = (im_row*nCols*chns) + im_col*chns;
      colors[i] = {(float) pixPtr[temp + 2], (float) pixPtr[temp + 1], (float) pixPtr[temp + 0]};
      colors[i] /= 255.0;
      if (x < minx) minx = x;
      if (y < miny) miny = y;
      if (z < minz) minz = z;
      if (x > maxx) maxx = x;
      if (y > maxy) maxy = y;
      if (z > maxz) maxz = z;
      if (mask(0, i) == 1) {
        domain.extend(particles[i] - vec3f(radius));
        domain.extend(particles[i] + vec3f(radius));
      }
    }
    // std::cout << "x range is " << minx << " : " << maxx << std::endl;
    // std::cout << "y range is " << miny << " : " << maxy << std::endl;
    // std::cout << "z range is " << minz << " : " << maxz << std::endl;
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
    colorsBuf = owlDeviceBufferCreate(owl,
                                      OWL_USER_TYPE(Color),
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

    owlBufferResize(colorsBuf, colors.size());
    owlBufferUpload(colorsBuf, colors.data());

    owlParamsSetGroup(lp, "world", tlasGroup);
    owlParamsSetBuffer(lp, "particles", particlesBuf);
    owlParamsSetBuffer(lp, "colors", colorsBuf);

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

    std::stringstream cameraName(get_nearest_camera(org));
    std::string seg;
    std::vector<std::string> substringlist;
    while(std::getline(cameraName, seg, '_'))
    {
      substringlist.push_back(seg);
    }
    camId = std::stoi(substringlist[1]);
  }

  void Renderer::render(const vec2i &fbSize,
                        uint32_t *fbPointer)
  {
    // unsigned int microsecond = 1000000;
    // usleep(1 * microsecond);
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

    frameId+=1;

    std::ostringstream imgPath, depPath;
    imgPath << "/home/arun/Desktop/data/Mock_scene_setup/RGB_camera_" << camId << "_" << std::setfill('0') << std::setw(4) << frameId << ".jpg";
    depPath << "/home/arun/Desktop/data/Mock_scene_setup/Depth_camera_" << camId << "_" << std::setfill('0') << std::setw(4) << frameId << ".exr";

    cv::Mat img = Renderer::load_image(imgPath.str(), 3);
    cv::Mat dep = Renderer::load_image(depPath.str(), 1);

    Eigen::Matrix4f K;
    Eigen::Matrix4f SrcPoseMat;
    get_cam_specs(camId, K, SrcPoseMat, Camfovy);
    Camfovy = Camfovy*180.0/M_PI;
    initCamRotMat = SrcPoseMat.topLeftCorner(3,3);
    Eigen::Vector3f camLoc= SrcPoseMat.topRightCorner(3,1);
    initCamLoc = vec3f(camLoc(0), camLoc(1), camLoc(2));
    // std::cout << "Source Pose matrix: " << SrcPoseMat.inverse() << std::endl;
    // std::cout << "Intrinsic matrix: " << K << std::endl;
    Eigen::MatrixXi mask;
    Eigen::Matrix4Xf pts = projectCam2World(projectPix2Camera(dep, K, 0.01, 10, mask), SrcPoseMat);
    // Eigen::Matrix4Xf pts = projectPix2Camera(dep, K, 0.01, 10);
    // Eigen::Matrix4Xf pts = projectPix2Camera2(dep, K, 5, 10);
    // std::cout << "points after projection" << pts.transpose() << std::endl;

    float radius = 0.01f;
    box3f domain;
    Renderer::resetAccum();
    particles.resize(pts.cols());
    colors.resize(pts.cols());
    int im_row, im_col, temp;
    int chns = img.channels();
    const int nCols = img.cols;
    uint8_t* pixPtr = (uint8_t*)img.data;
    for (int i = 0; i < pts.cols(); ++i)
    {
      im_col = i % dep.cols;
      im_row = i / dep.cols;
      float x = pts(0, i);
      float y = pts(1, i);
      float z = pts(2, i);
      particles[i] = {x, y, z};
      temp = (im_row*nCols*chns) + im_col*chns;
      colors[i] = {(float) pixPtr[temp + 2], (float) pixPtr[temp + 1], (float) pixPtr[temp + 0]};
      colors[i] /= 255.0;
      if (mask(0, i) == 1) {
        domain.extend(particles[i] - vec3f(radius));
        domain.extend(particles[i] + vec3f(radius));
      }
    }

    owlBufferResize(particlesBuf, particles.size());
    owlBufferUpload(particlesBuf, particles.data());

    owlBufferResize(colorsBuf, colors.size());
    owlBufferUpload(colorsBuf, colors.data());


    owlGroupRefitAccel(blasGroup);
    owlGroupRefitAccel(tlasGroup);
    // owlGroupBuildAccel(blasGroup);
    // owlGroupBuildAccel(tlasGroup);
    // owlBuildSBT(owl);
    owlLaunch2D(rayGen, fbSize.x, fbSize.y, lp);
  }

}
