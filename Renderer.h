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

#pragma once

#include <vector>
#include "deviceCode.h"
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <nlohmann/json.hpp>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

using json = nlohmann::json;

namespace dvr {

  struct Renderer {
    Renderer();

    void setCamera(const vec3f &org,
                   const vec3f &dir_00,
                   const vec3f &dir_du,
                   const vec3f &dir_dv);
    void render(const vec2i &fbSize,
                uint32_t *fbPointer);
    /*! set delta integration step for volume ray marching; _has_ to be
     *  called at least once before rendering a frame */
    void set_dt(float dt);

    OWLParams  lp;
    OWLRayGen  rayGen;
    OWLContext owl;
    OWLModule  module;
    int frameId;
    int camId;
    
    OWLBuffer particlesBuf { 0 };
    OWLBuffer colorsBuf { 0 };
    OWLGeomType geomType { 0 };
    OWLGeom geom;
    OWLGroup blasGroup;
    OWLGroup tlasGroup;

    std::vector<Particle> particles;
    std::vector<Color> colors;

    OWLGeomType triangleGeomType;
    OWLGroup modelGroup;

    box3f modelBounds;
    json annoData;

    bool showBoxes = 0;
    std::vector<vec4f> colorMap;
    OWLBuffer accumBuffer { 0 };
    int accumID { 0 };

    void resetAccum() { accumID = 0; }
    vec3f initCamLoc;
    Eigen :: Matrix3f initCamRotMat;
    float Camfovy;

    cv::Mat load_image(std::string filename, int c);
    void get_cam_specs(int cId, Eigen::Matrix4f& k, Eigen::Matrix4f& p, float& fovy);
    std::string get_nearest_camera(const vec3f &org);
    
#ifdef DUMP_FRAMES
    // to allow dumping rgba and depth for some unrelated compositing work....
    OWLBuffer  fbDepth;
#endif
    vec2i      fbSize { 1 };
    /*! for profiling only - always rebuild data structure, even if
        nothing changed... */
    bool alwaysRebuild = false;
    /*! domain of scalar field values over which the transfer
        function's color map is defines; values outside this domain
        will always map to (black,0.f) */
    range1f xfDomain;

    int spaceSkipMode = 0;
    
    static int   spp;
    static bool  heatMapEnabled;
    static float heatMapScale;
  };
  
} // ::dvr
