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

#include "Renderer.h"
#include "owlViewer/OWLViewer.h"
#include <fstream>

namespace dvr {
  using owl::viewer::SimpleCamera;

  const int XF_ALPHA_COUNT = 128;
  
  struct {
    bool  showBoxes = 0;
    vec3i dims = 0;
    vec3i subBrickID   = 0;
    int   subBrickSize = 0;
    std::string xfFileName = "";
    std::string formatString = "";
    std::string outFileName = "owlDVR.png";
    struct {
      vec3f vp = vec3f(0.f);
      vec3f vu = vec3f(0.f);
      vec3f vi = vec3f(0.f);
      float fovy = 50;
    } camera;
    float dt = .5f;
    vec2i windowSize  = vec2i(1024,1024);
  } cmdline;
  
  void usage(const std::string &err)
  {
    if (err != "")
      std::cout << OWL_TERMINAL_RED << "\nFatal error: " << err
                << OWL_TERMINAL_DEFAULT << std::endl << std::endl;

    std::cout << "Usage: ./owlDVR volumeFile.raw -dims x y z -f|--format float|byte" << std::endl;
    std::cout << std::endl;
    exit(1);
  }
  
  struct Viewer : public owl::viewer::OWLViewer {
  public:
    typedef owl::viewer::OWLViewer inherited;
    
    Viewer(Renderer *renderer)
      : inherited("owlDVR Sample Viewer", cmdline.windowSize),
        renderer(renderer)
    {
      renderer->set_dt(cmdline.dt);
    }
    
    /*! this function gets called whenever the viewer widget changes
      camera settings */
    void cameraChanged() override;
    void resize(const vec2i &newSize) override;
    /*! gets called whenever the viewer needs us to re-render out widget */
    void render() override;

    /*! this gets called when the user presses a key on the keyboard ... */
    void key(char key, const vec2i &where) override
    {
      inherited::key(key,where);
      renderer->resetAccum();
      switch (key) {
      case '/':
      case '?':
        renderer->spaceSkipMode++;
        break;
      case '!':
        std::cout << "saving screenshot to 'owlDVR.png'" << std::endl;
        screenShot("owlDVR.png");
        break;
      case 'R':
        renderer->alwaysRebuild = !renderer->alwaysRebuild;
        if (renderer->alwaysRebuild)
          std::cout << "rebuild: ALWAYS!" << std::endl;
        else
          std::cout << "rebuild: only when required..." << std::endl;
        break;
      case 'H':
        renderer->heatMapEnabled = !renderer->heatMapEnabled;
        break;
      case '<':
        renderer->heatMapScale /= 1.5f;
        break;
      case '>':
        renderer->heatMapScale *= 1.5f;
        break;
      case ')':
        renderer->spp++;
        PRINT(renderer->spp);
        break;
      case '(':
        renderer->spp = max(1,renderer->spp-1);
        PRINT(renderer->spp);
        break;
      }
    }
    
    
  public:

    Renderer *const renderer;
  };
  

  void Viewer::resize(const vec2i &newSize) 
  {
    // ... tell parent to resize (also resizes the pbo in the wingdow)
    inherited::resize(newSize);
    cameraChanged();
    renderer->resetAccum();
  }
    
  /*! this function gets called whenever the viewer widget changes
    camera settings */
  void Viewer::cameraChanged() 
  {
    inherited::cameraChanged();
    const SimpleCamera &camera = inherited::getCamera();
    
    const vec3f screen_du = camera.screen.horizontal / float(getWindowSize().x);
    const vec3f screen_dv = camera.screen.vertical   / float(getWindowSize().y);
    const vec3f screen_00 = camera.screen.lower_left;
    renderer->setCamera(camera.lens.center,screen_00,screen_du,screen_dv);
    renderer->resetAccum();
  }
    

  /*! gets called whenever the viewer needs us to re-render out widget */
  void Viewer::render() 
  {
    static double t_last = getCurrentTime();
    static double t_first = t_last;

    renderer->render(fbSize,fbPointer);
      
    double t_now = getCurrentTime();
    static double avg_t = t_now-t_last;
    // if (t_last >= 0)
    avg_t = 0.8*avg_t + 0.2*(t_now-t_last);

    char title[1000];
    sprintf(title,"mowlana - %.2f FPS",(1.f/avg_t));
    setTitle(title);
    t_last = t_now;


#ifdef DUMP_FRAMES
    // just dump the 10th frame, then hard-exit
    static int g_frameID = 0;
    if (g_frameID++ >= 10) {
      const float *fbDepth
        = (const float *)owlBufferGetPointer(renderer->fbDepth,0);
      std::ofstream out(cmdline.outFileName+".rgbaz",std::ios::binary);
      out.write((char*)&fbSize,sizeof(fbSize));
      out.write((char*)fbPointer,fbSize.x*fbSize.y*sizeof(*fbPointer));
      out.write((char*)fbDepth,fbSize.x*fbSize.y*sizeof(*fbDepth));
      // for (int i=0;i<fbSize.x*fbSize.y;i++) {
      //   if (fbDepth[i] < 1e10f && fbDepth[i] > 0.f)
      //     PRINT(fbDepth[i]);
      // }
      screenShot(cmdline.outFileName+".png");
      exit(0);
    }
#endif
  }

  extern "C" int main(int argc, char **argv)
  {
    std::string inFileName;
    bool useLargeModel = false;
    bool inferDepth = false;
    
    for (int i=1;i<argc;i++) {
      const std::string arg = argv[i];
      if (arg[0] != '-') {
        inFileName = arg;
      }
      else if (arg == "-xf") {
        cmdline.xfFileName = argv[++i];
      }
      else if (arg == "-fovy") {
        cmdline.camera.fovy = std::stof(argv[++i]);
      }
      else if (arg == "-win") {
        cmdline.windowSize.x = std::stoi(argv[++i]);
        cmdline.windowSize.y = std::stoi(argv[++i]);
      }
      else if (arg == "--camera") {
        cmdline.camera.vp.x = std::stof(argv[++i]);
        cmdline.camera.vp.y = std::stof(argv[++i]);
        cmdline.camera.vp.z = std::stof(argv[++i]);
        cmdline.camera.vi.x = std::stof(argv[++i]);
        cmdline.camera.vi.y = std::stof(argv[++i]);
        cmdline.camera.vi.z = std::stof(argv[++i]);
        cmdline.camera.vu.x = std::stof(argv[++i]);
        cmdline.camera.vu.y = std::stof(argv[++i]);
        cmdline.camera.vu.z = std::stof(argv[++i]);
      }
      else if (arg == "-win"  || arg == "--win" || arg == "--size") {
        cmdline.windowSize.x = std::atoi(argv[++i]);
        cmdline.windowSize.y = std::atoi(argv[++i]);
      }
      else if (arg == "-o") {
        cmdline.outFileName = argv[++i];
      }
      else if (arg == "-f" || arg == "--format" || arg == "-t") {
        cmdline.formatString = argv[++i];
      }
      else if (arg == "-d" || arg == "--dims" || arg == "-dims") {
        cmdline.dims.x = std::stoi(argv[++i]);
        cmdline.dims.y = std::stoi(argv[++i]);
        cmdline.dims.z = std::stoi(argv[++i]);
      }
      else if (arg == "-spp" || arg == "--spp") {
        Renderer::spp = std::stoi(argv[++i]);
      }
      else if (arg == "--heat-map") {
        Renderer::heatMapEnabled = true;
        Renderer::heatMapScale = std::stof(argv[++i]);
      }
      else if (arg == "-dt") {
        cmdline.dt = std::stof(argv[++i]);
      }
      else if (arg == "--show-boxes") {
        cmdline.showBoxes = true;
      }
      else if (arg == "--sub-brick") {
        cmdline.subBrickID.x = std::stoi(argv[++i]);
        cmdline.subBrickID.y = std::stoi(argv[++i]);
        cmdline.subBrickID.z = std::stoi(argv[++i]);
        cmdline.subBrickSize = std::stoi(argv[++i]);
      }
      else if (arg == "--infer-depth-s") {
        inferDepth = true;
        useLargeModel = false;
      }
      else if (arg == "--infer-depth-l") {
        inferDepth = true;
        useLargeModel = true;
      }
      else
        usage("unknown cmdline arg '"+arg+"'");
    }
    
    Renderer renderer(inferDepth, useLargeModel);//(model);

    const box3f modelBounds = renderer.modelBounds;
    Viewer *viewer = new Viewer(&renderer);

    viewer->enableFlyMode();
    viewer->enableInspectMode(/* valid range of poi*/modelBounds,
                              /* min distance      */1e-3f,
                              /* max distance      */1e8f);

    if (cmdline.camera.vu != vec3f(0.f)) {
      viewer->setCameraOrientation(/*origin   */cmdline.camera.vp,
                                   /*lookat   */cmdline.camera.vi,
                                   /*up-vector*/cmdline.camera.vu,
                                   /*fovy(deg)*/cmdline.camera.fovy);
    } else {
      Eigen::Vector3f oriCam= renderer.initCamRotMat*Eigen::Vector3f(0.f, 1.f, 0.f);
      Eigen::Vector3f oriCamZ= renderer.initCamRotMat*Eigen::Vector3f(0.f, 0.f, 1.f);
      viewer->setCameraOrientation(/*origin   */
                                    // modelBounds.center()
                                    renderer.initCamLoc,
                                  //  + vec3f(-0.005f, +0.f, 0.0125f) * modelBounds.span(), // -4.4
                                   /*lookat   */vec3f(oriCamZ(0), oriCamZ(1), oriCamZ(2)), //modelBounds.center(),
                                   /*up-vector*/vec3f(oriCam(0), oriCam(1), oriCam(2)),
                                   /*fovy(deg)*/50.f);
    }
    // viewer->setWorldScale(1.0f*length(modelBounds.span()));
    viewer->showAndRun();
  }
  
}

