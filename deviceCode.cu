#include <iostream>
#include <sstream>

#include "deviceCode.h"
#include "owl/common/math/random.h"

namespace dvr
{
    extern "C" __constant__ LaunchParams optixLaunchParams;


    typedef owl::common::LCG<4> Random;
    
    inline  __device__
    vec3f backGroundColor()
    {
        const vec2i pixelID = owl::getLaunchIndex();
        const float t = pixelID.y / (float)optixGetLaunchDimensions().y;
        const vec3f c = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
        return c;
    }

    inline  __device__ vec4f over(const vec4f &A, const vec4f &B)
    {
        return A + (1.f-A.w)*B;
    }

    struct PRD
    {
        float tHit;
        int primID;
    };

    inline __device__
    bool intersect(const Ray &ray,
                   const box3f &box,
                   float &t0,
                   float &t1)
    {
        vec3f lo = (box.lower - ray.origin) / ray.direction;
        vec3f hi = (box.upper - ray.origin) / ray.direction;
        
        vec3f nr = min(lo,hi);
        vec3f fr = max(lo,hi);

        t0 = max(ray.tmin,reduce_max(nr));
        t1 = min(ray.tmax,reduce_min(fr));

        return t0 < t1;
    }

    inline __device__
    float firstSampleT(const range1f &rayInterval,
                       const float dt,
                       const float ils_t0)
    {
        float numSegsf = floor((rayInterval.lower - dt*ils_t0)/dt);
        float t = dt * (ils_t0 + numSegsf);
        if (t < rayInterval.lower) t += dt;
        return t;
    }

    inline  __device__ Ray generateRay(const vec2f screen)
    {
        auto &lp = optixLaunchParams;
        vec3f org = lp.camera.org;
        vec3f dir
          = lp.camera.dir_00
          + screen.u * lp.camera.dir_du
          + screen.v * lp.camera.dir_dv;
        dir = normalize(dir);
        if (fabs(dir.x) < 1e-5f) dir.x = 1e-5f;
        if (fabs(dir.y) < 1e-5f) dir.y = 1e-5f;
        if (fabs(dir.z) < 1e-5f) dir.z = 1e-5f;
        return Ray(org,dir,0.f,1e10f);
    }
 
    OPTIX_BOUNDS_PROGRAM(Particles)(const void  *geomData,
                                    box3f       &primBounds,
                                    const int    primID)
    {
        const ParticleGeom& self = *(const ParticleGeom*)geomData;
        vec3f particle = self.particles[primID];
        float radius = self.radius;
        vec3f min(particle-radius);
        vec3f max(particle+radius);
        primBounds
            = box3f()
            .including(min)
            .including(max);
    }

    OPTIX_INTERSECT_PROGRAM(Particles)()
    {
        const ParticleGeom& self = owl::getProgramData<ParticleGeom>();
        PRD &prd = owl::getPRD<PRD>();

        int primID = optixGetPrimitiveIndex();
        owl::Ray ray(optixGetObjectRayOrigin(),
                     optixGetObjectRayDirection(),
                     optixGetRayTmin(),
                     optixGetRayTmax());
#if 1
        struct {
          vec3f center;
          float radius;
        } sphere;

        sphere.center = self.particles[primID];
        sphere.radius = self.radius;

        vec3f ori = ray.origin - sphere.center;
        ray.origin = ori;

        float A = dot(ray.direction, ray.direction);
        float B = dot(ray.direction, ray.origin) * 2.f;
        float C = dot(ray.origin, ray.origin) - sphere.radius * sphere.radius;

        // solve Ax**2 + Bx + C
        float disc = B * B - 4.f * A * C;
        float valid = disc >= 0.f;

        float root_disc = valid ? sqrtf(disc) : disc;

        float q = B<0.f ? -.5f * (B-root_disc) : -.5f * (B+root_disc);

        float t1 = q / A;
        float t2 = C / q;

        bool hit = valid && (t1>=0.f || t2 >=0.f);
        float tHit = -1.f;
        tHit = t1 >= 0.f && t2 >= 0.f ? min(t1, t2) : tHit;
        tHit = t1 >= 0.f && t2 <  0.f ? t1          : tHit;
        tHit = t1 <  0.f && t2 >= 0.f ? t2          : tHit;

        if (hit && optixReportIntersection(tHit, 0)) {
            prd.tHit = tHit;
            prd.primID = primID;
        }
#endif
    }

    OPTIX_CLOSEST_HIT_PROGRAM(Particles)()
    {
        // const ParticleGeom& self = owl::getProgramData<ParticleGeom>();
        // PRD &prd = owl::getPRD<PRD>();
        // prd.particleID = optixGetPrimitiveIndex();
    }

    inline __device__ vec3f hue_to_rgb(float hue)
    {
        float s = saturate( hue ) * 6.0f;
        float r = saturate( fabsf(s - 3.f) - 1.0f );
        float g = saturate( 2.0f - fabsf(s - 2.0f) );
        float b = saturate( 2.0f - fabsf(s - 4.0f) );
        return vec3f(r, g, b); 
    }
      
    inline __device__ vec3f temperature_to_rgb(float t)
    {
        float K = 4.0f / 6.0f;
        float h = K - K * t;
        float v = .5f + 0.5f * t;    return v * hue_to_rgb(h);
    }
      
                                      
    inline __device__
    vec3f heatMap(float t)
    {
#if 1
        return temperature_to_rgb(t);
#else
        if (t < .25f) return lerp(vec3f(0.f,1.f,0.f),vec3f(0.f,1.f,1.f),(t-0.f)/.25f);
        if (t < .5f)  return lerp(vec3f(0.f,1.f,1.f),vec3f(0.f,0.f,1.f),(t-.25f)/.25f);
        if (t < .75f) return lerp(vec3f(0.f,0.f,1.f),vec3f(1.f,1.f,1.f),(t-.5f)/.25f);
        if (t < 1.f)  return lerp(vec3f(1.f,1.f,1.f),vec3f(1.f,0.f,0.f),(t-.75f)/.25f);
        return vec3f(1.f,0.f,0.f);
#endif
    }
  
    OPTIX_RAYGEN_PROGRAM(renderFrame)()
    {
        auto& lp = optixLaunchParams;
#if 1
        const int spp = lp.render.spp;
        const vec2i threadIdx = owl::getLaunchIndex();
        Ray ray = generateRay(vec2f(threadIdx)+vec2f(.5f));

        vec4f bgColor = vec4f(backGroundColor(),1.f);
        int pixelID = threadIdx.x + owl::getLaunchDims().x*threadIdx.y;
        Random random(pixelID,lp.accumID);

        uint64_t clock_begin = clock64();

        vec4f accumColor = 0.f;

        float z = 1e20f;
        for (int sampleID=0;sampleID<spp;sampleID++) {
            vec4f color = 0.f;
            PRD prd = {-1.f,-1};
            owl::traceRay(lp.world,ray,prd,
                          OPTIX_RAY_FLAG_DISABLE_ANYHIT);
            if (prd.primID >= 0) {
                vec3f baseColor(0.f,0.f,.8f);
                vec3f isectPos = ray.origin + prd.tHit*ray.direction;
                vec3f N = (isectPos - lp.particles[prd.primID]) / lp.radius;
                color = vec4f(N,1.f);
            }
            color = over(color,bgColor);
            accumColor += color;
        }
#if DUMP_FRAMES
        lp.fbPointer[pixelID] = make_rgba(accumColor);
        lp.fbDepth[pixelID]   = z;
        return;
#endif

        uint64_t clock_end = clock64();
        if (lp.render.heatMapEnabled > 0.f) {
            float t = (clock_end-clock_begin)*(lp.render.heatMapScale/spp);
            accumColor = over(vec4f(heatMap(t),.5f),accumColor);
        }

        if (lp.accumID > 0)
            accumColor += vec4f(lp.accumBuffer[pixelID]);
        lp.accumBuffer[pixelID] = accumColor;
        accumColor *= (1.f/(lp.accumID+1));
        
        bool crossHairs = (owl::getLaunchIndex().x == owl::getLaunchDims().x/2
                           ||
                           owl::getLaunchIndex().y == owl::getLaunchDims().y/2
                           );
        if (crossHairs) accumColor = vec4f(1.f) - accumColor;
        
        lp.fbPointer[pixelID] = make_rgba(vec3f(accumColor*(1.f/spp)));
#endif
        // const vec2i threadIdx = owl::getLaunchIndex();
        // int pixelID = threadIdx.x + owl::getLaunchDims().x*threadIdx.y;
        // lp.fbPointer[pixelID] = make_rgba(vec3f(1.f));
    }
}
