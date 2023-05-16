#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
    #define MANET_INLINE __forceinline__
    #define MANET_HOST   __host__
    #define MANET_DEVICE __device__
#else
    #define MANET_INLINE inline
    #define MANET_HOST
    #define MANET_DEVICE
#endif

#define MANET_HOSTDEVICE MANET_HOST MANET_DEVICE
#define MANET_DECL MANET_INLINE MANET_HOSTDEVICE

#define MANET_PI     3.14159265358f
#define MANET_1DIVPI 0.31830988618f
#define MANET_PIDIV2 1.57079632679f
#define MANET_PIDIV4 0.78539816339f