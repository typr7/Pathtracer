#pragma once

#include <cuda.h>
#include <memory>
#include <vector>

#include "ManetMacros.h"

template <typename T>
class CudaObjectView
{
public:
    CudaObjectView(): m_object_ptr(0) {};
    ~CudaObjectView() {};

    CudaObjectView(const CudaObjectView& other): m_object_ptr(other.m_object_ptr) {}
    CudaObjectView& operator=(const CudaObjectView& other) { m_object_ptr = other.m_object_ptr; }

    CudaObjectView(CudaObjectView&& other): m_object_ptr(other.m_object_ptr) { other.m_object_ptr = 0; }
    CudaObjectView& operator=(CudaObjectView&& other) {
        m_object_ptr       = other.m_object_ptr;
        other.m_object_ptr = 0;
    }

    CudaObjectView(CUdeviceptr object_ptr): m_object_ptr(object_ptr) {}

    MANET_DECL bool isValid() const noexcept { return (m_object_ptr != 0); }
    MANET_DECL operator bool() const noexcept { return isValid(); }

    MANET_DECL void setObject(CUdeviceptr object_ptr) noexcept { m_object_ptr = object_ptr; }
    MANET_DECL CUdeviceptr getPtr() const noexcept { return m_object_ptr; }

    MANET_INLINE MANET_DEVICE T* operator->() noexcept { return reinterpret_cast<T*>(m_object_ptr); }
    MANET_INLINE MANET_DEVICE const T* operator->() const noexcept { return reinterpret_cast<T*>(m_object_ptr); }

    MANET_INLINE MANET_DEVICE T& operator*() noexcept { return *reinterpret_cast<T*>(m_object_ptr); }
    MANET_INLINE MANET_DEVICE const T& operator*() const noexcept { return *reinterpret_cast<T*>(m_object_ptr); }
    
private:
    CUdeviceptr m_object_ptr;
};