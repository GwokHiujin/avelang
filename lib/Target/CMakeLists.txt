add_subdirectory(GPU)

if(WITH_AMDGPU)
  add_subdirectory(AMDGPU)
endif()

if(WITH_CUDA)
  add_subdirectory(NVVM)
endif()