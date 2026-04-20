// Single translation unit for VMA implementation.
// Must be compiled exactly once — all other files include vk_mem_alloc.h as header-only.
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS  1
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <vma/vk_mem_alloc.h>
