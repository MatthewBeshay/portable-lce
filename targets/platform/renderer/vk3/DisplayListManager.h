#pragma once

#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "DeletionQueue.h"

namespace plce::vk3 {

struct DisplayListDraw {
    int primType = 0;
    int vertexType = 0;
    int shaderType = 0;
    std::vector<std::byte> verts;
};

struct DisplayListSubDraw {
    uint32_t vertex_offset = 0;
    uint32_t vertex_count  = 0;
    int      prim_type     = 0;
};

struct DisplayList {
    std::vector<DisplayListDraw> draws;
    std::vector<DisplayListSubDraw> gpu_draws;
    VkBuffer      vb = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    uint32_t      vb_size = 0;
    bool valid = false, uploaded = false;
};

struct PendingDestroy { VkBuffer buf; VmaAllocation alloc; };

class DisplayListManager {
public:
    void init();

    int create(int n);
    void delete_all(std::vector<PendingDestroy>& pending, std::mutex& pending_mutex);
    void start(int index);
    void clear(int index, std::vector<PendingDestroy>& pending, std::mutex& pending_mutex);
    int size(int index);
    void end();

    /// Returns pointer to uploaded display list, or nullptr if not available.
    const DisplayList* prepare(int index, DeletionQueue& deletions,
                               VmaAllocator allocator);

    void record_draw(int primType, int vertexType, const void* data, size_t bytes);

    /// Returns true if currently recording a display list.
    bool is_recording() const;

private:
    void upload(DisplayList& dl, DeletionQueue& deletions, VmaAllocator allocator);

    std::vector<DisplayList> display_lists_;
    int next_display_list_ = 1;
    mutable std::mutex display_list_mutex_;
};

}  // namespace plce::vk3
