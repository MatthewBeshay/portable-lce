#pragma once

#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "DeletionQueue.h"

namespace plce::vk {

struct DisplayListDraw {
    int primType = 0;
    int vertexType = 0;
    int shaderType = 0;
    std::vector<std::byte> verts;
};

struct DisplayListSubDraw {
    uint32_t vertex_offset = 0;   // bytes from start of DisplayList::vb
    uint32_t vertex_count  = 0;   // vertices at the subdraw's native stride
    int      prim_type     = 0;   // GL primitive type (0x0001 lines ... 0x0007 quads)
    bool     compact       = false;  // 16-byte packed vertices when true, else 32-byte
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

    /// Snapshot of a prepared display list — safe to use after the manager's
    /// mutex is released. A null vb means the display list is not available.
    struct Snapshot {
        VkBuffer vb = VK_NULL_HANDLE;
        std::vector<DisplayListSubDraw> draws;
    };

    /// Locks the manager, uploads the display list if needed, and returns a
    /// by-value snapshot safe to use after the lock is released. Returns an
    /// empty snapshot (vb == VK_NULL_HANDLE) if the display list is unusable.
    Snapshot prepare(int index, DeletionQueue& deletions, VmaAllocator allocator);

    void record_draw(int primType, int vertexType, const void* data, size_t bytes);

    /// Returns true if currently recording a display list.
    bool is_recording() const;

private:
    void upload(DisplayList& dl, DeletionQueue& deletions, VmaAllocator allocator);

    std::vector<DisplayList> display_lists_;
    int next_display_list_ = 1;
    mutable std::mutex display_list_mutex_;
};

}  // namespace plce::vk
