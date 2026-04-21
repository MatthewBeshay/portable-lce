#pragma once

#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "DeletionQueue.h"
#include "VmaResources.h"

namespace plce::vk {

struct DisplayListDraw {
    int primType = 0;
    int vertexType = 0;
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
    VmaBuffer vb;
    uint32_t  vb_size = 0;
    bool valid = false, uploaded = false;
};

class DisplayListManager {
public:
    void init();

    int create(int n);
    void delete_all(std::vector<VmaBuffer>& pending, std::mutex& pending_mutex);
    void start(int index);
    void clear(int index, std::vector<VmaBuffer>& pending, std::mutex& pending_mutex);
    int size(int index);
    void end();

    /// RAII handle returned by prepare(): keeps the manager mutex locked
    /// for its lifetime so the caller can read the DisplayList's VB and
    /// sub-draws directly without copying them out. Unlock happens at
    /// scope exit.
    class PreparedHandle {
    public:
        PreparedHandle() = default;
        PreparedHandle(std::unique_lock<std::mutex> lk, const DisplayList* dl)
            : lock_(std::move(lk)), dl_(dl) {}

        PreparedHandle(const PreparedHandle&) = delete;
        PreparedHandle& operator=(const PreparedHandle&) = delete;
        PreparedHandle(PreparedHandle&&) noexcept = default;
        PreparedHandle& operator=(PreparedHandle&&) noexcept = default;

        [[nodiscard]] explicit operator bool() const noexcept { return dl_ != nullptr; }
        [[nodiscard]] VkBuffer vb() const noexcept { return dl_ ? dl_->vb.handle() : VK_NULL_HANDLE; }
        [[nodiscard]] const std::vector<DisplayListSubDraw>& draws() const noexcept { return dl_->gpu_draws; }

    private:
        std::unique_lock<std::mutex> lock_;
        const DisplayList*           dl_ = nullptr;
    };

    /// Locks the manager, uploads the display list if needed, and returns a
    /// handle that keeps the lock for its lifetime. Dereferencing the
    /// handle is only valid while it is in scope. An empty handle
    /// (operator bool == false) means the display list is unusable.
    PreparedHandle prepare(int index, DeletionQueue& deletions, VmaAllocator allocator);

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
