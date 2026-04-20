#include "DisplayListManager.h"
#include "VkCheck.h"
#include "VertexFormats.h"

#include <cstring>

namespace plce::vk {

namespace {
// Thread-local display list recording state
struct RecState { int id = -1; std::vector<DisplayListDraw> draws; };
thread_local RecState t_rec;
}  // namespace

void DisplayListManager::init() {
    display_lists_.reserve(4096);
}

int DisplayListManager::create(int n) {
    std::lock_guard lk(display_list_mutex_);
    int first = next_display_list_;
    int needed = n > 0 ? n : 1;
    next_display_list_ += needed;
    if (size_t(first + needed) > display_lists_.size()) display_lists_.resize(first + needed);
    return first;
}

void DisplayListManager::delete_all(std::vector<PendingDestroy>& pending,
                                     std::mutex& pending_mutex) {
    std::lock_guard lk(display_list_mutex_);
    { std::lock_guard lk2(pending_mutex);
      for (auto& cb : display_lists_)
          if (cb.vb) pending.push_back({cb.vb, cb.alloc}); }
    display_lists_.clear();
    next_display_list_ = 1;
    t_rec.id = -1; t_rec.draws.clear();
}

void DisplayListManager::start(int index) { t_rec.id = index; t_rec.draws.clear(); }

void DisplayListManager::clear(int index, std::vector<PendingDestroy>& pending,
                                std::mutex& pending_mutex) {
    std::lock_guard lk(display_list_mutex_);
    if (index < 0 || size_t(index) >= display_lists_.size()) return;
    auto& cb = display_lists_[index];
    cb.draws.clear(); cb.gpu_draws.clear();
    if (cb.vb) {
        { std::lock_guard lk2(pending_mutex);
          pending.push_back({cb.vb, cb.alloc}); }
        cb.vb = VK_NULL_HANDLE; cb.alloc = nullptr; cb.vb_size = 0;
    }
    cb.valid = cb.uploaded = false;
}

int DisplayListManager::size(int index) {
    std::lock_guard lk(display_list_mutex_);
    if (index < 0 || size_t(index) >= display_lists_.size()) return 0;
    return display_lists_[index].valid ? 1 : 0;
}

void DisplayListManager::end() {
    int id = t_rec.id; t_rec.id = -1;
    if (id < 0) return;
    std::lock_guard lk(display_list_mutex_);
    if (size_t(id) >= display_lists_.size()) display_lists_.resize(id + 1);
    auto& cb = display_lists_[id];
    cb.draws = std::move(t_rec.draws);
    cb.valid = !cb.draws.empty();
    cb.uploaded = false;
    t_rec.draws.clear();
}

void DisplayListManager::record_draw(int primType, int vertexType,
                                      const void* data, size_t bytes) {
    DisplayListDraw d;
    d.primType = primType;
    d.vertexType = vertexType;
    d.verts.resize(bytes);
    std::memcpy(d.verts.data(), data, bytes);
    t_rec.draws.push_back(std::move(d));
}

bool DisplayListManager::is_recording() const {
    return t_rec.id >= 0;
}

DisplayListManager::Snapshot DisplayListManager::prepare(int index,
                                                           DeletionQueue& deletions,
                                                           VmaAllocator allocator) {
    std::lock_guard lk(display_list_mutex_);
    if (index < 0 || size_t(index) >= display_lists_.size()) return {};
    auto& cb = display_lists_[index];
    if (!cb.valid || cb.draws.empty()) return {};
    if (!cb.uploaded) { upload(cb, deletions, allocator); if (!cb.uploaded) return {}; }
    return {cb.vb, cb.gpu_draws};
}

void DisplayListManager::upload(DisplayList& cb, DeletionQueue& deletions,
                                 VmaAllocator allocator) {
    constexpr uint32_t kStdStride     = 32;
    constexpr uint32_t kCompactStride = 16;
    std::vector<std::byte> combined;
    cb.gpu_draws.clear();

    for (auto& d : cb.draws) {
        if (d.vertexType == 1) {
            // Compact 16-byte quads — stored natively; the vertex shader
            // decodes them and the GPU triangulates via quad_ib_.
            int vert_count = int(d.verts.size() / kCompactStride);
            if (vert_count == 0 || vert_count % 4 != 0) continue;
            uint32_t byte_off = uint32_t(combined.size());
            combined.insert(combined.end(), d.verts.begin(), d.verts.end());
            cb.gpu_draws.push_back({byte_off, uint32_t(vert_count), 0x0007, /*compact=*/true});
            continue;
        }

        int vert_count = int(d.verts.size() / kStdStride);
        if (vert_count == 0) continue;

        if (d.primType == 0x0007) {
            // Quads (32-byte) — still triangulated on CPU for now. Migrating
            // this to GPU triangulation is a separate change, since the
            // existing vertex offsets downstream assume triangle lists.
            if (vert_count % 4 != 0) continue;
            uint32_t quads = vert_count / 4;
            uint32_t byte_off = uint32_t(combined.size());
            for (uint32_t q = 0; q < quads; ++q) {
                const std::byte* b = d.verts.data() + q * 4 * kStdStride;
                auto push = [&](uint32_t i) {
                    combined.insert(combined.end(), b + i*kStdStride, b + (i+1)*kStdStride);
                };
                push(0); push(1); push(2); push(0); push(2); push(3);
            }
            cb.gpu_draws.push_back({byte_off, quads * 6, 0x0004, /*compact=*/false});
        } else if (d.primType == 0x0006) {
            // Fan -> triangles
            if (vert_count < 3) continue;
            int fan_count = vert_count;
            auto fan_data = fan_to_list(d.verts.data(), fan_count);
            if (fan_count == 0) continue;
            uint32_t byte_off = uint32_t(combined.size());
            combined.insert(combined.end(), fan_data.begin(), fan_data.end());
            cb.gpu_draws.push_back({byte_off, uint32_t(fan_count), 0x0004, /*compact=*/false});
        } else {
            uint32_t byte_off = uint32_t(combined.size());
            combined.insert(combined.end(), d.verts.begin(), d.verts.end());
            cb.gpu_draws.push_back({byte_off, uint32_t(vert_count), d.primType, /*compact=*/false});
        }
    }
    if (combined.empty()) { cb.uploaded = false; return; }

    uint32_t needed = uint32_t(combined.size());

    // Always allocate a new host-visible buffer to avoid data races with
    // the GPU reading the previous frame's data. Old buffer is deferred-
    // destroyed after the GPU is done. Direct memcpy eliminates staging
    // buffer, command buffer submission, and vkQueueWaitIdle entirely.
    if (cb.vb)
        deletions.push_buffer(allocator, cb.vb, cb.alloc);

    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size  = needed;
    bi.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
               VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info{};
    check(vmaCreateBuffer(allocator, &bi, &ai, &cb.vb, &cb.alloc, &info),
          "display list vb");
    cb.vb_size = needed;

    std::memcpy(info.pMappedData, combined.data(), needed);
    cb.uploaded = true;
}

}  // namespace plce::vk
