#include "DisplayListManager.h"
#include "VkCheck.h"
#include "VertexFormats.h"

#include <cassert>
#include <cstring>
#include <thread>

namespace plce::vk {

namespace {
// Thread-local display list recording state. `raw_verts` is the concat
// arena for every recorded draw's vertex bytes in this recording; one
// allocation per DisplayList instead of one per draw.
//
// Thread affinity: start() / record_draw() / end() must all run on the
// same thread — the state is thread_local so a cross-thread split would
// silently lose vertex data. `tid` stamps the owning thread on start()
// and the other two functions assert against it in debug builds.
struct RecState {
    int id = -1;
    std::thread::id tid{};
    std::vector<DisplayListDraw> draws;
    std::vector<std::byte>       raw_verts;
};
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

void DisplayListManager::delete_all(std::vector<VmaBuffer>& pending,
                                     std::mutex& pending_mutex) {
    std::lock_guard lk(display_list_mutex_);
    {
        std::lock_guard lk2(pending_mutex);
        for (auto& cb : display_lists_)
            if (cb.vb) pending.push_back(std::move(cb.vb));
    }
    display_lists_.clear();
    next_display_list_ = 1;
    t_rec.id = -1; t_rec.draws.clear(); t_rec.raw_verts.clear();
}

void DisplayListManager::start(int index) {
    t_rec.id = index;
    t_rec.tid = std::this_thread::get_id();
    t_rec.draws.clear();
    t_rec.raw_verts.clear();
}

void DisplayListManager::clear(int index, std::vector<VmaBuffer>& pending,
                                std::mutex& pending_mutex) {
    std::lock_guard lk(display_list_mutex_);
    if (index < 0 || size_t(index) >= display_lists_.size()) return;
    auto& cb = display_lists_[index];
    cb.draws.clear(); cb.raw_verts.clear(); cb.gpu_draws.clear();
    if (cb.vb) {
        std::lock_guard lk2(pending_mutex);
        pending.push_back(std::move(cb.vb));
    }
    cb.vb_size = 0;
    cb.valid = cb.uploaded = false;
}

int DisplayListManager::size(int index) {
    std::shared_lock<std::shared_mutex> lk(display_list_mutex_);
    if (index < 0 || size_t(index) >= display_lists_.size()) return 0;
    return display_lists_[index].valid ? 1 : 0;
}

void DisplayListManager::end() {
    assert((t_rec.id < 0 || t_rec.tid == std::this_thread::get_id()) &&
           "DisplayListManager::end called on a thread different from start()");
    int id = t_rec.id; t_rec.id = -1;
    if (id < 0) return;
    std::lock_guard lk(display_list_mutex_);
    if (size_t(id) >= display_lists_.size()) display_lists_.resize(id + 1);
    auto& cb = display_lists_[id];
    cb.draws     = std::move(t_rec.draws);
    cb.raw_verts = std::move(t_rec.raw_verts);
    cb.valid     = !cb.draws.empty();
    cb.uploaded  = false;
    t_rec.draws.clear();
    t_rec.raw_verts.clear();
}

void DisplayListManager::record_draw(int primType, int vertexType,
                                      const void* data, size_t bytes) {
    assert(t_rec.tid == std::this_thread::get_id() &&
           "record_draw called on a thread different from start()");
    const uint32_t off = uint32_t(t_rec.raw_verts.size());
    t_rec.raw_verts.insert(t_rec.raw_verts.end(),
                           static_cast<const std::byte*>(data),
                           static_cast<const std::byte*>(data) + bytes);
    DisplayListDraw d;
    d.primType   = primType;
    d.vertexType = vertexType;
    d.byte_offset = off;
    d.byte_size   = uint32_t(bytes);
    t_rec.draws.push_back(d);
}

bool DisplayListManager::is_recording() const {
    return t_rec.id >= 0;
}

DisplayListManager::PreparedHandle DisplayListManager::prepare(
        int index, DeletionQueue& deletions, VmaAllocator allocator) {
    // Fast path: already uploaded. Multiple CBuffCall readers can hold a
    // shared lock in parallel without serialising on the mutex.
    {
        std::shared_lock<std::shared_mutex> sk(display_list_mutex_);
        if (index < 0 || size_t(index) >= display_lists_.size()) return {};
        const auto& cb = display_lists_[index];
        if (!cb.valid || cb.draws.empty()) return {};
        if (cb.uploaded) {
            return PreparedHandle(std::move(sk), &cb);
        }
    }
    // Slow path: upload needed. Take the writer lock.
    {
        std::unique_lock<std::shared_mutex> uk(display_list_mutex_);
        if (index < 0 || size_t(index) >= display_lists_.size()) return {};
        auto& cb = display_lists_[index];
        if (!cb.valid || cb.draws.empty()) return {};
        if (!cb.uploaded) {
            upload(cb, deletions, allocator);
            if (!cb.uploaded) return {};
        }
    }
    // Re-acquire shared lock for the returned handle. A writer could
    // clear() the buffer between the two locks; re-verify.
    std::shared_lock<std::shared_mutex> sk(display_list_mutex_);
    if (index < 0 || size_t(index) >= display_lists_.size()) return {};
    const auto& cb2 = display_lists_[index];
    if (!cb2.valid || !cb2.uploaded) return {};
    return PreparedHandle(std::move(sk), &cb2);
}

void DisplayListManager::upload(DisplayList& cb, DeletionQueue& deletions,
                                 VmaAllocator allocator) {
    constexpr uint32_t kStdStride     = 32;
    constexpr uint32_t kCompactStride = 16;
    std::vector<std::byte> combined;
    cb.gpu_draws.clear();

    for (const auto& d : cb.draws) {
        const std::byte* src = cb.raw_verts.data() + d.byte_offset;
        const uint32_t   sz  = d.byte_size;

        if (d.vertexType == 1) {
            // Compact 16-byte quads — stored natively; the vertex shader
            // decodes them and the GPU triangulates via quad_ib_.
            int vert_count = int(sz / kCompactStride);
            if (vert_count == 0 || vert_count % 4 != 0) continue;
            uint32_t byte_off = uint32_t(combined.size());
            combined.insert(combined.end(), src, src + sz);
            cb.gpu_draws.push_back({byte_off, uint32_t(vert_count), 0x0007, /*compact=*/true});
            continue;
        }

        int vert_count = int(sz / kStdStride);
        if (vert_count == 0) continue;

        if (d.primType == 0x0007) {
            // Quads (32-byte) — still triangulated on CPU for now. Migrating
            // this to GPU triangulation is a separate change, since the
            // existing vertex offsets downstream assume triangle lists.
            if (vert_count % 4 != 0) continue;
            uint32_t quads = vert_count / 4;
            uint32_t byte_off = uint32_t(combined.size());
            for (uint32_t q = 0; q < quads; ++q) {
                const std::byte* b = src + q * 4 * kStdStride;
                auto push = [&](uint32_t i) {
                    combined.insert(combined.end(), b + i*kStdStride, b + (i+1)*kStdStride);
                };
                push(0); push(1); push(2); push(0); push(2); push(3);
            }
            cb.gpu_draws.push_back({byte_off, quads * 6, 0x0004, /*compact=*/false});
        } else if (d.primType == 0x0006) {
            // Fan -> triangles
            if (vert_count < 3) continue;
            auto fan = fan_to_list(src, vert_count);
            if (fan.vertex_count == 0) continue;
            uint32_t byte_off = uint32_t(combined.size());
            combined.insert(combined.end(), fan.data.begin(), fan.data.end());
            cb.gpu_draws.push_back({byte_off, uint32_t(fan.vertex_count), 0x0004, /*compact=*/false});
        } else {
            uint32_t byte_off = uint32_t(combined.size());
            combined.insert(combined.end(), src, src + sz);
            cb.gpu_draws.push_back({byte_off, uint32_t(vert_count), d.primType, /*compact=*/false});
        }
    }
    if (combined.empty()) { cb.uploaded = false; return; }

    uint32_t needed = uint32_t(combined.size());

    // Always allocate a new host-visible buffer to avoid data races with
    // the GPU reading the previous frame's data. Old buffer is deferred-
    // destroyed after the GPU is done. Direct memcpy eliminates staging
    // buffer, command buffer submission, and vkQueueWaitIdle entirely.
    if (cb.vb) deletions.push_buffer(std::move(cb.vb));

    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size  = needed;
    bi.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
               VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info{};
    VkBuffer      buf   = VK_NULL_HANDLE;
    VmaAllocation a     = nullptr;
    check(vmaCreateBuffer(allocator, &bi, &ai, &buf, &a, &info),
          "display list vb");
    cb.vb      = VmaBuffer(allocator, buf, a, info.pMappedData);
    cb.vb_size = needed;

    std::memcpy(info.pMappedData, combined.data(), needed);
    // Flush is a no-op on coherent memory; required on non-coherent
    // iGPU / mobile paths where the GPU would otherwise read stale data.
    vmaFlushAllocation(allocator, cb.vb.allocation(), 0, needed);
    cb.uploaded = true;
}

}  // namespace plce::vk
