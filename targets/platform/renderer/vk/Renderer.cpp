#include "Renderer.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <stdexcept>
#include <string>

#include "FrameUBO.h"
#include "VkCheck.h"
#include "VertexFormats.h"
#include "platform/fs/fs.h"

#include "vk/shaders/basic.vert.spv.h"          // kBasicVertSpv
#include "vk/shaders/basic.frag.spv.h"          // kBasicFragSpv
#include "vk/shaders/basic_compact.vert.spv.h"  // kBasicCompactVertSpv

namespace plce::vk {

namespace {
// Pipeline cache path under the platform user-data directory. Goes through
// IPlatformFilesystem so it respects the same location game saves and config use.
std::filesystem::path pipeline_cache_path() {
    return PlatformFilesystem.getUserDataPath() / "pipeline_cache.bin";
}
}

// ===================================================================
// LIFECYCLE
// ===================================================================

namespace {
// Validation layers default to debug-on / release-off, but PLCE_VK_VALIDATE=1
// forces them on so users can collect validation logs from Release builds.
bool want_validation() {
#ifndef NDEBUG
    return true;
#else
    const char* v = std::getenv("PLCE_VK_VALIDATE");
    return v && v[0] == '1';
#endif
}
}

Renderer::Renderer(SDL_Window* window)
    : dev_({window, want_validation()}),
      window_(window) {
#ifdef ENABLE_VSYNC
    swap_.create(dev_, 0, 0, VK_PRESENT_MODE_FIFO_KHR);
#else
    swap_.create(dev_, 0, 0, VK_PRESENT_MODE_MAILBOX_KHR);
#endif
    rebuild_viewport_rects();

    for (auto& f : frames_)
        f.create(dev_.handle(), dev_.allocator(), dev_.queue_family(),
                 dev_.timestamp_period_ns());

    // Immutable sampler table — 4 combinations covering every filter/wrap
    // combo the game emits through StateSetTextureFilter/Wrap. Layout:
    //   [0] nearest + repeat        (chunk terrain, most mobs)
    //   [1] nearest + clamp_to_edge (edge tiles, some GUI)
    //   [2] linear  + repeat        (blur mask, panorama)
    //   [3] linear  + clamp_to_edge (lightmap, UI smoothed icons)
    // Index 3 doubles as the lightmap sampler. Every texture slot picks
    // one index via TextureManager; the shader reads the index from
    // PushConstants::flags bits [30:31].
    {
        auto make = [&](::vk::Filter f, ::vk::SamplerAddressMode a,
                        float max_lod) {
            ::vk::SamplerCreateInfo sci;
            sci.magFilter    = f;
            sci.minFilter    = f;
            sci.mipmapMode   = (f == ::vk::Filter::eLinear)
                                   ? ::vk::SamplerMipmapMode::eNearest
                                   : ::vk::SamplerMipmapMode::eLinear;
            sci.addressModeU = a;
            sci.addressModeV = a;
            sci.addressModeW = a;
            sci.minLod       = 0.0f;
            sci.maxLod       = max_lod;
            return ::vk::raii::Sampler(dev_.vk_device(), sci);
        };
        samplers_[0] = make(::vk::Filter::eNearest,
                            ::vk::SamplerAddressMode::eRepeat,
                            VK_LOD_CLAMP_NONE);
        samplers_[1] = make(::vk::Filter::eNearest,
                            ::vk::SamplerAddressMode::eClampToEdge,
                            VK_LOD_CLAMP_NONE);
        samplers_[2] = make(::vk::Filter::eLinear,
                            ::vk::SamplerAddressMode::eRepeat,
                            VK_LOD_CLAMP_NONE);
        samplers_[3] = make(::vk::Filter::eLinear,
                            ::vk::SamplerAddressMode::eClampToEdge,
                            0.25f);  // lightmap: single mip level
    }

    // Bindless descriptor set layout:
    //   binding 0: SAMPLED_IMAGE[kMaxTextures]  (update-after-bind, partially-bound)
    //   binding 1: SAMPLER[4]                   (immutable, 4 combos above)
    {
        VkSampler immutable[4] = {*samplers_[0], *samplers_[1],
                                  *samplers_[2], *samplers_[3]};
        VkDescriptorSetLayoutBinding bindings[2]{};
        bindings[0].binding         = 0;
        bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        bindings[0].descriptorCount = TextureManager::kMaxTextures;
        bindings[0].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[1].binding         = 1;
        bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLER;
        bindings[1].descriptorCount = 4;
        bindings[1].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[1].pImmutableSamplers = immutable;

        VkDescriptorBindingFlags flags[2] = {
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
            VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT,
            0
        };
        VkDescriptorSetLayoutBindingFlagsCreateInfo bf{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
        bf.bindingCount  = 2;
        bf.pBindingFlags = flags;

        VkDescriptorSetLayoutCreateInfo ci{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.pNext        = &bf;
        ci.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        ci.bindingCount = 2;
        ci.pBindings    = bindings;
        bindless_set_layout_ = ::vk::raii::DescriptorSetLayout(
            dev_.vk_device(), ::vk::DescriptorSetLayoutCreateInfo(ci));
    }

    // Descriptor pool for the one bindless set.
    {
        VkDescriptorPoolSize ps[2]{};
        ps[0].type            = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        ps[0].descriptorCount = TextureManager::kMaxTextures;
        ps[1].type            = VK_DESCRIPTOR_TYPE_SAMPLER;
        ps[1].descriptorCount = 4;
        VkDescriptorPoolCreateInfo ci{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ci.flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        ci.maxSets       = 1;
        ci.poolSizeCount = 2;
        ci.pPoolSizes    = ps;
        bindless_pool_ = ::vk::raii::DescriptorPool(
            dev_.vk_device(), ::vk::DescriptorPoolCreateInfo(ci));
    }

    // Allocate the one bindless set. The set itself is owned by the pool
    // (no FREE_DESCRIPTOR_SET flag), so we keep the raw handle and let
    // the pool's destructor reclaim it.
    {
        VkDescriptorSetLayout layout_raw = *bindless_set_layout_;
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool     = *bindless_pool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &layout_raw;
        check(vkAllocateDescriptorSets(dev_.handle(), &ai, &bindless_set_),
              "bindless desc set");
    }

    // Per-frame UBO descriptor set layout (set = 1, binding 0, UNIFORM_BUFFER,
    // visible to both vertex and fragment stages).
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding         = 0;
        b.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        b.descriptorCount = 1;
        b.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo ci{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = 1;
        ci.pBindings    = &b;
        frame_ubo_layout_ = ::vk::raii::DescriptorSetLayout(
            dev_.vk_device(), ::vk::DescriptorSetLayoutCreateInfo(ci));
    }

    // Pool for the per-frame UBO sets (one per FrameContext).
    {
        VkDescriptorPoolSize ps{};
        ps.type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        ps.descriptorCount = kFramesInFlight;
        VkDescriptorPoolCreateInfo ci{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ci.maxSets       = kFramesInFlight;
        ci.poolSizeCount = 1;
        ci.pPoolSizes    = &ps;
        frame_ubo_pool_ = ::vk::raii::DescriptorPool(
            dev_.vk_device(), ::vk::DescriptorPoolCreateInfo(ci));
    }

    // Allocate one frame-UBO set per FrameContext and wire each to that
    // frame's host-visible UBO buffer. The binding never changes after
    // this — only the buffer's CPU-mapped contents get updated per frame.
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        VkDescriptorSetLayout layout_raw = *frame_ubo_layout_;
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool     = *frame_ubo_pool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &layout_raw;
        check(vkAllocateDescriptorSets(dev_.handle(), &ai,
                                       &frames_[i].frame_ubo_set),
              "frame ubo desc set");

        VkDescriptorBufferInfo bi{};
        bi.buffer = frames_[i].frame_ubo_buf();
        bi.offset = 0;
        bi.range  = sizeof(FrameUBO);
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet          = frames_[i].frame_ubo_set;
        w.dstBinding      = 0;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        w.pBufferInfo     = &bi;
        vkUpdateDescriptorSets(dev_.handle(), 1, &w, 0, nullptr);
    }

    // Pipeline layout: set 0 bindless images/samplers, set 1 per-frame UBO,
    // 176-byte push constant range shared by vertex + fragment.
    {
        VkDescriptorSetLayout layouts[2] = {
            *bindless_set_layout_,
            *frame_ubo_layout_,
        };
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pc.offset = 0;
        pc.size   = sizeof(PushConstants);
        VkPipelineLayoutCreateInfo ci{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        ci.setLayoutCount         = 2;
        ci.pSetLayouts            = layouts;
        ci.pushConstantRangeCount = 1;
        ci.pPushConstantRanges    = &pc;
        pipeline_layout_ = ::vk::raii::PipelineLayout(
            dev_.vk_device(), ::vk::PipelineLayoutCreateInfo(ci));
    }

    // Pipeline cache + warm-up
    {
        PipelineCache::Config pc{};
        pc.device       = dev_.handle();
        pc.layout       = *pipeline_layout_;
        pc.color_format = swap_.format();
        pc.depth_format   = swap_.depth_format();
        pc.stencil_format = swap_.has_stencil() ? swap_.depth_format()
                                                : VK_FORMAT_UNDEFINED;
        pc.vert_spv          = kBasicVertSpv;
        pc.vert_size         = sizeof(kBasicVertSpv);
        pc.frag_spv          = kBasicFragSpv;
        pc.frag_size         = sizeof(kBasicFragSpv);
        pc.vert_compact_spv  = kBasicCompactVertSpv;
        pc.vert_compact_size = sizeof(kBasicCompactVertSpv);
        pipelines_.init(pc);
        // Feed the on-disk cache blob (if any) through the platform filesystem.
        const auto path = pipeline_cache_path();
        std::vector<std::uint8_t> blob;
        if (PlatformFilesystem.exists(path))
            blob = PlatformFilesystem.readFileToVec(path);
        pipelines_.load_cache(blob);
        pipelines_.warm_up();
    }

    // Quad index buffer
    {
        std::vector<uint32_t> indices(kMaxQuads * 6);
        for (uint32_t q = 0; q < kMaxQuads; ++q) {
            uint32_t b = q * 4;
            indices[q*6+0] = b;   indices[q*6+1] = b+1; indices[q*6+2] = b+2;
            indices[q*6+3] = b;   indices[q*6+4] = b+2; indices[q*6+5] = b+3;
        }
        VkDeviceSize bytes = indices.size() * sizeof(uint32_t);
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size  = bytes;
        bi.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO;
        VkBuffer      quad_ib_raw   = VK_NULL_HANDLE;
        VmaAllocation quad_ib_alloc = nullptr;
        check(vmaCreateBuffer(dev_.allocator(), &bi, &ai, &quad_ib_raw,
                              &quad_ib_alloc, nullptr), "quad ib");
        quad_ib_ = VmaBuffer(dev_.allocator(), quad_ib_raw, quad_ib_alloc);

        VkBuffer stg = VK_NULL_HANDLE; VmaAllocation sa = nullptr;
        VmaAllocationInfo si{};
        VkBufferCreateInfo sci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        sci.size = bytes; sci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo sai{};
        sai.usage = VMA_MEMORY_USAGE_AUTO;
        sai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        check(vmaCreateBuffer(dev_.allocator(), &sci, &sai, &stg, &sa, &si), "quad ib staging");
        std::memcpy(si.pMappedData, indices.data(), bytes);
        vmaFlushAllocation(dev_.allocator(), sa, 0, bytes);

        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pci.queueFamilyIndex = dev_.queue_family();
        VkCommandPool pool; check(vkCreateCommandPool(dev_.handle(), &pci, nullptr, &pool), "quad ib pool");
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        VkCommandBuffer cmd; check(vkAllocateCommandBuffers(dev_.handle(), &cai, &cmd), "quad ib cmd");
        VkCommandBufferBeginInfo bbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bbi);
        VkBufferCopy rgn{0, 0, bytes};
        vkCmdCopyBuffer(cmd, stg, quad_ib_.handle(), 1, &rgn);
        vkEndCommandBuffer(cmd);
        VkCommandBufferSubmitInfo csi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        csi.commandBuffer = cmd;
        VkSubmitInfo2 sub{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        sub.commandBufferInfoCount = 1; sub.pCommandBufferInfos = &csi;
        check(dev_.submit2(1, &sub, VK_NULL_HANDLE), "quad ib submit");
        vkQueueWaitIdle(dev_.queue());
        vmaDestroyBuffer(dev_.allocator(), stg, sa);
        vkDestroyCommandPool(dev_.handle(), pool, nullptr);
    }

    tex_mgr_.init(dev_, bindless_set_);
    dl_mgr_.init();

#ifndef NDEBUG
    fn_begin_label_ = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
        vkGetDeviceProcAddr(dev_.handle(), "vkCmdBeginDebugUtilsLabelEXT"));
    fn_end_label_ = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
        vkGetDeviceProcAddr(dev_.handle(), "vkCmdEndDebugUtilsLabelEXT"));
#endif

    {
        int w, h;
        SDL_GetWindowSize(window_, &w, &h);
        fb_.width  = uint32_t(w);
        fb_.height = uint32_t(h);
        fb_.aspect = h > 0 ? float(w) / float(h) : 1.0f;
        fb_.is_widescreen = fb_.aspect > 1.5f;
        fb_.is_hi_def     = h >= 720;
    }

    std::fprintf(stderr, "[vk] renderer ready %ux%u images=%u\n",
                 swap_.extent().width, swap_.extent().height,
                 swap_.image_count());
}

Renderer::~Renderer() {
    vkDeviceWaitIdle(dev_.handle());
    if (auto blob = pipelines_.save_cache(); !blob.empty())
        (void)PlatformFilesystem.writeFile(pipeline_cache_path(), blob.data(), blob.size());
    pending_destroys_.clear();  // VmaBuffer destructors run vmaDestroyBuffer
    // Release all persistent mesh buffers before the allocator dies.
    // In-flight frames are already idle (vkDeviceWaitIdle above).
    for (auto& rec : meshes_) rec.vb.reset();
    meshes_.clear();
    tex_mgr_.destroy(dev_.handle(), dev_.allocator());
    quad_ib_.reset();
    pipelines_.destroy();
    // sampler_{diffuse,lightmap}_, bindless_set_layout_, bindless_pool_,
    // and pipeline_layout_ destruct automatically here — their vk::raii
    // destructors run in reverse declaration order, which matches the
    // required Vulkan destroy order (PipelineLayout before DescriptorSet-
    // Layout before samplers; DescriptorPool reclaims bindless_set_).
    for (auto& f : frames_) f.destroy(dev_.handle(), dev_.allocator());
    swap_.destroy(dev_);
}

// ===================================================================
// FRAME LOOP
// ===================================================================

void Renderer::StartFrame() {
    if (frame_active_) return;
    auto& f = frame();

    {
        int w, h;
        SDL_GetWindowSize(window_, &w, &h);
        fb_.width  = uint32_t(w);
        fb_.height = uint32_t(h);
        fb_.aspect = h > 0 ? float(w) / float(h) : 1.0f;
        fb_.is_widescreen = fb_.aspect > 1.5f;
        fb_.is_hi_def     = h >= 720;
    }

    // Wait on this slot's last submit via the Device's shared frame
    // timeline. Value 0 means the slot has never submitted, so nothing
    // to wait on. Replaces the per-slot VkFence + vkWaitForFences /
    // vkResetFences pair.
    if (f.submit_timeline_value > 0) {
        VkSemaphore sem = dev_.frame_timeline();
        VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wi.semaphoreCount = 1;
        wi.pSemaphores    = &sem;
        wi.pValues        = &f.submit_timeline_value;
        vkWaitSemaphores(dev_.handle(), &wi, UINT64_MAX);
    }

    // Poll async texture uploads and mark completed ones ready.
    tex_mgr_.poll_uploads();

    // Flush deferred buffer destructions (from worker thread CBuffClear).
    // Safe now because the fence wait guarantees the GPU is done.
    {
        std::lock_guard lk(pending_destroy_mutex_);
        pending_destroys_.clear();  // VmaBuffer destructors run vmaDestroyBuffer
    }

    VkResult acq = vkAcquireNextImageKHR(dev_.handle(), swap_.handle(),
                                          UINT64_MAX, f.sem_acquired,
                                          VK_NULL_HANDLE, &acquired_img_);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR || acq == VK_SUBOPTIMAL_KHR) {
        resize(fb_.width, fb_.height);
        acq = vkAcquireNextImageKHR(dev_.handle(), swap_.handle(),
                                     UINT64_MAX, f.sem_acquired,
                                     VK_NULL_HANDLE, &acquired_img_);
        if (acq != VK_SUCCESS) return;
    }

    f.begin(dev_.handle(), dev_.allocator());

    // Accumulate the GPU-time measurement that f.begin() just read back
    // and emit a one-line log every ~60 frames. Only active when the
    // device supports timestamps (timestamp_period_ns() > 0).
    if (dev_.timestamp_period_ns() > 0.0f) {
        const double ms = f.last_frame_gpu_ms();
        if (ms > 0.0) { gpu_ms_accum_ += ms; ++gpu_ms_count_; }
        // Log ~once per second at 60 fps.
        static uint64_t frame_counter = 0;
        ++frame_counter;
        if (frame_counter >= next_gpu_log_frame_ && gpu_ms_count_ > 0) {
            std::fprintf(stderr, "[vk] gpu avg %.2f ms (%u frames)\n",
                         gpu_ms_accum_ / gpu_ms_count_, gpu_ms_count_);
            // Per-pass breakdown, if any push_timestamp pairs were
            // recorded last frame. Tags come straight from the emitter
            // (string literals; no copy). Single-frame snapshot — no
            // rolling average — since passes come and go.
            for (const auto& [tag, ms_pass] : f.last_pass_ms()) {
                std::fprintf(stderr, "[vk]   %-16s %.3f ms\n",
                             tag ? tag : "(null)", ms_pass);
            }
            // VMA allocation churn — gates the MeshArena decision
            // (B5 in the renderer plan). High steady-state
            // alloc/free deltas mean per-chunk VmaCreateBuffer is
            // thrashing; zero deltas mean the arena isn't worth
            // the code. Print both the total count + the delta
            // versus the previous log so the 1-Hz rate is visible.
            VmaTotalStatistics vma_stats{};
            vmaCalculateStatistics(dev_.allocator(), &vma_stats);
            const uint64_t now_allocs = vma_stats.total.statistics.allocationCount;
            const uint64_t now_bytes  = vma_stats.total.statistics.allocationBytes;
            const int64_t  d_allocs   = int64_t(now_allocs) - int64_t(vma_prev_allocs_);
            const int64_t  d_bytes    = int64_t(now_bytes)  - int64_t(vma_prev_bytes_);
            std::fprintf(stderr,
                         "[vk]   vma              %llu allocs (%+lld/s) "
                         "%.1f MB (%+.1f MB/s)\n",
                         (unsigned long long)now_allocs,
                         (long long)d_allocs,
                         double(now_bytes) / (1024.0 * 1024.0),
                         double(d_bytes)  / (1024.0 * 1024.0));
            vma_prev_allocs_ = now_allocs;
            vma_prev_bytes_  = now_bytes;

            gpu_ms_accum_ = 0.0;
            gpu_ms_count_ = 0;
            next_gpu_log_frame_ = frame_counter + 60;
        }
    }

    // Reset pipeline state to safe defaults each frame.
    // Prevents stale blend/depth state from a previous frame's draw
    // leaking into the next frame's terrain draws (causes flashing).
    pso_key_ = PipelineKey{};
    pso_dirty_    = true;
    pass_active_  = false;
    frame_active_ = true;

    // Fill this frame's UBO from current state. Host-visible mapped
    // write — no fence needed because we've already waited on this
    // frame's fence above.
    {
        FrameUBO ubo{};
        ubo.light0_dir    = glm::vec4(light0_dir_eye_, 0.0f);
        ubo.light1_dir    = glm::vec4(light1_dir_eye_, 0.0f);
        ubo.light_diffuse = glm::vec4(light_diffuse_, 0.0f);
        ubo.light_ambient = glm::vec4(light_ambient_, 0.0f);
        ubo.fog_params    = glm::vec4(fog_mode_f_, fog_start_, fog_end_, fog_density_);
        ubo.fog_colour    = glm::vec4(fog_colour_[0], fog_colour_[1],
                                       fog_colour_[2], 0.0f);
        ubo.global_lm_packed =
            uint32_t(global_lm_uv_[0]) | (uint32_t(global_lm_uv_[1]) << 16);
        ubo.inv_gamma     = inv_gamma_;
        current_frame_ubo_offset_ = f.alloc_frame_ubo_slot(&ubo, sizeof(ubo));
    }

    frame().begin_cmd_timestamps();
}

void Renderer::Present() {
    if (!frame_active_) return;
    auto& f = frame();
    ensure_pass();
    end_pass();
    f.end_cmd_timestamps();
    f.flush_transient_writes(dev_.allocator());
    f.end_cmd();

    VkSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    wait.semaphore = f.sem_acquired;
    wait.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

    // Two signals: the binary sem_done for present (narrow
    // COLOR_ATTACHMENT_OUTPUT stage so the compositor releases early)
    // and the shared frame timeline so the next pass through this slot
    // can wait on a single semaphore instead of a dedicated fence.
    const uint64_t next_timeline = dev_.next_frame_timeline_value();
    VkSemaphoreSubmitInfo sigs[2]{};
    sigs[0].sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    sigs[0].semaphore = f.sem_done;
    sigs[0].stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    sigs[1].sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    sigs[1].semaphore = dev_.frame_timeline();
    sigs[1].value     = next_timeline;
    sigs[1].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    VkCommandBufferSubmitInfo csi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    csi.commandBuffer = f.cmd;
    VkSubmitInfo2 sub{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    sub.waitSemaphoreInfoCount    = 1; sub.pWaitSemaphoreInfos    = &wait;
    sub.commandBufferInfoCount    = 1; sub.pCommandBufferInfos    = &csi;
    sub.signalSemaphoreInfoCount  = 2; sub.pSignalSemaphoreInfos  = sigs;
    check(dev_.submit2(1, &sub, VK_NULL_HANDLE), "submit");
    f.submit_timeline_value = next_timeline;

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &f.sem_done;
    pi.swapchainCount     = 1;
    auto sc = swap_.handle();
    pi.pSwapchains        = &sc;
    pi.pImageIndices      = &acquired_img_;
    VkResult pr = dev_.present_khr(&pi);
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) {
        int w, h; SDL_GetWindowSize(window_, &w, &h);
        resize(uint32_t(w), uint32_t(h));
    }

    frame_active_ = false;
    frame_idx_ = (frame_idx_ + 1) % kFramesInFlight;
}

void Renderer::ensure_pass() {
    if (pass_active_) return;
    begin_pass();
    pass_active_ = true;
}

void Renderer::begin_pass() {
    auto& f = frame();
    VkCommandBuffer cmd = f.cmd;

    VkImageMemoryBarrier2 bars[2]{};
    bars[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    // No prior producer — oldLayout=UNDEFINED discards contents.
    // srcStage/Access=NONE since nothing wrote this image yet this frame.
    bars[0].srcStageMask  = VK_PIPELINE_STAGE_2_NONE;
    bars[0].dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    bars[0].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    bars[0].oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
    bars[0].newLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    bars[0].image         = swap_.image(acquired_img_);
    bars[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    const bool has_stencil = swap_.has_stencil();
    VkImageAspectFlags ds_aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (has_stencil) ds_aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;

    bars[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    bars[1].srcStageMask  = VK_PIPELINE_STAGE_2_NONE;
    bars[1].dstStageMask  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
    bars[1].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    bars[1].oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
    bars[1].newLayout     = has_stencil
                              ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                              : VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    bars[1].image         = swap_.depth_image();
    bars[1].subresourceRange = {ds_aspect, 0, 1, 0, 1};

    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 2;
    dep.pImageMemoryBarriers    = bars;
    vkCmdPipelineBarrier2(cmd, &dep);

    VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    color.imageView   = swap_.view(acquired_img_);
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    std::memcpy(color.clearValue.color.float32, clear_color_.data(), 16);

    VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depth.imageView   = swap_.depth_view();
    depth.imageLayout = has_stencil
                          ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                          : VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.clearValue.depthStencil.depth = 1.0f;

    // Stencil attachment — share the same view (combined depth+stencil
    // format). Clear to 0, don't-care on store. Only wired when the
    // swapchain picked a format with a stencil aspect.
    VkRenderingAttachmentInfo stencil{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    stencil.imageView   = swap_.depth_view();
    stencil.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    stencil.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    stencil.storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    stencil.clearValue.depthStencil.stencil = 0;

    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea      = {{0, 0}, swap_.extent()};
    ri.layerCount      = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments    = &color;
    ri.pDepthAttachment     = &depth;
    if (has_stencil) ri.pStencilAttachment = &stencil;
    vkCmdBeginRendering(cmd, &ri);

    // Y-flip via negative viewport height (VK_KHR_maintenance1, core in 1.1).
    // Geometry in game space (Y-up) renders with CCW front faces and back-face
    // culling — the canonical Vulkan configuration. No MVP row negation needed.
    apply_viewport_and_scissor();

    // Bind set 0 (bindless images + samplers) and set 1 (per-frame UBO)
    // once per frame. Every draw reads from set 0 via the tex_id packed
    // in flags, and set 1 supplies lighting / fog / gamma constants via
    // a DYNAMIC uniform buffer — the dynamic offset selects which
    // FrameUBO slot from this frame's ring the shader sees. StartFrame
    // allocates the default slot and puts its byte offset in
    // current_frame_ubo_offset_; future per-view processing rebinds
    // set 1 with a different offset before emitting view draws.
    VkDescriptorSet sets[2] = { bindless_set_, frame().frame_ubo_set };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            *pipeline_layout_, 0, 2, sets,
                            1, &current_frame_ubo_offset_);
}

void Renderer::rebuild_viewport_rects() {
    const uint32_t W = swap_.extent().width;
    const uint32_t H = swap_.extent().height;
    auto rect = [](int32_t x, int32_t y, uint32_t w, uint32_t h) {
        return VkRect2D{{x, y}, {w, h}};
    };
    // Order matches rp::ViewportLayout enum.
    viewport_rects_[int(rp::ViewportLayout::fullscreen)]            = rect(0, 0, W, H);
    viewport_rects_[int(rp::ViewportLayout::split_top)]             = rect(0, 0, W, H / 2);
    viewport_rects_[int(rp::ViewportLayout::split_bottom)]          = rect(0, int32_t(H / 2), W, H - H / 2);
    viewport_rects_[int(rp::ViewportLayout::split_left)]            = rect(0, 0, W / 2, H);
    viewport_rects_[int(rp::ViewportLayout::split_right)]           = rect(int32_t(W / 2), 0, W - W / 2, H);
    viewport_rects_[int(rp::ViewportLayout::quadrant_top_left)]     = rect(0, 0, W / 2, H / 2);
    viewport_rects_[int(rp::ViewportLayout::quadrant_top_right)]    = rect(int32_t(W / 2), 0, W - W / 2, H / 2);
    viewport_rects_[int(rp::ViewportLayout::quadrant_bottom_left)]  = rect(0, int32_t(H / 2), W / 2, H - H / 2);
    viewport_rects_[int(rp::ViewportLayout::quadrant_bottom_right)] = rect(int32_t(W / 2), int32_t(H / 2), W - W / 2, H - H / 2);
}

void Renderer::apply_viewport_and_scissor() {
    // Index into the pre-computed rect table. Y-flip is applied on the
    // viewport emit (negative height) so the existing game-space (Y-up)
    // coordinate system keeps working under CCW-front-face + back-face
    // cull.
    const VkRect2D& r = viewport_rects_[int(viewport_layout_)];

    VkViewport vp{};
    vp.x        = float(r.offset.x);
    vp.y        = float(r.offset.y + int32_t(r.extent.height));  // negative-height Y-flip
    vp.width    = float(r.extent.width);
    vp.height   = -float(r.extent.height);
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;

    VkCommandBuffer cmd = frame().cmd;
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &r);
    viewport_dirty_ = false;
}

void Renderer::StateSetViewport(int viewport_type) {
    // Caller passes the ViewportLayout enum cast to int (see
    // Minecraft.cpp:1636). Values outside the enum clamp to fullscreen.
    rp::ViewportLayout layout = rp::ViewportLayout::fullscreen;
    if (viewport_type >= 0 &&
        viewport_type <= int(rp::ViewportLayout::quadrant_bottom_right)) {
        layout = static_cast<rp::ViewportLayout>(viewport_type);
    }
    if (layout != viewport_layout_) {
        viewport_layout_ = layout;
        viewport_dirty_  = true;
    }
}

void Renderer::end_pass() {
    auto& f = frame();
    vkCmdEndRendering(f.cmd);

    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    // Present consumes via semaphore, not a pipeline stage — dstStage=NONE.
    b.srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    b.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    b.dstStageMask  = VK_PIPELINE_STAGE_2_NONE;
    b.oldLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    b.newLayout     = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    b.image         = swap_.image(acquired_img_);
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers    = &b;
    vkCmdPipelineBarrier2(f.cmd, &dep);
}

void Renderer::Clear(int flags) {
    if (!frame_active_) return;
    ensure_pass();
    uint32_t n = 0;
    VkClearAttachment atts[2]{};
    if (flags & 0x1) {
        atts[n].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        std::memcpy(atts[n].clearValue.color.float32, clear_color_.data(), 16);
        ++n;
    }
    if (flags & 0x2) {
        atts[n].aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        atts[n].clearValue.depthStencil.depth = 1.0f;
        ++n;
    }
    if (n) {
        VkClearRect r{};
        r.rect.extent = {swap_.extent().width, swap_.extent().height};
        r.layerCount  = 1;
        vkCmdClearAttachments(frame().cmd, n, atts, 1, &r);
    }
}

void Renderer::SetClearColour(const float rgba[4]) {
    std::memcpy(clear_color_.data(), rgba, 16);
}

void Renderer::resize(uint32_t w, uint32_t h) {
    if (w == 0 || h == 0) return;
    // Drain every frame slot's last graphics submit via the shared
    // timeline before touching the swapchain. Slots never submitted
    // have submit_timeline_value == 0 and need no wait.
    for (auto& f : frames_) {
        if (f.submit_timeline_value == 0) continue;
        VkSemaphore sem = dev_.frame_timeline();
        VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wi.semaphoreCount = 1;
        wi.pSemaphores    = &sem;
        wi.pValues        = &f.submit_timeline_value;
        vkWaitSemaphores(dev_.handle(), &wi, UINT64_MAX);
    }
    // A VK_SUBOPTIMAL_KHR return from vkAcquireNextImageKHR leaves the
    // acquire semaphore signaled even though we are discarding the image.
    // Reusing a still-signaled semaphore in the next acquire is UB. After
    // the fence wait above, no pending command buffer references these
    // semaphores, so it is safe to destroy + recreate them here.
    for (auto& f : frames_)
        f.reset_acquire_semaphore(dev_.handle());
    swap_.resize(dev_, w, h);
    fb_.width = swap_.extent().width;
    fb_.height = swap_.extent().height;
    fb_.aspect = fb_.height > 0 ? float(fb_.width) / float(fb_.height) : 1.0f;
    fb_.is_widescreen = fb_.aspect > 1.5f;
    fb_.is_hi_def     = fb_.height >= 720;
    rebuild_viewport_rects();
    viewport_dirty_ = true;
}

// ===================================================================
// MATRIX STACK
// ===================================================================

Renderer::MatrixStack& Renderer::stack() {
    switch (matrix_mode_) {
        case rp::MatrixStack::modelview:  return mv_stack_;
        case rp::MatrixStack::projection: return proj_stack_;
        case rp::MatrixStack::texture:    return tex_stack_;
    }
    return mv_stack_;
}

void Renderer::MatrixMode(rp::MatrixStack s) { matrix_mode_ = s; }
void Renderer::MatrixSetIdentity() { stack().top() = glm::mat4(1); }
void Renderer::MatrixTranslate(float x, float y, float z) {
    stack().top() = glm::translate(stack().top(), glm::vec3(x,y,z));
}
void Renderer::MatrixRotate(float a, float x, float y, float z) {
    stack().top() = glm::rotate(stack().top(), a, glm::vec3(x,y,z));
}
void Renderer::MatrixScale(float x, float y, float z) {
    stack().top() = glm::scale(stack().top(), glm::vec3(x,y,z));
}
void Renderer::MatrixPerspective(float fovy, float asp, float zn, float zf) {
    stack().top() = stack().top() * glm::perspective(glm::radians(fovy), asp, zn, zf);
}
void Renderer::MatrixOrthogonal(float l, float r, float b, float t, float zn, float zf) {
    stack().top() = stack().top() * glm::ortho(l, r, b, t, zn, zf);
}
void Renderer::MatrixPush() {
    auto& s = stack();
    assert(s.depth < kMaxStackDepth && "Matrix stack overflow");
    if (s.depth >= kMaxStackDepth) {
        // Previously logged-and-returned, but that silently unbalances every
        // subsequent MatrixPop. A throw surfaces the misuse instead.
        throw std::runtime_error(
            "vk::Renderer::MatrixPush: matrix stack overflow at depth 16 "
            "(raise kMaxStackDepth if the game legitimately needs deeper stacks)");
    }
    s.data[s.depth] = s.data[s.depth - 1];
    ++s.depth;
}
void Renderer::MatrixPop() {
    auto& s = stack();
    assert(s.depth > 1 && "Matrix stack underflow");
    if (s.depth > 1) --s.depth;
    else std::fprintf(stderr, "[vk] matrix stack underflow\n");
}
void Renderer::MatrixMult(float* m) {
    glm::mat4 mat; std::memcpy(&mat[0][0], m, 64);
    stack().top() = stack().top() * mat;
}
const float* Renderer::MatrixGet(rp::MatrixStack s) {
    switch (s) {
        case rp::MatrixStack::modelview:  return &mv_stack_.top()[0][0];
        case rp::MatrixStack::projection: return &proj_stack_.top()[0][0];
        case rp::MatrixStack::texture:    return &tex_stack_.top()[0][0];
    }
    return &mv_stack_.top()[0][0];
}

// ===================================================================
// STATE
// ===================================================================

void Renderer::StateSetColour(float r, float g, float b, float a) { state_colour_ = {r,g,b,a}; }
void Renderer::StateSetDepthMask(bool e) { if (pso_key_.depth_write() != e) { pso_key_.set_depth_write(e); pso_dirty_ = true; } }
void Renderer::StateSetBlendEnable(bool e) { if (pso_key_.blend_enable() != e) { pso_key_.set_blend_enable(e); pso_dirty_ = true; } }
void Renderer::StateSetBlendFunc(rp::BlendFactor s, rp::BlendFactor d) {
    uint8_t vs = blend_to_vk(s), vd = blend_to_vk(d);
    if (pso_key_.blend_src() != vs || pso_key_.blend_dst() != vd) {
        pso_key_.set_blend_src(vs); pso_key_.set_blend_dst(vd); pso_dirty_ = true;
    }
}
void Renderer::StateSetBlendFactor(unsigned int argb) {
    blend_constants_[0] = float((argb >> 16) & 0xFF) / 255.0f;
    blend_constants_[1] = float((argb >>  8) & 0xFF) / 255.0f;
    blend_constants_[2] = float((argb      ) & 0xFF) / 255.0f;
    blend_constants_[3] = float((argb >> 24) & 0xFF) / 255.0f;
}
void Renderer::StateSetAlphaFunc(rp::AlphaTest f, float ref) { alpha_test_func_ = f; alpha_ref_ = ref; }
void Renderer::StateSetDepthFunc(rp::DepthTest f) {
    uint8_t v = depth_to_vk(f);
    if (pso_key_.depth_func() != v) { pso_key_.set_depth_func(v); pso_dirty_ = true; }
}
void Renderer::StateSetFaceCull(bool e) { if (pso_key_.cull_back() != e) { pso_key_.set_cull_back(e); pso_dirty_ = true; } }
void Renderer::StateSetLineWidth(float w) {
    // Clamp to 1.0 if the device does not support wideLines. Passing any
    // non-1.0 value to vkCmdSetLineWidth on such a device is a validation
    // error (VUID-vkCmdSetLineWidth-lineWidth-00788).
    line_width_ = dev_.wide_lines_enabled() ? w : 1.0f;
}
void Renderer::StateSetWriteEnable(bool r, bool g, bool b, bool a) {
    uint8_t m = (r?1:0)|(g?2:0)|(b?4:0)|(a?8:0);
    if (pso_key_.color_mask() != m) { pso_key_.set_color_mask(m); pso_dirty_ = true; }
}
void Renderer::StateSetDepthTestEnable(bool e) { if (pso_key_.depth_test() != e) { pso_key_.set_depth_test(e); pso_dirty_ = true; } }
void Renderer::StateSetAlphaTestEnable(bool e) { alpha_test_enabled_ = e; }
void Renderer::StateSetDepthSlopeAndBias(float slope, float bias) {
    depth_bias_slope_ = slope; depth_bias_constant_ = bias;
}
void Renderer::StateSetStencil(int func, uint8_t ref, uint8_t funcMask,
                               uint8_t writeMask) {
    // Legacy GL compare constants 0x0200..0x0207 map directly to VkCompareOp
    // 0..7 (NEVER/LESS/EQUAL/LEQUAL/GREATER/NOTEQUAL/GEQUAL/ALWAYS). Any
    // out-of-range value falls back to ALWAYS.
    int offset = func - 0x0200;
    uint8_t vk_func = (offset >= 0 && offset <= 7) ? uint8_t(offset)
                                                   : uint8_t(VK_COMPARE_OP_ALWAYS);
    if (!pso_key_.stencil_test()) {
        pso_key_.set_stencil_test(true);
        pso_dirty_ = true;
    }
    if (pso_key_.stencil_func() != vk_func) {
        pso_key_.set_stencil_func(vk_func);
        pso_dirty_ = true;
    }
    stencil_ref_          = ref;
    stencil_compare_mask_ = funcMask;
    stencil_write_mask_   = writeMask;
}
void Renderer::StateSetLightDirection(int idx, float x, float y, float z) {
    glm::vec3 d = glm::normalize(glm::mat3(mv_stack_.top()) * glm::vec3(x,y,z));
    if (idx == 0) light0_dir_eye_ = d; else light1_dir_eye_ = d;
}
void Renderer::StateSetTextureEnable(bool e) { texture_enabled_ = e; }
void Renderer::StateSetLightmapEnable(bool e) { lightmap_enabled_ = e; }

void Renderer::StateSetTextureFilter(rp::TextureFilter min, rp::TextureFilter mag) {
    // mag filter decides the visible crispness; min tracks it. The
    // sampler-index packing is: bit 1 = linear-filter, bit 0 = clamp-wrap.
    const int bound = tex_mgr_.bound_tex();
    const uint8_t current = tex_mgr_.sampler_idx_for(bound);
    const bool linear = (mag == rp::TextureFilter::linear);
    const uint8_t next = uint8_t((current & 0x1) | (linear ? 0x2 : 0x0));
    tex_mgr_.set_bound_sampler_idx(next);
    (void)min;
}

void Renderer::StateSetTextureWrap(rp::TextureWrap s, rp::TextureWrap t) {
    const int bound = tex_mgr_.bound_tex();
    const uint8_t current = tex_mgr_.sampler_idx_for(bound);
    const bool clamp = (s == rp::TextureWrap::clamp_to_edge);
    const uint8_t next = uint8_t((current & 0x2) | (clamp ? 0x1 : 0x0));
    tex_mgr_.set_bound_sampler_idx(next);
    (void)t;
}
void Renderer::UpdateGamma(unsigned short g) {
    float gamma = 0.5f + float(g) / 32768.0f;
    if (gamma < 0.01f) gamma = 0.01f;
    inv_gamma_ = 1.0f / gamma;
}

void Renderer::refresh_fog_mode_f() {
    if (!fog_enabled_) { fog_mode_f_ = 0.0f; return; }
    switch (fog_mode_) {
        case rp::FogMode::linear:         fog_mode_f_ = 1.0f; break;
        case rp::FogMode::exponential:    fog_mode_f_ = 2.0f; break;
        case rp::FogMode::exponential_sq: fog_mode_f_ = 3.0f; break;
        default:                          fog_mode_f_ = 0.0f; break;
    }
}

// ===================================================================
// PUSH CONSTANTS HELPER
// ===================================================================

void Renderer::fill_push_constants(void* out, bool textured, bool lm_active,
                                    uint32_t tex_id, uint32_t lm_tex_id,
                                    const glm::vec4* tint) {
    PushConstants& pc = *static_cast<PushConstants*>(out);
    const auto& mv = mv_stack_.top();
    pc.mvp = proj_stack_.top() * mv;
    // Y-flip is handled by the negative viewport height in begin_pass() —
    // no MVP manipulation needed here.

    glm::mat3 nm(mv);
    const auto& tm = tex_stack_.top();
    pc.nm0       = glm::vec4(nm[0], tm[0][0]);
    pc.nm1       = glm::vec4(nm[1], tm[1][1]);
    pc.nm2       = glm::vec4(nm[2], tm[3][0]);
    pc.chunk_lit = glm::vec4(chunk_offset_[0], chunk_offset_[1],
                             chunk_offset_[2], lighting_enabled_ ? 1.0f : 0.0f);
    // Per-draw scalars that previously hid in the .w of light/mv-trans
    // slots now live in their own vec4. Layout: (tex_offset_y,
    // mv_translation.x, mv_translation.y, mv_translation.z).
    pc.tex_mv    = glm::vec4(tm[3][1], mv[3][0], mv[3][1], mv[3][2]);

    if (tint) {
        pc.state_colour = glm::vec4(state_colour_[0] * (*tint)[0],
                                    state_colour_[1] * (*tint)[1],
                                    state_colour_[2] * (*tint)[2],
                                    state_colour_[3] * (*tint)[3]);
    } else {
        pc.state_colour = glm::vec4(state_colour_[0], state_colour_[1],
                                    state_colour_[2], state_colour_[3]);
    }
    pc.alpha_ref  = alpha_ref_;
    // flags bit layout:
    //   [0]     textured       (diffuse sample enabled)
    //   [1]     alpha_test
    //   [2]     lm_active      (lightmap modulation enabled)
    //   [3]     force_lod_on   (StateSetForceLOD; shader uses textureLod)
    //   [4:15]  tex_id         (12 bits — slot in bindless sampled image array)
    //   [16:27] lm_tex_id      (12 bits — lightmap slot)
    //   [28:29] force_lod      (2 bits — mipmap LOD level)
    //   [30:31] sampler_idx    (2 bits — entry into the 4-sampler table)
    uint32_t flags = 0;
    if (textured && texture_enabled_) flags |= 1u;
    if (alpha_test_enabled_)          flags |= 2u;
    if (lm_active)                    flags |= 4u;
    if (force_lod_ != 0xFFu) {
        flags |= 8u;
        flags |= (force_lod_ & 0x3u) << 28u;  // 2 bits; callers use 0..2
    }
    flags |= (tex_id    & 0xFFFu) << 4u;
    flags |= (lm_tex_id & 0xFFFu) << 16u;
    const uint32_t sampler_idx = tex_mgr_.sampler_idx_for(int(tex_id)) & 0x3u;
    flags |= (sampler_idx << 30u);
    pc.flags = flags;
}

// ===================================================================
// DRAW
// ===================================================================

void Renderer::DrawVertices(int primType, int count, void* data, int vType) {
    if (count <= 0 || !data) return;

    // CBuff recording must be checked BEFORE frame_active_ — worker
    // threads rebuild chunks between frames when frame_active_ is false.
    // Recording just copies to CPU memory, no GPU state needed.
    if (dl_mgr_.is_recording()) {
        constexpr uint32_t kStd = 32;
        size_t bytes = (vType == 1) ? size_t(count) * 16 : size_t(count) * kStd;
        dl_mgr_.record_draw(primType, vType, data, bytes);
        return;
    }

    if (!frame_active_) return;
    ensure_pass();

    // Compact (vType==1) vertices are decoded in the vertex shader; we feed
    // them to the GPU at their native 16-byte stride and triangulate the
    // quads via the precomputed quad_ib_. Non-compact vertices go through
    // the 32-byte WorldStandardVertex path (triangle fan is still expanded
    // on CPU since it's uncommon and the compact format never uses it).
    const bool is_compact = (vType == 1);
    const void* vdata = data;
    if (!is_compact && primType == 0x0006) {
        auto fan = fan_to_list(vdata, count);
        if (fan.vertex_count == 0) return;
        vdata = fan.data.data();
        count = fan.vertex_count;
        primType = 0x0004;
        // fan.data is a view into a thread-local scratch; it stays valid
        // until this function returns since we don't call fan_to_list
        // again before memcpy-ing into the transient VB below.
    }

    VkPrimitiveTopology topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    bool is_quads = false;
    if (is_compact) {
        // Compact format is always quads from chunk meshing.
        is_quads = true;
    } else {
        switch (primType) {
            case 0x0001: topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;      break;
            case 0x0003: topo = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;     break;
            case 0x0004: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;  break;
            case 0x0005: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
            case 0x0007: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;  is_quads = true; break;
            default: return;
        }
    }
    if (is_quads && (count % 4) != 0) return;
    if (pso_key_.compact() != is_compact) { pso_key_.set_compact(is_compact); pso_dirty_ = true; }

    auto& f = frame();
    const VkDeviceSize stride = is_compact ? 16 : 32;
    VkDeviceSize bytes = VkDeviceSize(count) * stride;
    void* dst = f.alloc_transient(bytes);
    if (!dst) {
        // Overflow recorded inside alloc_transient; next begin() will grow.
        // The draw is dropped this frame but the slot will have headroom
        // when it's reused.
        return;
    }
    std::memcpy(dst, vdata, bytes);
    VkDeviceSize off = f.transient_pos() - bytes;

    // Bind pipeline
    if (pso_dirty_ || !(pso_key_ == last_bound_pso_)) {
        vkCmdBindPipeline(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipelines_.get(pso_key_));
        last_bound_pso_ = pso_key_;
        pso_dirty_ = false;
    }
    vkCmdSetPrimitiveTopology(f.cmd, topo);
    if (viewport_dirty_) apply_viewport_and_scissor();
    vkCmdSetBlendConstants(f.cmd, blend_constants_.data());
    vkCmdSetLineWidth(f.cmd, line_width_);
    if (pso_key_.stencil_test()) {
        vkCmdSetStencilCompareMask(f.cmd, VK_STENCIL_FACE_FRONT_AND_BACK, stencil_compare_mask_);
        vkCmdSetStencilWriteMask  (f.cmd, VK_STENCIL_FACE_FRONT_AND_BACK, stencil_write_mask_);
        vkCmdSetStencilReference  (f.cmd, VK_STENCIL_FACE_FRONT_AND_BACK, stencil_ref_);
    }

    // Dynamic depth bias
    vkCmdSetDepthBias(f.cmd, depth_bias_constant_, 0.0f, depth_bias_slope_);

    bool textured  = false;
    bool lm_active = false;
    const uint32_t tex_id    = tex_mgr_.resolve_bound_slot(textured);
    const uint32_t lm_tex_id = tex_mgr_.resolve_lightmap_slot(lm_active);
    lm_active = lm_active && lightmap_enabled_;

    VkBuffer transient_buf = f.transient_vb();
    vkCmdBindVertexBuffers(f.cmd, 0, 1, &transient_buf, &off);

    PushConstants pc{};
    fill_push_constants(&pc, textured, lm_active, tex_id, lm_tex_id);
    vkCmdPushConstants(f.cmd, *pipeline_layout_,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(PushConstants), &pc);

    if (is_quads) {
        uint32_t qc = uint32_t(count) / 4;
        if (qc > kMaxQuads) {
            throw std::runtime_error(
                "vk::Renderer::DrawVertices: quad count exceeds kMaxQuads=16384 "
                "(raise the quad index buffer allocation if the game needs more)");
        }
        vkCmdBindIndexBuffer(f.cmd, quad_ib_.handle(), 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(f.cmd, qc * 6, 1, 0, 0, 0);
    } else {
        vkCmdDraw(f.cmd, uint32_t(count), 1, 0, 0);
    }

    // Reset depth bias after draw
    depth_bias_constant_ = 0;
    depth_bias_slope_    = 0;
}

// ===================================================================
// WINDOW / FRAMEBUFFER
// ===================================================================

void Renderer::GetFramebufferSize(int& w, int& h) { w = fb_.width; h = fb_.height; }
void Renderer::SetWindowSize(int w, int h) { SDL_SetWindowSize(window_, w, h); }
void Renderer::SetFullscreen(bool fs) {
    SDL_SetWindowFullscreen(window_, fs ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0);
}
void Renderer::ReadPixels(int x, int y, int w, int h, void* buf) {
    // Synchronous readback of the most recently acquired swapchain image.
    // Intended to be called between Present and the next StartFrame so
    // the image is in PRESENT_SRC_KHR layout; any other call site will
    // generate validation errors because this function does not track
    // mid-frame transitions. `buf` receives w*h*4 bytes in the
    // swapchain's native byte order (B8G8R8A8 on our config).
    if (!buf || w <= 0 || h <= 0) return;
    const VkDeviceSize bytes = VkDeviceSize(w) * VkDeviceSize(h) * 4;

    // Full-device sync is overkill but simplifies correctness — ReadPixels
    // is a debug / screenshot entry point, not a hot path.
    vkDeviceWaitIdle(dev_.handle());

    // Staging buffer sized to the requested region.
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size  = bytes;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
               VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info{};
    VkBuffer      staging_buf   = VK_NULL_HANDLE;
    VmaAllocation staging_alloc = nullptr;
    check(vmaCreateBuffer(dev_.allocator(), &bi, &ai, &staging_buf,
                          &staging_alloc, &info),
          "readpixels staging");
    VmaBuffer staging(dev_.allocator(), staging_buf, staging_alloc,
                      info.pMappedData);

    // One-shot transient command pool + command buffer.
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pci.queueFamilyIndex = dev_.queue_family();
    VkCommandPool pool;
    check(vkCreateCommandPool(dev_.handle(), &pci, nullptr, &pool), "readpixels pool");

    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    check(vkAllocateCommandBuffers(dev_.handle(), &cai, &cmd), "readpixels cmd");

    VkCommandBufferBeginInfo bbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bbi);

    VkImage src = swap_.image(acquired_img_);

    VkImageMemoryBarrier2 to_src{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    to_src.srcStageMask  = VK_PIPELINE_STAGE_2_NONE;
    to_src.srcAccessMask = 0;
    to_src.dstStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
    to_src.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    to_src.oldLayout     = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    to_src.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_src.image         = src;
    to_src.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 1; dep.pImageMemoryBarriers = &to_src;
    vkCmdPipelineBarrier2(cmd, &dep);

    VkBufferImageCopy rgn{};
    rgn.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    rgn.imageOffset      = {x, y, 0};
    rgn.imageExtent      = {uint32_t(w), uint32_t(h), 1};
    vkCmdCopyImageToBuffer(cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           staging.handle(), 1, &rgn);

    VkImageMemoryBarrier2 to_present = to_src;
    to_present.srcStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
    to_present.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    to_present.dstStageMask  = VK_PIPELINE_STAGE_2_NONE;
    to_present.dstAccessMask = 0;
    to_present.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_present.newLayout     = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    dep.pImageMemoryBarriers = &to_present;
    vkCmdPipelineBarrier2(cmd, &dep);

    vkEndCommandBuffer(cmd);

    VkCommandBufferSubmitInfo csi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    csi.commandBuffer = cmd;
    VkSubmitInfo2 sub{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    sub.commandBufferInfoCount = 1; sub.pCommandBufferInfos = &csi;

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    check(vkCreateFence(dev_.handle(), &fci, nullptr, &fence), "readpixels fence");
    check(dev_.submit2(1, &sub, fence), "readpixels submit");
    vkWaitForFences(dev_.handle(), 1, &fence, VK_TRUE, UINT64_MAX);

    // Flush is a no-op on coherent memory; required on non-coherent.
    vmaInvalidateAllocation(dev_.allocator(), staging.allocation(), 0, bytes);
    std::memcpy(buf, info.pMappedData, size_t(bytes));

    vkDestroyFence(dev_.handle(), fence, nullptr);
    vkDestroyCommandPool(dev_.handle(), pool, nullptr);
    // staging destructs automatically.
}

void Renderer::Close() { should_close_ = true; }
bool Renderer::ShouldClose() { return should_close_; }
const rp::FrameFramebuffer& Renderer::framebuffer() const { return fb_; }
bool Renderer::IsWidescreen() { return fb_.is_widescreen; }
bool Renderer::IsHiDef() { return fb_.is_hi_def; }

// ===================================================================
// DEBUG
// ===================================================================

void Renderer::push_debug_event(const char* name) {
#ifndef NDEBUG
    if (fn_begin_label_ && frame_active_) {
        VkDebugUtilsLabelEXT l{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
        l.pLabelName = name;
        fn_begin_label_(frame().cmd, &l);
    }
#else
    (void)name;
#endif
}

void Renderer::pop_debug_event() {
#ifndef NDEBUG
    if (fn_end_label_ && frame_active_) fn_end_label_(frame().cmd);
#endif
}

// ===================================================================
// TEXTURES (delegated to TextureManager)
// ===================================================================

int Renderer::TextureCreate() { return tex_mgr_.create(); }
void Renderer::TextureFree(int idx) { tex_mgr_.free(idx, frame().deletions); }
void Renderer::TextureBind(int idx) { tex_mgr_.bind(idx); }
void Renderer::TextureData(int w, int h, void* data, int level, int) { tex_mgr_.data(w, h, data, level); }
void Renderer::TextureDataUpdate(int xo, int yo, int w, int h, void* data, int lvl) { tex_mgr_.data_update(xo, yo, w, h, data, lvl); }
std::optional<rp::LoadedImage> Renderer::load_texture_data(const char* filename) {
    return tex_mgr_.load_texture_data(filename);
}
std::optional<rp::LoadedImage> Renderer::load_texture_data(std::span<const uint8_t> bytes) {
    return tex_mgr_.load_texture_data(bytes);
}

// ===================================================================
// CBUFF (DISPLAY LISTS — delegated to DisplayListManager)
// ===================================================================

int Renderer::CBuffCreate(int n) { return dl_mgr_.create(n); }
void Renderer::CBuffDeleteAll() { dl_mgr_.delete_all(pending_destroys_, pending_destroy_mutex_); }
void Renderer::CBuffStart(int index, bool) { dl_mgr_.start(index); }
void Renderer::CBuffClear(int index) { dl_mgr_.clear(index, pending_destroys_, pending_destroy_mutex_); }
int Renderer::CBuffSize(int index) { return dl_mgr_.size(index); }
void Renderer::CBuffEnd() { dl_mgr_.end(); }

bool Renderer::CBuffCall(int index, bool) {
    if (index < 0 || !frame_active_) return false;

    auto snap = dl_mgr_.prepare(index, frame().deletions, dev_.allocator());
    if (!snap) return false;

    VkBuffer vb = snap.vb();
    const auto& draws = snap.draws();

    auto& f = frame();
    ensure_pass();

    if (viewport_dirty_) apply_viewport_and_scissor();
    vkCmdSetBlendConstants(f.cmd, blend_constants_.data());
    vkCmdSetLineWidth(f.cmd, line_width_);
    if (pso_key_.stencil_test()) {
        vkCmdSetStencilCompareMask(f.cmd, VK_STENCIL_FACE_FRONT_AND_BACK, stencil_compare_mask_);
        vkCmdSetStencilWriteMask  (f.cmd, VK_STENCIL_FACE_FRONT_AND_BACK, stencil_write_mask_);
        vkCmdSetStencilReference  (f.cmd, VK_STENCIL_FACE_FRONT_AND_BACK, stencil_ref_);
    }
    vkCmdSetDepthBias(f.cmd, depth_bias_constant_, 0.0f, depth_bias_slope_);

    bool textured  = false;
    bool lm_active = false;
    const uint32_t tex_id    = tex_mgr_.resolve_bound_slot(textured);
    const uint32_t lm_tex_id = tex_mgr_.resolve_lightmap_slot(lm_active);
    lm_active = lm_active && lightmap_enabled_;

    PushConstants pc{};
    fill_push_constants(&pc, textured, lm_active, tex_id, lm_tex_id);

    vkCmdPushConstants(f.cmd, *pipeline_layout_,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(PushConstants), &pc);

    for (auto& sd : draws) {
        // Pipeline may change between subdraws when a display list mixes
        // compact (16-byte) and standard (32-byte) vertex formats.
        if (pso_key_.compact() != sd.compact) {
            pso_key_.set_compact(sd.compact);
            pso_dirty_ = true;
        }
        if (pso_dirty_ || !(pso_key_ == last_bound_pso_)) {
            vkCmdBindPipeline(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              pipelines_.get(pso_key_));
            last_bound_pso_ = pso_key_;
            pso_dirty_ = false;
        }

        VkDeviceSize off = sd.vertex_offset;
        vkCmdBindVertexBuffers(f.cmd, 0, 1, &vb, &off);

        if (sd.compact && sd.prim_type == 0x0007) {
            // Compact quads — GPU triangulation via quad_ib_.
            vkCmdSetPrimitiveTopology(f.cmd, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
            uint32_t qc = std::min(sd.vertex_count / 4, kMaxQuads);
            vkCmdBindIndexBuffer(f.cmd, quad_ib_.handle(), 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(f.cmd, qc * 6, 1, 0, 0, 0);
        } else {
            VkPrimitiveTopology topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            switch (sd.prim_type) {
                case 0x0001: topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; break;
                case 0x0003: topo = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; break;
                case 0x0005: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
            }
            vkCmdSetPrimitiveTopology(f.cmd, topo);
            vkCmdDraw(f.cmd, sd.vertex_count, 1, 0, 0);
        }
    }

    depth_bias_constant_ = 0;
    depth_bias_slope_    = 0;
    return true;
}

// ===================================================================
// MATERIALS + TRANSIENT + SUBMIT_IMMEDIATE
// ===================================================================

rp::MaterialHandle Renderer::create_material(const rp::MaterialDesc& desc) {
    // Linear-push registry — handle.index is vector index + 1 so that
    // kInvalidMaterial (index = 0, generation = 0) stays invalid.
    // Generation starts at 1; slot reuse would bump it, but we don't
    // support material deletion yet.
    MaterialRecord rec;
    rec.desc       = desc;
    rec.generation = 1;
    material_descs_.push_back(rec);
    rp::MaterialHandle h{};
    h.index      = uint32_t(material_descs_.size());
    h.generation = rec.generation;
    return h;
}

rp::MeshHandle Renderer::create_mesh(const rp::MeshDesc& desc) {
    if (!desc.vertex_data || desc.vertex_count == 0) return {};
    // Strides are fixed per known VertexLayout. world_standard and
    // world_texgen both emit the 32-byte WorldStandardVertex layout;
    // chunk_compact uses the 16-byte packed form.
    const uint32_t stride =
        (desc.layout == rp::VertexLayout::chunk_compact) ? 16u : 32u;
    const VkDeviceSize bytes = VkDeviceSize(desc.vertex_count) * stride;

    // --- Staging buffer (host-visible, mapped) ---
    VkBufferCreateInfo sci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    sci.size  = bytes;
    sci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo sai{};
    sai.usage = VMA_MEMORY_USAGE_AUTO;
    sai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo si{};
    VkBuffer      stg = VK_NULL_HANDLE;
    VmaAllocation sa  = nullptr;
    check(vmaCreateBuffer(dev_.allocator(), &sci, &sai, &stg, &sa, &si),
          "mesh staging");
    std::memcpy(si.pMappedData, desc.vertex_data, bytes);
    vmaFlushAllocation(dev_.allocator(), sa, 0, bytes);

    // --- Device-local VB ---
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size  = bytes;
    bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo bai{};
    bai.usage = VMA_MEMORY_USAGE_AUTO;
    VkBuffer      buf = VK_NULL_HANDLE;
    VmaAllocation a   = nullptr;
    check(vmaCreateBuffer(dev_.allocator(), &bci, &bai, &buf, &a, nullptr),
          "mesh vb");

    // --- One-shot copy cmd buffer + submit. vkQueueWaitIdle is acceptable
    //     here because create_mesh is called at resource-load time, not
    //     per-frame on the hot path. Async upload (timeline + deletion-
    //     queued staging) is a future optimisation if profiling flags it.
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pci.queueFamilyIndex = dev_.queue_family();
    VkCommandPool pool = VK_NULL_HANDLE;
    check(vkCreateCommandPool(dev_.handle(), &pci, nullptr, &pool),
          "mesh cmd pool");
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool        = pool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    check(vkAllocateCommandBuffers(dev_.handle(), &cai, &cmd), "mesh cmd buf");
    VkCommandBufferBeginInfo bbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bbi);
    VkBufferCopy rgn{0, 0, bytes};
    vkCmdCopyBuffer(cmd, stg, buf, 1, &rgn);
    vkEndCommandBuffer(cmd);
    VkCommandBufferSubmitInfo csi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    csi.commandBuffer = cmd;
    VkSubmitInfo2 sub{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    sub.commandBufferInfoCount = 1;
    sub.pCommandBufferInfos    = &csi;
    check(dev_.submit2(1, &sub, VK_NULL_HANDLE), "mesh submit");
    vkQueueWaitIdle(dev_.queue());
    vmaDestroyBuffer(dev_.allocator(), stg, sa);
    vkDestroyCommandPool(dev_.handle(), pool, nullptr);

    // --- Register in the mesh slot vector, reusing a freed slot if one
    //     exists so long-running sessions don't grow the vector without
    //     bound. Generation bumps on reuse so stale handles to the old
    //     occupant are rejected by record_draw_call.
    MeshRecord rec;
    rec.vb           = VmaBuffer(dev_.allocator(), buf, a);
    rec.vertex_count = desc.vertex_count;
    rec.stride       = stride;
    rec.primitive    = desc.primitive;
    for (size_t i = 0; i < meshes_.size(); ++i) {
        if (meshes_[i].generation == 0) {
            rec.generation = meshes_[i].generation + 1;
            if (rec.generation == 0) rec.generation = 1;  // wraparound guard
            meshes_[i] = std::move(rec);
            rp::MeshHandle h{};
            h.index      = uint32_t(i + 1);
            h.generation = meshes_[i].generation;
            return h;
        }
    }
    rec.generation = 1;
    meshes_.push_back(std::move(rec));
    rp::MeshHandle h{};
    h.index      = uint32_t(meshes_.size());
    h.generation = 1;
    return h;
}

void Renderer::destroy_mesh(rp::MeshHandle handle) {
    if (handle.index == 0 || size_t(handle.index - 1) >= meshes_.size()) return;
    MeshRecord& rec = meshes_[handle.index - 1];
    if (rec.generation != handle.generation) return;
    // Defer VmaBuffer destruction through the current frame's deletion
    // queue. In-flight draws that reference the old buffer finish before
    // the queue drains (guarded by the frame's fence).
    if (rec.vb) frame().deletions.push_buffer(std::move(rec.vb));
    rec.vertex_count = 0;
    rec.stride       = 32;
    rec.primitive    = rp::PrimitiveType::triangle_list;
    rec.generation   = 0;
}

PipelineKey Renderer::pipeline_key_from_material(const rp::MaterialDesc& m) const {
    PipelineKey k{};
    k.set_depth_test(m.depth_test != rp::DepthTest::off);
    k.set_depth_write(m.depth_write);
    k.set_blend_enable(m.blend != rp::BlendMode::opaque);
    k.set_cull_back(m.cull == rp::CullMode::back_ccw ||
                    m.cull == rp::CullMode::back_cw);
    k.set_stencil_test(false);
    k.set_compact(false);
    k.set_depth_func(depth_to_vk(m.depth_test));

    // Map MaterialDesc::blend enum to Vulkan (src, dst) factors.
    using BF = rp::BlendFactor;
    BF src = BF::one, dst = BF::zero;
    switch (m.blend) {
        case rp::BlendMode::opaque:         src = BF::one;       dst = BF::zero; break;
        case rp::BlendMode::alpha:          src = BF::src_alpha; dst = BF::one_minus_src_alpha; break;
        case rp::BlendMode::additive:       src = BF::src_alpha; dst = BF::one; break;
        case rp::BlendMode::multiply:       src = BF::dst_color; dst = BF::zero; break;
        case rp::BlendMode::premultiplied:  src = BF::one;       dst = BF::one_minus_src_alpha; break;
        case rp::BlendMode::custom:         src = m.blend_src_custom; dst = m.blend_dst_custom; break;
    }
    k.set_blend_src(blend_to_vk(src));
    k.set_blend_dst(blend_to_vk(dst));
    k.set_color_mask(0xF);
    return k;
}

void Renderer::record_draw_call(const rp::DrawCall& dc) {
    // Resolve the vertex source — either a frame-scoped transient
    // allocation or a persistent MeshHandle registered via create_mesh.
    VkBuffer          vb        = VK_NULL_HANDLE;
    VkDeviceSize      vb_offset = 0;
    uint32_t          vertex_count = 0;
    rp::PrimitiveType primitive = rp::PrimitiveType::triangle_list;

    auto& f = frame();

    if (dc.source == rp::VertexSource::transient) {
        const auto& tvb = dc.transient;
        if (tvb.vertex_count == 0) return;
        vb           = f.transient_vb();
        vb_offset    = tvb.offset;
        vertex_count = tvb.vertex_count;
        primitive    = tvb.primitive;
    } else {  // VertexSource::mesh
        const uint32_t mh_i = dc.mesh.index;
        if (mh_i == 0 || size_t(mh_i - 1) >= meshes_.size()) return;
        const MeshRecord& mr = meshes_[mh_i - 1];
        if (mr.generation != dc.mesh.generation) return;
        if (mr.vertex_count == 0 || !mr.vb) return;
        vb           = mr.vb.handle();
        vb_offset    = 0;
        vertex_count = mr.vertex_count;
        primitive    = mr.primitive;
    }

    if (dc.material.index == 0) return;  // invalid handle
    const uint32_t mi = dc.material.index - 1;
    if (mi >= material_descs_.size()) return;
    const MaterialRecord& rec = material_descs_[mi];
    if (rec.generation != dc.material.generation) return;
    const rp::MaterialDesc& m = rec.desc;

    ensure_pass();

    // Derive pipeline state from the material and bind if it changed.
    const PipelineKey key = pipeline_key_from_material(m);
    if (!(key == last_bound_pso_)) {
        vkCmdBindPipeline(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipelines_.get(key));
        last_bound_pso_ = key;
        pso_dirty_      = true;  // legacy path must rebind on next submit_immediate / DrawVertices
    }
    if (viewport_dirty_) apply_viewport_and_scissor();
    vkCmdSetBlendConstants(f.cmd, blend_constants_.data());
    vkCmdSetLineWidth(f.cmd, dc.line_width > 0.0f ? dc.line_width : line_width_);

    VkPrimitiveTopology topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    switch (primitive) {
        case rp::PrimitiveType::triangle_strip: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
        case rp::PrimitiveType::triangle_fan:   topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN; break;
        case rp::PrimitiveType::line_list:      topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; break;
        case rp::PrimitiveType::line_strip:     topo = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; break;
        default: break;
    }
    vkCmdSetPrimitiveTopology(f.cmd, topo);
    vkCmdSetDepthBias(f.cmd, dc.depth_bias, 0.0f, dc.depth_slope);

    // Pull the texture from the DrawCall's override first, then the
    // material's slot[0], so ui_overlay draws are self-describing and
    // don't inherit the live texture bind the way submit_immediate
    // does. If neither carries an id (legacy materials, partially-
    // migrated sites), fall back to the currently-bound texture.
    bool material_textured = m.textured;
    uint32_t tex_id = 0;
    if (material_textured) {
        const int override_tex = int(dc.texture_override.index);
        const int slot_tex     = int(m.texture_slots[0].index);
        const int pick_tex     = override_tex > 0 ? override_tex : slot_tex;
        if (pick_tex > 0) {
            tex_id = tex_mgr_.resolve_slot(pick_tex, material_textured);
        } else {
            tex_id = tex_mgr_.resolve_bound_slot(material_textured);
        }
    }
    // Lightmap modulation is driven by the material's `lit` flag — no
    // live-state override. Migrated UI overlays set lit=false, chunk
    // materials will set lit=true. Legacy path keeps its own AND-with-
    // lightmap_enabled_ in fill_push_constants; here we ignore it.
    bool lm_active = false;
    const uint32_t lm_tex_id = tex_mgr_.resolve_lightmap_slot(lm_active);
    lm_active = lm_active && m.lit;

    vkCmdBindVertexBuffers(f.cmd, 0, 1, &vb, &vb_offset);

    PushConstants pc{};
    glm::vec4 tint(dc.tint_color[0], dc.tint_color[1],
                   dc.tint_color[2], dc.tint_color[3]);
    fill_push_constants(&pc, material_textured, lm_active, tex_id, lm_tex_id, &tint);

    // fill_push_constants is shared with the legacy path and mixes live
    // flags (alpha_test_enabled_, force_lod_, texture_enabled_) into
    // pc.flags + pc.alpha_ref. Overwrite every field that could have
    // leaked through so the final PC is derived purely from the material
    // and the DrawCall. Without this, a migrated subsystem's rendering
    // would silently depend on whatever legacy StateSet*/MatrixMode
    // calls ran most recently.
    pc.alpha_ref = dc.alpha_ref_override >= 0.0f ? dc.alpha_ref_override
                                                 : m.alpha_ref;
    uint32_t mat_flags = 0;
    if (material_textured)                     mat_flags |= 0x1u;   // FLAG_TEXTURED
    if (m.alpha_test != rp::AlphaTest::off)    mat_flags |= 0x2u;   // FLAG_ALPHA_TEST
    if (lm_active)                             mat_flags |= 0x4u;   // FLAG_LIGHTMAP
    if (dc.forced_lod > 0) {                                        // FLAG_FORCE_LOD
        mat_flags |= 0x8u;
        mat_flags |= (uint32_t(dc.forced_lod) & 0x3u) << 28u;
    }
    mat_flags |= (tex_id    & 0xFFFu) << 4u;
    mat_flags |= (lm_tex_id & 0xFFFu) << 16u;
    const uint32_t sampler_idx = tex_mgr_.sampler_idx_for(int(tex_id)) & 0x3u;
    mat_flags |= (sampler_idx << 30u);
    pc.flags = mat_flags;
    // DrawCall carries its own texture transform — override the live
    // tex_stack values fill_push_constants pulled in, so migrated draws
    // don't inherit a stale legacy TextureMatrix.
    pc.nm0.w    = dc.uv_scale[0];
    pc.nm1.w    = dc.uv_scale[1];
    pc.nm2.w    = dc.uv_offset[0];
    pc.tex_mv.x = dc.uv_offset[1];
    // Same story for state_colour: fill_push_constants multiplies the
    // live state_colour_ register into tint, which means queued
    // ui_overlay / world_draws DrawCalls that drain later in the frame
    // get retroactively tinted by whichever legacy StateSetColour call
    // fired last (e.g. a Shiggy-UI button-hover yellow overlay). Make
    // the DrawCall fully self-describing — pc.state_colour is just
    // dc.tint_color.
    pc.state_colour = glm::vec4(dc.tint_color[0], dc.tint_color[1],
                                dc.tint_color[2], dc.tint_color[3]);
    // And lighting_enabled_ leaks the same way: a late-frame
    // StateSetLightingEnable(false) (HUD / text passes do this) would
    // turn off per-vertex directional lighting for every queued
    // entity / model DrawCall at drain time. Use the material's lit
    // flag instead — the Tier-C material table picks it per bucket.
    // Also zero chunk_lit.xyz: fill_push_constants reads the live
    // chunk_offset_ register (set per-chunk during legacy terrain
    // rendering and never reset at the end), so entity / UI DrawCalls
    // that drain later in the frame add a stale (cx*16, cy*16, cz*16)
    // offset to every vertex position — the whole draw shifts
    // arbitrarily many tiles offscreen. Regular DrawCalls carry no
    // chunk offset; only ChunkDrawCall does and it has its own path.
    pc.chunk_lit = glm::vec4(0.0f, 0.0f, 0.0f, m.lit ? 1.0f : 0.0f);
    // DrawCall.transform composes on top of the current matrix stacks
    // (same convention the legacy MatrixPush/Translate pattern produces).
    // For screen-space UI overlays the transform is almost always identity
    // — the vertex positions are authored directly in screen pixels — so
    // skip the multiply when it is.
    bool tform_is_identity = true;
    for (int i = 0; i < 16 && tform_is_identity; ++i) {
        const float expected = (i % 5 == 0) ? 1.0f : 0.0f;  // diag = 1, off-diag = 0
        if (dc.transform[i] != expected) tform_is_identity = false;
    }
    if (!tform_is_identity) {
        glm::mat4 tform(1.0f);
        std::memcpy(&tform[0][0], dc.transform, sizeof(float) * 16);
        pc.mvp = pc.mvp * tform;
    }
    vkCmdPushConstants(f.cmd, *pipeline_layout_,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(PushConstants), &pc);

    vkCmdDraw(f.cmd, vertex_count, 1, 0, 0);
}

void Renderer::render_frame(const rp::FrameDesc& desc) {
    if (!frame_active_) return;
    if (desc.views.empty() && desc.ui_overlay.empty()) return;

    auto& f = this->frame();

    // Save matrix stack tops once so the single stored default survives
    // both the per-view loops (which swap in view.camera) and the
    // ui_overlay loop (which forces identity).
    const glm::mat4 saved_proj = proj_stack_.top();
    const glm::mat4 saved_mv   = mv_stack_.top();
    const uint32_t  saved_ubo_offset = current_frame_ubo_offset_;

    // --- Per-view processing --------------------------------------------
    //
    // During the Tier-C migration period most views are empty (legacy
    // path still does the drawing), but we still process each view so
    // subsystems can push DrawCalls incrementally into rp::world_draws
    // and see them flow through without a renderer-side toggle.
    // GameRenderer keeps view.clear.flags at CLEAR_NONE until Phase 5
    // retires the legacy Clear() — see the comment next to
    // current_view.clear in GameRenderer::renderLevel.
    for (const rp::ViewDesc& view : desc.views) {
        f.push_timestamp("view");
        ensure_pass();

        // Matrix convention for view.world_* draws: force both stacks to
        // identity, same as the ui_overlay loop below. Migrated Tier-C
        // producers (plce::world::MeshBuilder) snapshot the full live
        // proj*mv at flush() time into DrawCall.transform — replaying that
        // on top of a non-identity view.camera would double-apply the
        // world camera (view.proj * view.view * (live_proj * live_mv)) and
        // smear / shrink every entity / tile-entity / beacon / weather
        // draw. view.camera stays on ViewDesc for future chunk_*
        // processing, which will consume it directly (ChunkDrawCall
        // carries chunk_offset, not a full transform).
        proj_stack_.top() = glm::mat4(1.0f);
        mv_stack_.top()   = glm::mat4(1.0f);

        // Viewport + scissor override. Legacy viewport_layout_ stays
        // untouched; we set the VkViewport + VkRect2D directly on the
        // command buffer and mark the legacy state dirty so it gets
        // reapplied after render_frame returns.
        if (view.viewport_layout != rp::ViewportLayout::fullscreen ||
            view.scissor.width != 0) {
            viewport_layout_ = view.viewport_layout;
            viewport_dirty_  = true;
            apply_viewport_and_scissor();
            if (view.scissor.width > 0) {
                VkRect2D sc{};
                sc.offset.x      = view.scissor.x;
                sc.offset.y      = view.scissor.y;
                sc.extent.width  = view.scissor.width;
                sc.extent.height = view.scissor.height;
                vkCmdSetScissor(f.cmd, 0, 1, &sc);
            }
        }

        // Build a FrameUBO from this view's fog_profiles[0] + lighting.
        // Allocate a ring slot, rebind set 1 with the new dynamic offset.
        FrameUBO vubo{};
        const rp::FogProfile& fp = view.fog_profiles[0];
        const float fog_mode_f = (fp.mode == rp::FogMode::linear)         ? 1.0f
                               : (fp.mode == rp::FogMode::exponential)    ? 2.0f
                               : (fp.mode == rp::FogMode::exponential_sq) ? 3.0f
                                                                           : 0.0f;
        vubo.fog_params = glm::vec4(fog_mode_f, fp.start, fp.end, fp.density);
        vubo.fog_colour = glm::vec4(fp.color[0], fp.color[1], fp.color[2], 0.0f);
        vubo.light0_dir = glm::vec4(view.lighting.directional[0].direction[0],
                                    view.lighting.directional[0].direction[1],
                                    view.lighting.directional[0].direction[2], 0);
        vubo.light1_dir = glm::vec4(view.lighting.directional[1].direction[0],
                                    view.lighting.directional[1].direction[1],
                                    view.lighting.directional[1].direction[2], 0);
        vubo.light_diffuse = glm::vec4(view.lighting.directional[0].color[0],
                                       view.lighting.directional[0].color[1],
                                       view.lighting.directional[0].color[2], 0);
        vubo.light_ambient = glm::vec4(view.lighting.ambient_color[0],
                                       view.lighting.ambient_color[1],
                                       view.lighting.ambient_color[2], 0);
        vubo.global_lm_packed = 0;
        vubo.inv_gamma        = inv_gamma_;

        current_frame_ubo_offset_ = f.alloc_frame_ubo_slot(&vubo, sizeof(vubo));

        // Rebind set 1 with the new dynamic offset. Set 0 (bindless) stays.
        {
            VkDescriptorSet one_set = f.frame_ubo_set;
            vkCmdBindDescriptorSets(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    *pipeline_layout_, 1, 1, &one_set,
                                    1, &current_frame_ubo_offset_);
        }

        // Optional in-pass clear of this view's attachments. Rare — most
        // views inherit the frame's begin_pass LOAD_OP_CLEAR. vkCmdClear-
        // Attachments clears the current scissor rect inside an active
        // render pass.
        if (view.clear.flags != rp::CLEAR_NONE) {
            VkClearAttachment atts[3]{};
            uint32_t n = 0;
            if (view.clear.flags & rp::CLEAR_COLOR) {
                atts[n].aspectMask      = VK_IMAGE_ASPECT_COLOR_BIT;
                atts[n].colorAttachment = 0;
                std::memcpy(atts[n].clearValue.color.float32,
                            view.clear.color, sizeof(float) * 4);
                ++n;
            }
            if (view.clear.flags & rp::CLEAR_DEPTH) {
                atts[n].aspectMask      = VK_IMAGE_ASPECT_DEPTH_BIT;
                atts[n].clearValue.depthStencil.depth = view.clear.depth;
                ++n;
            }
            if (view.clear.flags & rp::CLEAR_STENCIL) {
                atts[n].aspectMask      = VK_IMAGE_ASPECT_STENCIL_BIT;
                atts[n].clearValue.depthStencil.stencil = view.clear.stencil;
                ++n;
            }
            if (n > 0) {
                VkClearRect r{};
                r.rect.extent = {swap_.extent().width, swap_.extent().height};
                r.layerCount  = 1;
                vkCmdClearAttachments(f.cmd, n, atts, 1, &r);
            }
        }

        // Draw order: chunks (not yet wired — ChunkDrawCall has a
        // slightly different shape; Tier C7 will wire them), then
        // world spans in opaque → alpha_test → transparent order, then
        // debug overlay.
        // TODO(C7): process view.chunk_opaque / chunk_alpha_test /
        // chunk_transparent once ChunkDrawCall handling lands.
        for (const rp::DrawCall& dc : view.world_opaque)      record_draw_call(dc);
        for (const rp::DrawCall& dc : view.world_alpha_test)  record_draw_call(dc);
        for (const rp::DrawCall& dc : view.world_transparent) record_draw_call(dc);
        for (const rp::DrawCall& dc : view.debug_overlay)     record_draw_call(dc);

        f.pop_timestamp();
    }

    // --- UI overlay (identity matrices; DrawCall::transform carries its
    //     own screen-space ortho) ----------------------------------------
    if (!desc.ui_overlay.empty()) {
        f.push_timestamp("ui_overlay");
        ensure_pass();
        proj_stack_.top() = glm::mat4(1.0f);
        mv_stack_.top()   = glm::mat4(1.0f);

        // Rebind set 1 with the frame-default UBO offset (StartFrame's
        // snapshot). ui_overlay draws inherit the frame's lighting/fog/
        // gamma — they don't need per-view UBO state.
        current_frame_ubo_offset_ = saved_ubo_offset;
        VkDescriptorSet one_set = f.frame_ubo_set;
        vkCmdBindDescriptorSets(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                *pipeline_layout_, 1, 1, &one_set,
                                1, &current_frame_ubo_offset_);

        for (const rp::DrawCall& dc : desc.ui_overlay) {
            record_draw_call(dc);
        }
        f.pop_timestamp();
    }

    // Restore the matrix stacks for any subsequent legacy draws. The
    // FrameUBO dynamic offset has already been reset above.
    proj_stack_.top() = saved_proj;
    mv_stack_.top()   = saved_mv;
}

std::pair<rp::TransientVertexBuffer, std::span<std::byte>>
Renderer::alloc_transient_vertices(uint32_t count, rp::VertexLayout,
                                   rp::PrimitiveType prim) {
    if (!frame_active_ || count == 0) return {{}, {}};
    constexpr VkDeviceSize stride = 32;
    VkDeviceSize bytes = VkDeviceSize(count) * stride;
    void* p = frame().alloc_transient(bytes);
    if (!p) return {{}, {}};
    VkDeviceSize off = frame().transient_pos() - bytes;
    rp::TransientVertexBuffer tvb{};
    tvb.frame_index  = frame_idx_;
    tvb.offset       = uint32_t(off);
    tvb.vertex_count = count;
    tvb.primitive    = prim;
    return {tvb, std::span<std::byte>{static_cast<std::byte*>(p), size_t(bytes)}};
}

void Renderer::submit_immediate(const rp::DrawCall& dc) {
    if (!frame_active_ || dc.source != rp::VertexSource::transient) return;
    const auto& tvb = dc.transient;
    if (tvb.vertex_count == 0) return;
    auto& f = frame();
    ensure_pass();

    if (pso_dirty_ || !(pso_key_ == last_bound_pso_)) {
        vkCmdBindPipeline(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipelines_.get(pso_key_));
        last_bound_pso_ = pso_key_; pso_dirty_ = false;
    }
    if (viewport_dirty_) apply_viewport_and_scissor();
    vkCmdSetBlendConstants(f.cmd, blend_constants_.data());
    vkCmdSetLineWidth(f.cmd, line_width_);
    if (pso_key_.stencil_test()) {
        vkCmdSetStencilCompareMask(f.cmd, VK_STENCIL_FACE_FRONT_AND_BACK, stencil_compare_mask_);
        vkCmdSetStencilWriteMask  (f.cmd, VK_STENCIL_FACE_FRONT_AND_BACK, stencil_write_mask_);
        vkCmdSetStencilReference  (f.cmd, VK_STENCIL_FACE_FRONT_AND_BACK, stencil_ref_);
    }
    vkCmdSetDepthBias(f.cmd, depth_bias_constant_, 0.0f, depth_bias_slope_);

    VkPrimitiveTopology topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    switch (tvb.primitive) {
        case rp::PrimitiveType::triangle_strip: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
        case rp::PrimitiveType::triangle_fan:   topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN; break;
        case rp::PrimitiveType::line_list:      topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; break;
        case rp::PrimitiveType::line_strip:     topo = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; break;
        default: break;
    }
    vkCmdSetPrimitiveTopology(f.cmd, topo);

    bool textured  = false;
    bool lm_active = false;
    const uint32_t tex_id    = tex_mgr_.resolve_bound_slot(textured);
    const uint32_t lm_tex_id = tex_mgr_.resolve_lightmap_slot(lm_active);
    lm_active = lm_active && lightmap_enabled_;

    VkDeviceSize off = tvb.offset;
    VkBuffer transient_buf = f.transient_vb();
    vkCmdBindVertexBuffers(f.cmd, 0, 1, &transient_buf, &off);

    PushConstants pc{};
    glm::vec4 tint(dc.tint_color[0], dc.tint_color[1],
                   dc.tint_color[2], dc.tint_color[3]);
    fill_push_constants(&pc, textured, lm_active, tex_id, lm_tex_id, &tint);
    vkCmdPushConstants(f.cmd, *pipeline_layout_,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(PushConstants), &pc);

    vkCmdDraw(f.cmd, tvb.vertex_count, 1, 0, 0);

    depth_bias_constant_ = 0;
    depth_bias_slope_    = 0;
}

}  // namespace plce::vk

// ===================================================================
// FACTORY
// ===================================================================

std::unique_ptr<rp::IRenderPath> make_vulkan_render_path(SDL_Window* window) {
    return std::make_unique<plce::vk::Renderer>(window);
}
