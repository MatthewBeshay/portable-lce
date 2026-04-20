#include "Renderer.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <stdexcept>

#include "platform/PlatformTypes.h"
#include "vk3/shaders/basic.vert.spv.h"  // kBasicVertSpv
#include "vk3/shaders/basic.frag.spv.h"  // kBasicFragSpv

namespace plce::vk3 {

namespace {

void check(VkResult r, const char* w) {
    if (r != VK_SUCCESS) {
        char b[128]; std::snprintf(b, sizeof b, "%s: %d", w, int(r));
        throw std::runtime_error(b);
    }
}

uint8_t blend_to_vk(rp::BlendFactor f) {
    using BF = rp::BlendFactor;
    switch (f) {
        case BF::zero:                     return VK_BLEND_FACTOR_ZERO;
        case BF::one:                      return VK_BLEND_FACTOR_ONE;
        case BF::src_color:                return VK_BLEND_FACTOR_SRC_COLOR;
        case BF::one_minus_src_color:      return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case BF::src_alpha:                return VK_BLEND_FACTOR_SRC_ALPHA;
        case BF::one_minus_src_alpha:      return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case BF::dst_color:                return VK_BLEND_FACTOR_DST_COLOR;
        case BF::one_minus_dst_color:      return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case BF::dst_alpha:                return VK_BLEND_FACTOR_DST_ALPHA;
        case BF::one_minus_dst_alpha:      return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case BF::constant_alpha:           return VK_BLEND_FACTOR_CONSTANT_ALPHA;
        case BF::one_minus_constant_alpha: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    }
    return VK_BLEND_FACTOR_ONE;
}

uint8_t depth_to_vk(rp::DepthTest f) {
    using DT = rp::DepthTest;
    switch (f) {
        case DT::off:           return VK_COMPARE_OP_ALWAYS;
        case DT::less:          return VK_COMPARE_OP_LESS;
        case DT::less_equal:    return VK_COMPARE_OP_LESS_OR_EQUAL;
        case DT::equal:         return VK_COMPARE_OP_EQUAL;
        case DT::greater:       return VK_COMPARE_OP_GREATER;
        case DT::greater_equal: return VK_COMPARE_OP_GREATER_OR_EQUAL;
        case DT::always:        return VK_COMPARE_OP_ALWAYS;
    }
    return VK_COMPARE_OP_LESS_OR_EQUAL;
}

int* stb_to_argb(unsigned char* px, int w, int h) {
    int* out = new int[w * h];
    for (int i = 0; i < w * h; ++i) {
        unsigned char r = px[i*4], g = px[i*4+1], b = px[i*4+2], a = px[i*4+3];
        out[i] = (a << 24) | (r << 16) | (g << 8) | b;
    }
    return out;
}

// Push constant block -- matches the GLSL layout in basic.vert / basic.frag.
// Single 256-byte range shared by both vertex and fragment stages.
struct alignas(16) PushConstants {
    glm::mat4 mvp;              // 0
    glm::vec4 nm0;              // 64   mat3(mv)[0].xyz, tex_scale_x
    glm::vec4 nm1;              // 80   mat3(mv)[1].xyz, tex_scale_y
    glm::vec4 nm2;              // 96   mat3(mv)[2].xyz, tex_offset_x
    glm::vec4 chunk_lit;        // 112  chunk_offset.xyz, lighting_enabled
    glm::vec4 l0;               // 128  light0_dir.xyz, tex_offset_y
    glm::vec4 l1;               // 144  light1_dir.xyz, mv_translation.x
    glm::vec4 ldiff;            // 160  light_diffuse.xyz, mv_translation.y
    glm::vec4 lamb;             // 176  light_ambient.xyz, mv_translation.z
    glm::vec4 fog_params;       // 192  mode, start, end, density
    glm::vec4 state_colour;     // 208
    glm::vec4 fog_colour;       // 224
    float     alpha_ref;        // 240
    float     inv_gamma;        // 244
    uint32_t  flags;            // 248  bit0=textured, bit1=alpha_test, bit2=lightmap
    uint32_t  global_lm_packed; // 252
};
static_assert(sizeof(PushConstants) == 256);

// Expand compact 16-byte vertex format to 32-byte world_standard.
std::vector<std::byte> expand_compact(const void* data, int& count) {
    constexpr uint32_t kStride = 32;
    int quads = count / 4;
    int tri_verts = quads * 6;
    std::vector<std::byte> out(size_t(tri_verts) * kStride);
    const int16_t* src = static_cast<const int16_t*>(data);
    for (int q = 0; q < quads; ++q) {
        std::byte expanded[4 * kStride];
        for (int v = 0; v < 4; ++v) {
            const int16_t* sv = src + q * 4 * 8 + v * 8;
            auto* dst = expanded + v * kStride;
            auto* dstF = reinterpret_cast<float*>(dst);
            dstF[0] = sv[0] / 1024.0f;
            dstF[1] = sv[1] / 1024.0f;
            dstF[2] = sv[2] / 1024.0f;
            dstF[3] = sv[4] / 8192.0f;
            dstF[4] = sv[5] / 8192.0f;
            uint16_t packed = uint16_t(int(sv[3]) + 32768);
            dst[20] = std::byte(uint8_t((packed & 0x1F) * 255 / 31));
            dst[21] = std::byte(uint8_t(((packed >> 5) & 0x3F) * 255 / 63));
            dst[22] = std::byte(uint8_t(((packed >> 11) & 0x1F) * 255 / 31));
            dst[23] = std::byte(255);
            dst[24] = std::byte(0); dst[25] = std::byte(127);
            dst[26] = std::byte(0); dst[27] = std::byte(0);
            auto* dstS = reinterpret_cast<int16_t*>(dst + 28);
            dstS[0] = sv[6]; dstS[1] = sv[7];
        }
        auto put = [&](int ti, int vi) {
            std::memcpy(out.data() + (q * 6 + ti) * kStride,
                        expanded + vi * kStride, kStride);
        };
        put(0,0); put(1,1); put(2,2); put(3,0); put(4,2); put(5,3);
    }
    count = tri_verts;
    return out;
}

// Convert triangle fan to triangle list on CPU.
std::vector<std::byte> fan_to_list(const void* data, int& count) {
    constexpr uint32_t kStride = 32;
    if (count < 3) { count = 0; return {}; }
    int tri_count = count - 2;
    int tri_verts = tri_count * 3;
    std::vector<std::byte> out(size_t(tri_verts) * kStride);
    const std::byte* src = static_cast<const std::byte*>(data);
    for (int i = 0; i < tri_count; ++i) {
        std::memcpy(out.data() + (i * 3 + 0) * kStride, src, kStride);
        std::memcpy(out.data() + (i * 3 + 1) * kStride, src + (i + 1) * kStride, kStride);
        std::memcpy(out.data() + (i * 3 + 2) * kStride, src + (i + 2) * kStride, kStride);
    }
    count = tri_verts;
    return out;
}

// Thread-local display list recording state
struct RecState { int id = -1; std::vector<Renderer::DisplayListDraw> draws; };
thread_local RecState t_rec;

}  // namespace

// ===================================================================
// LIFECYCLE
// ===================================================================

Renderer::Renderer(SDL_Window* window)
    : dev_({window,
#ifdef NDEBUG
            false
#else
            true
#endif
           }),
      window_(window) {
    textures_.reserve(256);
    display_lists_.reserve(4096);

    swap_.create(dev_, 0, 0);

    for (auto& f : frames_)
        f.create(dev_.handle(), dev_.allocator(), dev_.queue_family());

    // Descriptor set layout: two combined image samplers (diffuse + lightmap)
    {
        VkDescriptorSetLayoutBinding bindings[2]{};
        bindings[0].binding         = 0;
        bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[1].binding         = 1;
        bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo ci{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = 2;
        ci.pBindings    = bindings;
        check(vkCreateDescriptorSetLayout(dev_.handle(), &ci, nullptr,
                                          &tex_set_layout_),
              "desc layout");
    }

    // Descriptor pool
    {
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8192};
        VkDescriptorPoolCreateInfo ci{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ci.maxSets       = 4096;
        ci.poolSizeCount = 1;
        ci.pPoolSizes    = &ps;
        check(vkCreateDescriptorPool(dev_.handle(), &ci, nullptr, &tex_pool_),
              "desc pool");
    }

    // Pipeline layout: single shared push constant range (256 bytes)
    {
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pc.offset = 0;
        pc.size   = 256;
        VkPipelineLayoutCreateInfo ci{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        ci.setLayoutCount         = 1;
        ci.pSetLayouts            = &tex_set_layout_;
        ci.pushConstantRangeCount = 1;
        ci.pPushConstantRanges    = &pc;
        check(vkCreatePipelineLayout(dev_.handle(), &ci, nullptr,
                                     &pipeline_layout_),
              "pipeline layout");
    }

    // Pipeline cache + warm-up
    {
        PipelineCache::Config pc{};
        pc.device       = dev_.handle();
        pc.layout       = pipeline_layout_;
        pc.color_format = swap_.format();
        pc.depth_format = swap_.depth_format();
        pc.vert_spv     = kBasicVertSpv;
        pc.vert_size    = sizeof(kBasicVertSpv);
        pc.frag_spv     = kBasicFragSpv;
        pc.frag_size    = sizeof(kBasicFragSpv);
        pipelines_.init(pc);
        pipelines_.load_cache("pipeline_cache.bin");
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
        check(vmaCreateBuffer(dev_.allocator(), &bi, &ai, &quad_ib_,
                              &quad_ib_alloc_, nullptr), "quad ib");

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
        vkCmdCopyBuffer(cmd, stg, quad_ib_, 1, &rgn);
        vkEndCommandBuffer(cmd);
        VkCommandBufferSubmitInfo csi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        csi.commandBuffer = cmd;
        VkSubmitInfo2 sub{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        sub.commandBufferInfoCount = 1; sub.pCommandBufferInfos = &csi;
        vkQueueSubmit2(dev_.queue(), 1, &sub, VK_NULL_HANDLE);
        vkQueueWaitIdle(dev_.queue());
        vmaDestroyBuffer(dev_.allocator(), stg, sa);
        vkDestroyCommandPool(dev_.handle(), pool, nullptr);
    }

    // Staging buffer + upload pool for textures
    {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = kStagingSize; bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO;
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                   VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo info{};
        check(vmaCreateBuffer(dev_.allocator(), &bi, &ai, &staging_buf_,
                              &staging_alloc_, &info), "staging buf");
        staging_mapped_ = static_cast<std::byte*>(info.pMappedData);

        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = dev_.queue_family();
        check(vkCreateCommandPool(dev_.handle(), &pci, nullptr, &upload_pool_), "upload pool");
    }

    default_tex_ = ensure_default_texture();
    default_lm_  = ensure_default_lightmap();
    bound_tex_   = default_tex_;

    {
        int w, h;
        SDL_GetWindowSize(window_, &w, &h);
        fb_.width  = uint32_t(w);
        fb_.height = uint32_t(h);
        fb_.aspect = h > 0 ? float(w) / float(h) : 1.0f;
        fb_.is_widescreen = fb_.aspect > 1.5f;
        fb_.is_hi_def     = h >= 720;
    }

    std::fprintf(stderr, "[vk3] renderer ready %ux%u images=%u\n",
                 swap_.extent().width, swap_.extent().height,
                 swap_.image_count());
}

Renderer::~Renderer() {
    vkDeviceWaitIdle(dev_.handle());
    pipelines_.save_cache("pipeline_cache.bin");
    for (auto& pd : pending_destroys_)
        vmaDestroyBuffer(dev_.allocator(), pd.buf, pd.alloc);
    pending_destroys_.clear();
    for (auto& cb : display_lists_)
        if (cb.vb) vmaDestroyBuffer(dev_.allocator(), cb.vb, cb.alloc);
    if (upload_pool_) vkDestroyCommandPool(dev_.handle(), upload_pool_, nullptr);
    if (staging_buf_) vmaDestroyBuffer(dev_.allocator(), staging_buf_, staging_alloc_);
    if (quad_ib_) vmaDestroyBuffer(dev_.allocator(), quad_ib_, quad_ib_alloc_);
    pipelines_.destroy();
    if (pipeline_layout_) vkDestroyPipelineLayout(dev_.handle(), pipeline_layout_, nullptr);
    for (auto& t : textures_) {
        if (t.view)  vkDestroyImageView(dev_.handle(), t.view, nullptr);
        if (t.image) vmaDestroyImage(dev_.allocator(), t.image, t.alloc);
    }
    for (auto& [k, s] : sampler_cache_)
        vkDestroySampler(dev_.handle(), s, nullptr);
    if (tex_pool_)       vkDestroyDescriptorPool(dev_.handle(), tex_pool_, nullptr);
    if (tex_set_layout_) vkDestroyDescriptorSetLayout(dev_.handle(), tex_set_layout_, nullptr);
    for (auto& f : frames_) f.destroy(dev_.handle(), dev_.allocator());
    swap_.destroy(dev_);
}

// ===================================================================
// SAMPLER CACHE
// ===================================================================

VkSampler Renderer::get_or_create_sampler(const SamplerKey& key) {
    auto it = sampler_cache_.find(key);
    if (it != sampler_cache_.end()) return it->second;

    VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    ci.magFilter    = key.mag_filter;
    ci.minFilter    = key.min_filter;
    ci.mipmapMode   = key.mip_mode;
    ci.addressModeU = key.wrap_s;
    ci.addressModeV = key.wrap_t;
    ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    ci.maxLod       = VK_LOD_CLAMP_NONE;

    VkSampler sampler = VK_NULL_HANDLE;
    check(vkCreateSampler(dev_.handle(), &ci, nullptr, &sampler), "sampler");
    sampler_cache_.emplace(key, sampler);
    return sampler;
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

    vkWaitForFences(dev_.handle(), 1, &f.fence, VK_TRUE, UINT64_MAX);

    // Flush deferred buffer destructions (from worker thread CBuffClear).
    // Safe now because the fence wait guarantees the GPU is done.
    {
        std::lock_guard lk(pending_destroy_mutex_);
        for (auto& pd : pending_destroys_)
            vmaDestroyBuffer(dev_.allocator(), pd.buf, pd.alloc);
        pending_destroys_.clear();
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

    f.begin(dev_.handle());

    // Reset pipeline state to safe defaults each frame.
    // Prevents stale blend/depth state from a previous frame's draw
    // leaking into the next frame's terrain draws (causes flashing).
    pso_key_ = PipelineKey{};
    pso_dirty_    = true;
    pass_active_  = false;
    frame_active_ = true;
}

void Renderer::Present() {
    if (!frame_active_) return;
    auto& f = frame();
    ensure_pass();
    end_pass();
    f.end_cmd();

    VkSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    wait.semaphore = f.sem_acquired;
    wait.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSemaphoreSubmitInfo sig{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    sig.semaphore = f.sem_done;
    sig.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
    VkCommandBufferSubmitInfo csi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    csi.commandBuffer = f.cmd;
    VkSubmitInfo2 sub{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    sub.waitSemaphoreInfoCount    = 1; sub.pWaitSemaphoreInfos    = &wait;
    sub.commandBufferInfoCount    = 1; sub.pCommandBufferInfos    = &csi;
    sub.signalSemaphoreInfoCount  = 1; sub.pSignalSemaphoreInfos  = &sig;
    check(vkQueueSubmit2(dev_.queue(), 1, &sub, f.fence), "submit");

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &f.sem_done;
    pi.swapchainCount     = 1;
    auto sc = swap_.handle();
    pi.pSwapchains        = &sc;
    pi.pImageIndices      = &acquired_img_;
    VkResult pr = vkQueuePresentKHR(dev_.queue(), &pi);
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

    bars[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    bars[1].srcStageMask  = VK_PIPELINE_STAGE_2_NONE;
    bars[1].dstStageMask  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
    bars[1].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    bars[1].oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
    bars[1].newLayout     = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    bars[1].image         = swap_.depth_image();
    bars[1].subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

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
    depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.clearValue.depthStencil.depth = 1.0f;

    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea      = {{0, 0}, swap_.extent()};
    ri.layerCount      = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments    = &color;
    ri.pDepthAttachment     = &depth;
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp{0, 0, float(swap_.extent().width),
                  float(swap_.extent().height), 0, 1};
    VkRect2D sc_r{{0,0}, swap_.extent()};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc_r);
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
    for (auto& f : frames_)
        vkWaitForFences(dev_.handle(), 1, &f.fence, VK_TRUE, UINT64_MAX);
    swap_.resize(dev_, w, h);
    fb_.width = swap_.extent().width;
    fb_.height = swap_.extent().height;
    fb_.aspect = fb_.height > 0 ? float(fb_.width) / float(fb_.height) : 1.0f;
    fb_.is_widescreen = fb_.aspect > 1.5f;
    fb_.is_hi_def     = fb_.height >= 720;
}

// ===================================================================
// MATRIX STACK
// ===================================================================

std::vector<glm::mat4>& Renderer::stack() {
    switch (matrix_mode_) {
        case rp::MatrixStack::modelview:  return mv_stack_;
        case rp::MatrixStack::projection: return proj_stack_;
        case rp::MatrixStack::texture:    return tex_stack_;
    }
    return mv_stack_;
}

void Renderer::MatrixMode(rp::MatrixStack s) { matrix_mode_ = s; }
void Renderer::MatrixSetIdentity() { stack().back() = glm::mat4(1); }
void Renderer::MatrixTranslate(float x, float y, float z) {
    stack().back() = glm::translate(stack().back(), glm::vec3(x,y,z));
}
void Renderer::MatrixRotate(float a, float x, float y, float z) {
    stack().back() = glm::rotate(stack().back(), a, glm::vec3(x,y,z));
}
void Renderer::MatrixScale(float x, float y, float z) {
    stack().back() = glm::scale(stack().back(), glm::vec3(x,y,z));
}
void Renderer::MatrixPerspective(float fovy, float asp, float zn, float zf) {
    stack().back() = stack().back() * glm::perspective(glm::radians(fovy), asp, zn, zf);
}
void Renderer::MatrixOrthogonal(float l, float r, float b, float t, float zn, float zf) {
    stack().back() = stack().back() * glm::ortho(l, r, b, t, zn, zf);
}
void Renderer::MatrixPush() { auto& s = stack(); s.push_back(s.back()); }
void Renderer::MatrixPop()  { auto& s = stack(); if (s.size() > 1) s.pop_back(); }
void Renderer::MatrixMult(float* m) {
    glm::mat4 mat; std::memcpy(&mat[0][0], m, 64);
    stack().back() = stack().back() * mat;
}
const float* Renderer::MatrixGet(rp::MatrixStack s) {
    switch (s) {
        case rp::MatrixStack::modelview:  return &mv_stack_.back()[0][0];
        case rp::MatrixStack::projection: return &proj_stack_.back()[0][0];
        case rp::MatrixStack::texture:    return &tex_stack_.back()[0][0];
    }
    return &mv_stack_.back()[0][0];
}

// ===================================================================
// STATE
// ===================================================================

void Renderer::StateSetColour(float r, float g, float b, float a) { state_colour_ = {r,g,b,a}; }
void Renderer::StateSetDepthMask(bool e) { if (pso_key_.depth_write != e) { pso_key_.depth_write = e; pso_dirty_ = true; } }
void Renderer::StateSetBlendEnable(bool e) { if (pso_key_.blend_enable != e) { pso_key_.blend_enable = e; pso_dirty_ = true; } }
void Renderer::StateSetBlendFunc(rp::BlendFactor s, rp::BlendFactor d) {
    uint8_t vs = blend_to_vk(s), vd = blend_to_vk(d);
    if (pso_key_.blend_src != vs || pso_key_.blend_dst != vd) {
        pso_key_.blend_src = vs; pso_key_.blend_dst = vd; pso_dirty_ = true;
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
    if (pso_key_.depth_func != v) { pso_key_.depth_func = v; pso_dirty_ = true; }
}
void Renderer::StateSetFaceCull(bool e) { if (pso_key_.cull_back != e) { pso_key_.cull_back = e; pso_dirty_ = true; } }
void Renderer::StateSetWriteEnable(bool r, bool g, bool b, bool a) {
    uint8_t m = (r?1:0)|(g?2:0)|(b?4:0)|(a?8:0);
    if (pso_key_.color_mask != m) { pso_key_.color_mask = m; pso_dirty_ = true; }
}
void Renderer::StateSetDepthTestEnable(bool e) { if (pso_key_.depth_test != e) { pso_key_.depth_test = e; pso_dirty_ = true; } }
void Renderer::StateSetAlphaTestEnable(bool e) { alpha_test_enabled_ = e; }
void Renderer::StateSetDepthSlopeAndBias(float slope, float bias) {
    depth_bias_slope_ = slope; depth_bias_constant_ = bias;
}
void Renderer::StateSetLightDirection(int idx, float x, float y, float z) {
    glm::vec3 d = glm::normalize(glm::mat3(mv_stack_.back()) * glm::vec3(x,y,z));
    if (idx == 0) light0_dir_eye_ = d; else light1_dir_eye_ = d;
}
void Renderer::StateSetTextureEnable(bool e) { if (active_tex_unit_ == 0) texture_enabled_ = e; }
void Renderer::StateSetActiveTexture(int gl) { active_tex_unit_ = (gl == 0x84C1) ? 1 : 0; }
void Renderer::UpdateGamma(unsigned short g) {
    float gamma = 0.5f + float(g) / 32768.0f;
    if (gamma < 0.01f) gamma = 0.01f;
    inv_gamma_ = 1.0f / gamma;
}

// ===================================================================
// PUSH CONSTANTS HELPER
// ===================================================================

void Renderer::fill_push_constants(void* out, bool textured, const glm::vec4* tint) {
    PushConstants& pc = *static_cast<PushConstants*>(out);
    const auto& mv = mv_stack_.back();
    pc.mvp = proj_stack_.back() * mv;
    // Y-flip for Vulkan NDC: negate ROW 1 (not column 1)
    for (int c = 0; c < 4; ++c) pc.mvp[c][1] = -pc.mvp[c][1];

    glm::mat3 nm(mv);
    const auto& tm = tex_stack_.back();
    pc.nm0       = glm::vec4(nm[0], tm[0][0]);
    pc.nm1       = glm::vec4(nm[1], tm[1][1]);
    pc.nm2       = glm::vec4(nm[2], tm[3][0]);
    pc.chunk_lit = glm::vec4(chunk_offset_[0], chunk_offset_[1],
                             chunk_offset_[2], lighting_enabled_ ? 1.0f : 0.0f);
    pc.l0        = glm::vec4(light0_dir_eye_, tm[3][1]);
    pc.l1        = glm::vec4(light1_dir_eye_, mv[3][0]);
    pc.ldiff     = glm::vec4(light_diffuse_,  mv[3][1]);
    pc.lamb      = glm::vec4(light_ambient_,  mv[3][2]);

    float fog_mode_f = 0;
    if (fog_enabled_) {
        switch (fog_mode_) {
            case rp::FogMode::linear:         fog_mode_f = 1; break;
            case rp::FogMode::exponential:    fog_mode_f = 2; break;
            case rp::FogMode::exponential_sq: fog_mode_f = 3; break;
            default: break;
        }
    }
    pc.fog_params = glm::vec4(fog_mode_f, fog_start_, fog_end_, fog_density_);

    if (tint) {
        pc.state_colour = glm::vec4(state_colour_[0] * (*tint)[0],
                                    state_colour_[1] * (*tint)[1],
                                    state_colour_[2] * (*tint)[2],
                                    state_colour_[3] * (*tint)[3]);
    } else {
        pc.state_colour = glm::vec4(state_colour_[0], state_colour_[1],
                                    state_colour_[2], state_colour_[3]);
    }
    pc.fog_colour = glm::vec4(fog_colour_[0], fog_colour_[1],
                              fog_colour_[2], fog_colour_[3]);
    pc.alpha_ref  = alpha_ref_;
    pc.inv_gamma  = inv_gamma_;
    pc.flags      = ((textured && texture_enabled_) ? 1u : 0u) |
                    (alpha_test_enabled_ ? 2u : 0u);
    pc.global_lm_packed = uint32_t(global_lm_uv_[0]) | (uint32_t(global_lm_uv_[1]) << 16);
}

// ===================================================================
// TEXTURE BINDING HELPER
// ===================================================================

Renderer::BoundTexResult Renderer::bind_textures(VkCommandBuffer cmd) {
    VkDescriptorSet ds = VK_NULL_HANDLE;
    bool textured = false;
    bool lm_active = false;
    {
        std::lock_guard lk(texture_mutex_);
        int tex = (bound_tex_ > 0 && size_t(bound_tex_) < textures_.size() &&
                   textures_[bound_tex_].ready) ? bound_tex_ : default_tex_;
        auto& ts = textures_[tex];
        if (ts.sampler_dirty) { update_tex_descriptor(ts); ts.sampler_dirty = false; }
        ds = ts.desc_set;
        textured = (tex != default_tex_);
        // Update lightmap binding only when the bound lightmap actually changes.
        // bound_lm is invalidated (-1) whenever update_tex_descriptor or
        // upload_texture overwrites the descriptor set's binding 1.
        if (ds && lightmap_tex_ > 0 && size_t(lightmap_tex_) < textures_.size() &&
            textures_[lightmap_tex_].ready) {
            if (ts.bound_lm != lightmap_tex_) {
                auto& lm = textures_[lightmap_tex_];
                VkSampler lm_sampler = get_or_create_sampler(lm_sampler_key_);
                VkDescriptorImageInfo lm_dii{lm_sampler, lm.view,
                                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                VkWriteDescriptorSet wd{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                wd.dstSet = ds; wd.dstBinding = 1; wd.descriptorCount = 1;
                wd.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                wd.pImageInfo = &lm_dii;
                vkUpdateDescriptorSets(dev_.handle(), 1, &wd, 0, nullptr);
                ts.bound_lm = lightmap_tex_;
            }
            lm_active = true;
        }
    }
    if (ds) vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     pipeline_layout_, 0, 1, &ds, 0, nullptr);
    return {ds, textured, lm_active};
}

// ===================================================================
// DRAW
// ===================================================================

void Renderer::DrawVertices(int primType, int count, void* data,
                            int vType, int /*sType*/) {
    if (count <= 0 || !data) return;

    // CBuff recording must be checked BEFORE frame_active_ — worker
    // threads rebuild chunks between frames when frame_active_ is false.
    // Recording just copies to CPU memory, no GPU state needed.
    if (t_rec.id >= 0) {
        constexpr uint32_t kStd = 32;
        DisplayListDraw d;
        d.primType = primType;
        d.vertexType = vType;
        d.shaderType = 0;
        size_t bytes = (vType == 1) ? size_t(count) * 16 : size_t(count) * kStd;
        d.verts.resize(bytes);
        std::memcpy(d.verts.data(), data, bytes);
        t_rec.draws.push_back(std::move(d));
        return;
    }

    if (!frame_active_) return;
    ensure_pass();

    // Expand compact format (vType==1) to world_standard 32-byte
    const void* vdata = data;
    std::vector<std::byte> expanded;
    if (vType == 1) {
        expanded = expand_compact(data, count);
        vdata = expanded.data();
        primType = 0x0004;  // already triangulated
    }

    // Convert triangle fans to triangle list on CPU
    std::vector<std::byte> fan_expanded;
    if (primType == 0x0006) {
        fan_expanded = fan_to_list(vdata, count);
        if (count == 0) return;
        vdata = fan_expanded.data();
        primType = 0x0004;
    }

    VkPrimitiveTopology topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    bool is_quads = false, is_lines = false;
    switch (primType) {
        case 0x0001: topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;      is_lines = true; break;
        case 0x0003: topo = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;     is_lines = true; break;
        case 0x0004: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;  break;
        case 0x0005: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
        case 0x0007: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;  is_quads = true; break;
        default: return;
    }
    if (is_quads && (count % 4) != 0) return;
    if (pso_key_.lines != is_lines) { pso_key_.lines = is_lines; pso_dirty_ = true; }

    auto& f = frame();
    constexpr VkDeviceSize stride = 32;
    VkDeviceSize bytes = VkDeviceSize(count) * stride;
    void* dst = f.alloc_transient(bytes);
    if (!dst) return;
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
    vkCmdSetBlendConstants(f.cmd, blend_constants_.data());

    // Dynamic depth bias
    vkCmdSetDepthBias(f.cmd, depth_bias_constant_, 0.0f, depth_bias_slope_);

    auto [ds, textured, lm_active] = bind_textures(f.cmd);

    vkCmdBindVertexBuffers(f.cmd, 0, 1, &f.transient_vb, &off);

    PushConstants pc{};
    fill_push_constants(&pc, textured);
    if (lm_active) pc.flags |= 4u;
    vkCmdPushConstants(f.cmd, pipeline_layout_,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, 256, &pc);

    if (is_quads) {
        uint32_t qc = std::min(uint32_t(count) / 4, kMaxQuads);
        vkCmdBindIndexBuffer(f.cmd, quad_ib_, 0, VK_INDEX_TYPE_UINT32);
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
    auto fn = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
        vkGetInstanceProcAddr(dev_.instance(), "vkCmdBeginDebugUtilsLabelEXT"));
    if (fn && frame_active_) {
        VkDebugUtilsLabelEXT l{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
        l.pLabelName = name;
        fn(frame().cmd, &l);
    }
#else
    (void)name;
#endif
}

void Renderer::pop_debug_event() {
#ifndef NDEBUG
    auto fn = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
        vkGetInstanceProcAddr(dev_.instance(), "vkCmdEndDebugUtilsLabelEXT"));
    if (fn && frame_active_) fn(frame().cmd);
#endif
}

// ===================================================================
// TEXTURES
// ===================================================================

int Renderer::TextureCreate() {
    std::lock_guard lk(texture_mutex_);
    textures_.emplace_back();
    return int(textures_.size()) - 1;
}

void Renderer::TextureFree(int idx) {
    std::lock_guard lk(texture_mutex_);
    if (idx <= 0 || size_t(idx) >= textures_.size()) return;
    if (idx == default_tex_) return;
    auto& t = textures_[idx];
    if (t.view || t.image) {
        auto view = t.view; auto img = t.image; auto alloc = t.alloc;
        auto dev = dev_.handle(); auto vma = dev_.allocator();
        frame().deletions.push([=]() {
            if (view) vkDestroyImageView(dev, view, nullptr);
            if (img)  vmaDestroyImage(vma, img, alloc);
        });
    }
    t = {};
}

void Renderer::TextureBind(int idx) {
    std::lock_guard lk(texture_mutex_);
    if (idx < 0) { bound_tex_ = default_tex_; return; }
    if (size_t(idx) >= textures_.size()) textures_.resize(idx + 1);
    bound_tex_ = idx;
}

void Renderer::TextureData(int w, int h, void* data, int level, int) {
    if (level != 0 || !data) return;
    int idx;
    { std::lock_guard lk(texture_mutex_); idx = bound_tex_; if (idx <= 0) return;
      if (size_t(idx) >= textures_.size()) textures_.resize(idx + 1); }

    upload_texture(idx, w, h, data);
}

void Renderer::TextureDataUpdate(int xo, int yo, int w, int h, void* data, int level) {
    if (level != 0 || !data) return;
    int idx;
    { std::lock_guard lk(texture_mutex_); idx = bound_tex_; if (idx <= 0 || size_t(idx) >= textures_.size()) return; }
    auto& t = textures_[idx];
    if (!t.ready) return;
    if (xo == 0 && yo == 0 && uint32_t(w) == t.width && uint32_t(h) == t.height) {
        // Full rewrite — use existing upload path
        upload_texture(idx, w, h, data);
    } else if (xo >= 0 && yo >= 0 && uint32_t(xo + w) <= t.width && uint32_t(yo + h) <= t.height) {
        // Partial sub-region update
        VkDeviceSize bytes = VkDeviceSize(w) * h * 4;
        if (bytes > kStagingSize) return;
        std::memcpy(staging_mapped_, data, bytes);

        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = upload_pool_; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        VkCommandBuffer cmd; check(vkAllocateCommandBuffers(dev_.handle(), &cai, &cmd), "tex update cmd");
        VkCommandBufferBeginInfo bbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bbi);

        // Transition to TRANSFER_DST for the sub-region copy
        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        b.srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        b.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.image = t.image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = 1; dep.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &dep);

        VkBufferImageCopy rgn{};
        rgn.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        rgn.imageOffset = {xo, yo, 0};
        rgn.imageExtent = {uint32_t(w), uint32_t(h), 1};
        vkCmdCopyBufferToImage(cmd, staging_buf_, t.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &rgn);

        // Transition back to SHADER_READ_ONLY
        b.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier2(cmd, &dep);

        vkEndCommandBuffer(cmd);
        VkCommandBufferSubmitInfo csi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        csi.commandBuffer = cmd;
        VkSubmitInfo2 sub{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        sub.commandBufferInfoCount = 1; sub.pCommandBufferInfos = &csi;
        vkQueueSubmit2(dev_.queue(), 1, &sub, VK_NULL_HANDLE);
        vkQueueWaitIdle(dev_.queue());
        vkResetCommandBuffer(cmd, 0);
    }
}

void Renderer::TextureSetParam(int param, int value) {
    std::lock_guard lk(texture_mutex_);
    if (bound_tex_ <= 0 || size_t(bound_tex_) >= textures_.size()) return;
    auto& t = textures_[bound_tex_];

    constexpr int GL_TEXTURE_MIN_FILTER = 0x2801;
    constexpr int GL_TEXTURE_MAG_FILTER = 0x2800;
    constexpr int GL_TEXTURE_WRAP_S     = 0x2802;
    constexpr int GL_TEXTURE_WRAP_T     = 0x2803;
    constexpr int GL_NEAREST            = 0x2600;
    constexpr int GL_LINEAR             = 0x2601;
    constexpr int GL_NEAREST_MIPMAP_NEAREST = 0x2700;
    constexpr int GL_LINEAR_MIPMAP_NEAREST  = 0x2701;
    constexpr int GL_NEAREST_MIPMAP_LINEAR  = 0x2702;
    constexpr int GL_LINEAR_MIPMAP_LINEAR   = 0x2703;
    constexpr int GL_CLAMP_TO_EDGE = 0x812F;

    switch (param) {
        case GL_TEXTURE_MIN_FILTER:
            switch (value) {
                case GL_NEAREST:
                    t.sampler_key.min_filter = VK_FILTER_NEAREST;
                    t.sampler_key.mip_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
                    break;
                case GL_LINEAR:
                    t.sampler_key.min_filter = VK_FILTER_LINEAR;
                    t.sampler_key.mip_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
                    break;
                case GL_NEAREST_MIPMAP_NEAREST:
                    t.sampler_key.min_filter = VK_FILTER_NEAREST;
                    t.sampler_key.mip_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
                    break;
                case GL_LINEAR_MIPMAP_NEAREST:
                    t.sampler_key.min_filter = VK_FILTER_LINEAR;
                    t.sampler_key.mip_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
                    break;
                case GL_NEAREST_MIPMAP_LINEAR:
                    t.sampler_key.min_filter = VK_FILTER_NEAREST;
                    t.sampler_key.mip_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
                    break;
                case GL_LINEAR_MIPMAP_LINEAR:
                    t.sampler_key.min_filter = VK_FILTER_LINEAR;
                    t.sampler_key.mip_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
                    break;
            }
            t.sampler_dirty = true;
            break;

        case GL_TEXTURE_MAG_FILTER:
            t.sampler_key.mag_filter = (value == GL_NEAREST)
                ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
            t.sampler_dirty = true;
            break;

        case GL_TEXTURE_WRAP_S:
            t.sampler_key.wrap_s = (value == GL_CLAMP_TO_EDGE)
                ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
                : VK_SAMPLER_ADDRESS_MODE_REPEAT;
            t.sampler_dirty = true;
            break;

        case GL_TEXTURE_WRAP_T:
            t.sampler_key.wrap_t = (value == GL_CLAMP_TO_EDGE)
                ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
                : VK_SAMPLER_ADDRESS_MODE_REPEAT;
            t.sampler_dirty = true;
            break;
    }
}

int Renderer::ensure_default_texture() {
    int idx = TextureCreate();
    uint32_t pixel = 0xFFFFFFFF;
    { std::lock_guard lk(texture_mutex_); bound_tex_ = idx; }
    upload_texture(idx, 1, 1, &pixel);
    return idx;
}

int Renderer::ensure_default_lightmap() {
    int idx = TextureCreate();
    uint32_t pixel = 0xFFFFFFFF;
    { std::lock_guard lk(texture_mutex_); bound_tex_ = idx; }
    upload_texture(idx, 1, 1, &pixel);
    return idx;
}

void Renderer::update_tex_descriptor(TextureSlot& t) {
    if (!t.desc_set || !t.view) return;
    VkSampler sampler = get_or_create_sampler(t.sampler_key);
    VkDescriptorImageInfo dii{sampler, t.view,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet writes[2]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = t.desc_set; writes[0].dstBinding = 0; writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &dii;
    uint32_t n = 1;
    // Also write binding 1 with default lightmap fallback
    VkDescriptorImageInfo lm_dii{};
    if (default_lm_ > 0 && size_t(default_lm_) < textures_.size() && textures_[default_lm_].view) {
        auto& lm = textures_[default_lm_];
        VkSampler lm_sampler = get_or_create_sampler(lm_sampler_key_);
        lm_dii = {lm_sampler, lm.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = t.desc_set; writes[1].dstBinding = 1; writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].pImageInfo = &lm_dii;
        n = 2;
    }
    vkUpdateDescriptorSets(dev_.handle(), n, writes, 0, nullptr);
    // Binding 1 was overwritten with default lightmap — invalidate cache
    t.bound_lm = -1;
}

void Renderer::upload_texture(int idx, int w, int h, const void* pixels) {
    auto& t = textures_[idx];
    bool reuse = t.ready && t.width == uint32_t(w) && t.height == uint32_t(h);
    if (t.ready && !reuse) {
        auto view = t.view; auto img = t.image; auto alloc = t.alloc;
        auto dev = dev_.handle(); auto vma = dev_.allocator();
        frame().deletions.push([=]() {
            if (view) vkDestroyImageView(dev, view, nullptr);
            if (img)  vmaDestroyImage(vma, img, alloc);
        });
        VkDescriptorSet keep = t.desc_set;
        SamplerKey sk = t.sampler_key;
        t = {}; t.desc_set = keep; t.sampler_key = sk;
    }
    t.width = w; t.height = h;

    uint32_t mips = 1;
    { uint32_t d = std::max(uint32_t(w), uint32_t(h)); while (d > 1) { d >>= 1; ++mips; } }

    if (!reuse) {
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format    = VK_FORMAT_R8G8B8A8_UNORM;
        ici.extent    = {t.width, t.height, 1};
        ici.mipLevels = mips; ici.arrayLayers = 1;
        ici.samples   = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling    = VK_IMAGE_TILING_OPTIMAL;
        ici.usage     = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                        VK_IMAGE_USAGE_SAMPLED_BIT;
        VmaAllocationCreateInfo ai{}; ai.usage = VMA_MEMORY_USAGE_AUTO;
        check(vmaCreateImage(dev_.allocator(), &ici, &ai, &t.image, &t.alloc, nullptr), "texture image");

        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = t.image; vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_R8G8B8A8_UNORM;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 1};
        check(vkCreateImageView(dev_.handle(), &vci, nullptr, &t.view), "texture view");
    }

    if (!t.desc_set) {
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = tex_pool_; dai.descriptorSetCount = 1;
        dai.pSetLayouts = &tex_set_layout_;
        check(vkAllocateDescriptorSets(dev_.handle(), &dai, &t.desc_set), "texture desc set");
    }
    // Write descriptor with per-texture sampler (binding 0 = diffuse, binding 1 = lightmap fallback)
    {
        VkSampler sampler = get_or_create_sampler(t.sampler_key);
        VkDescriptorImageInfo dii{sampler, t.view,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet writes[2]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = t.desc_set; writes[0].dstBinding = 0; writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &dii;
        uint32_t n = 1;
        VkDescriptorImageInfo lm_dii{};
        if (default_lm_ > 0 && size_t(default_lm_) < textures_.size() && textures_[default_lm_].view) {
            auto& lm = textures_[default_lm_];
            VkSampler lm_sampler = get_or_create_sampler(lm_sampler_key_);
            lm_dii = {lm_sampler, lm.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = t.desc_set; writes[1].dstBinding = 1; writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[1].pImageInfo = &lm_dii;
            n = 2;
        }
        vkUpdateDescriptorSets(dev_.handle(), n, writes, 0, nullptr);
    }
    t.sampler_dirty = false;
    t.bound_lm = -1;  // binding 1 was overwritten with default lightmap

    // Stage + copy via upload pool
    VkDeviceSize bytes = VkDeviceSize(w) * h * 4;
    if (bytes > kStagingSize) return;

    std::memcpy(staging_mapped_, pixels, bytes);

    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = upload_pool_; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd; check(vkAllocateCommandBuffers(dev_.handle(), &cai, &cmd), "texture upload cmd");
    VkCommandBufferBeginInfo bbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bbi);

    // UNDEFINED -> TRANSFER_DST
    {
        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        b.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.image = t.image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 1};
        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = 1; dep.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &dep);
    }
    VkBufferImageCopy rgn{};
    rgn.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    rgn.imageExtent = {t.width, t.height, 1};
    vkCmdCopyBufferToImage(cmd, staging_buf_, t.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &rgn);

    // Mip generation
    int32_t mw = w, mh = h;
    for (uint32_t i = 1; i < mips; ++i) {
        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
        b.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.image = t.image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, i-1, 1, 0, 1};
        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = 1; dep.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &dep);

        int32_t nw = std::max(mw/2, 1), nh = std::max(mh/2, 1);
        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i-1, 0, 1};
        blit.srcOffsets[1] = {mw, mh, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1};
        blit.dstOffsets[1] = {nw, nh, 1};
        vkCmdBlitImage(cmd, t.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &blit, VK_FILTER_LINEAR);
        mw = nw; mh = nh;
    }

    // All mips -> SHADER_READ_ONLY
    {
        VkImageMemoryBarrier2 ends[2]{}; uint32_t n = 0;
        if (mips > 1) {
            ends[n].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            ends[n].srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
            ends[n].srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            ends[n].dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            ends[n].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            ends[n].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            ends[n].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            ends[n].image = t.image;
            ends[n].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips-1, 0, 1};
            ++n;
        }
        ends[n].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        ends[n].srcStageMask = (mips>1) ? VK_PIPELINE_STAGE_2_BLIT_BIT : VK_PIPELINE_STAGE_2_COPY_BIT;
        ends[n].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        ends[n].dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        ends[n].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        ends[n].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        ends[n].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ends[n].image = t.image;
        ends[n].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mips-1, 1, 0, 1};
        ++n;
        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = n; dep.pImageMemoryBarriers = ends;
        vkCmdPipelineBarrier2(cmd, &dep);
    }
    vkEndCommandBuffer(cmd);
    VkCommandBufferSubmitInfo csi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    csi.commandBuffer = cmd;
    VkSubmitInfo2 sub{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    sub.commandBufferInfoCount = 1; sub.pCommandBufferInfos = &csi;
    vkQueueSubmit2(dev_.queue(), 1, &sub, VK_NULL_HANDLE);
    vkQueueWaitIdle(dev_.queue());
    vkResetCommandBuffer(cmd, 0);
    t.ready = true;
}

int Renderer::LoadTextureData(const char* fn, void* srcInfo, int** out) {
    int w, h, c;
    unsigned char* px = stbi_load(fn, &w, &h, &c, 4);
    if (!px) return -1;
    if (auto* i = static_cast<D3DXIMAGE_INFO*>(srcInfo)) { i->Width = w; i->Height = h; }
    *out = stb_to_argb(px, w, h);
    stbi_image_free(px);
    return 0;
}

int Renderer::LoadTextureData(uint8_t* data, uint32_t bytes, void* srcInfo, int** out) {
    int w, h, c;
    unsigned char* px = stbi_load_from_memory(data, int(bytes), &w, &h, &c, 4);
    if (!px) return -1;
    if (auto* i = static_cast<D3DXIMAGE_INFO*>(srcInfo)) { i->Width = w; i->Height = h; }
    *out = stb_to_argb(px, w, h);
    stbi_image_free(px);
    return 0;
}

// ===================================================================
// CBUFF (DISPLAY LISTS)
// ===================================================================

int Renderer::CBuffCreate(int n) {
    std::lock_guard lk(display_list_mutex_);
    int first = next_display_list_;
    int needed = n > 0 ? n : 1;
    next_display_list_ += needed;
    if (size_t(first + needed) > display_lists_.size()) display_lists_.resize(first + needed);
    return first;
}

void Renderer::CBuffDeleteAll() {
    std::lock_guard lk(display_list_mutex_);
    { std::lock_guard lk2(pending_destroy_mutex_);
      for (auto& cb : display_lists_)
          if (cb.vb) pending_destroys_.push_back({cb.vb, cb.alloc}); }
    display_lists_.clear();
    next_display_list_ = 1;
    t_rec.id = -1; t_rec.draws.clear();
}

void Renderer::CBuffStart(int index, bool) { t_rec.id = index; t_rec.draws.clear(); }

void Renderer::CBuffClear(int index) {
    std::lock_guard lk(display_list_mutex_);
    if (index < 0 || size_t(index) >= display_lists_.size()) return;
    auto& cb = display_lists_[index];
    cb.draws.clear(); cb.gpu_draws.clear();
    if (cb.vb) {
        { std::lock_guard lk2(pending_destroy_mutex_);
          pending_destroys_.push_back({cb.vb, cb.alloc}); }
        cb.vb = VK_NULL_HANDLE; cb.alloc = nullptr; cb.vb_size = 0;
    }
    cb.valid = cb.uploaded = false;
}

int Renderer::CBuffSize(int index) {
    std::lock_guard lk(display_list_mutex_);
    if (index < 0 || size_t(index) >= display_lists_.size()) return 0;
    return display_lists_[index].valid ? 1 : 0;
}

void Renderer::CBuffEnd() {
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

void Renderer::display_list_upload(DisplayList& cb) {
    constexpr uint32_t kStride = 32;
    std::vector<std::byte> combined;
    cb.gpu_draws.clear();

    for (auto& d : cb.draws) {
        const void* src = d.verts.data();
        int vert_count;
        std::vector<std::byte> expanded_storage;

        if (d.vertexType == 1) {
            vert_count = int(d.verts.size() / 16);
            expanded_storage = expand_compact(src, vert_count);
            src = expanded_storage.data();
            // expand_compact already triangulates quads
            uint32_t byte_off = uint32_t(combined.size());
            combined.insert(combined.end(),
                            static_cast<const std::byte*>(src),
                            static_cast<const std::byte*>(src) + size_t(vert_count) * kStride);
            cb.gpu_draws.push_back({byte_off, uint32_t(vert_count), 0x0004});
            continue;
        }

        vert_count = int(d.verts.size() / kStride);
        if (vert_count == 0) continue;

        if (d.primType == 0x0007) {
            // Quads -> triangles
            if (vert_count % 4 != 0) continue;
            uint32_t quads = vert_count / 4;
            uint32_t byte_off = uint32_t(combined.size());
            for (uint32_t q = 0; q < quads; ++q) {
                const std::byte* b = d.verts.data() + q * 4 * kStride;
                auto push = [&](uint32_t i) {
                    combined.insert(combined.end(), b + i*kStride, b + (i+1)*kStride);
                };
                push(0); push(1); push(2); push(0); push(2); push(3);
            }
            cb.gpu_draws.push_back({byte_off, quads * 6, 0x0004});
        } else if (d.primType == 0x0006) {
            // Fan -> triangles
            if (vert_count < 3) continue;
            int fan_count = vert_count;
            auto fan_data = fan_to_list(d.verts.data(), fan_count);
            if (fan_count == 0) continue;
            uint32_t byte_off = uint32_t(combined.size());
            combined.insert(combined.end(), fan_data.begin(), fan_data.end());
            cb.gpu_draws.push_back({byte_off, uint32_t(fan_count), 0x0004});
        } else {
            uint32_t byte_off = uint32_t(combined.size());
            combined.insert(combined.end(), d.verts.begin(), d.verts.end());
            cb.gpu_draws.push_back({byte_off, uint32_t(vert_count), d.primType});
        }
    }
    if (combined.empty()) { cb.uploaded = false; return; }

    uint32_t needed = uint32_t(combined.size());

    // Always allocate a new host-visible buffer to avoid data races with
    // the GPU reading the previous frame's data. Old buffer is deferred-
    // destroyed after the GPU is done. Direct memcpy eliminates staging
    // buffer, command buffer submission, and vkQueueWaitIdle entirely.
    if (cb.vb)
        frame().deletions.push_buffer(dev_.allocator(), cb.vb, cb.alloc);

    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size  = needed;
    bi.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
               VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info{};
    check(vmaCreateBuffer(dev_.allocator(), &bi, &ai, &cb.vb, &cb.alloc, &info),
          "display list vb");
    cb.vb_size = needed;

    std::memcpy(info.pMappedData, combined.data(), needed);
    cb.uploaded = true;
}

bool Renderer::CBuffCall(int index, bool) {
    if (index < 0 || !frame_active_) return false;

    VkBuffer vb = VK_NULL_HANDLE;
    std::vector<DisplayListSubDraw> draws;
    {
        std::lock_guard lk(display_list_mutex_);
        if (size_t(index) >= display_lists_.size()) return false;
        auto& cb = display_lists_[index];
        if (!cb.valid || cb.draws.empty()) return false;
        if (!cb.uploaded) { display_list_upload(cb); if (!cb.uploaded) return false; }
        vb = cb.vb; draws = cb.gpu_draws;
    }

    auto& f = frame();
    ensure_pass();

    if (pso_dirty_ || !(pso_key_ == last_bound_pso_)) {
        vkCmdBindPipeline(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipelines_.get(pso_key_));
        last_bound_pso_ = pso_key_; pso_dirty_ = false;
    }
    vkCmdSetBlendConstants(f.cmd, blend_constants_.data());
    vkCmdSetDepthBias(f.cmd, depth_bias_constant_, 0.0f, depth_bias_slope_);

    auto [ds, textured, lm_active] = bind_textures(f.cmd);

    PushConstants pc{};
    fill_push_constants(&pc, textured);
    if (lm_active) pc.flags |= 4u;

    vkCmdPushConstants(f.cmd, pipeline_layout_,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, 256, &pc);

    for (auto& sd : draws) {
        VkDeviceSize off = sd.vertex_offset;
        vkCmdBindVertexBuffers(f.cmd, 0, 1, &vb, &off);
        VkPrimitiveTopology topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        switch (sd.prim_type) {
            case 0x0001: topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; break;
            case 0x0003: topo = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; break;
            case 0x0005: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
        }
        vkCmdSetPrimitiveTopology(f.cmd, topo);
        vkCmdDraw(f.cmd, sd.vertex_count, 1, 0, 0);
    }

    depth_bias_constant_ = 0;
    depth_bias_slope_    = 0;
    return true;
}

// ===================================================================
// MATERIALS + TRANSIENT + SUBMIT_IMMEDIATE
// ===================================================================

rp::MaterialHandle Renderer::create_material(const rp::MaterialDesc&) {
    return {++next_material_id_, 1};
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
    vkCmdSetBlendConstants(f.cmd, blend_constants_.data());
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

    auto [ds, textured, lm_active] = bind_textures(f.cmd);

    VkDeviceSize off = tvb.offset;
    vkCmdBindVertexBuffers(f.cmd, 0, 1, &f.transient_vb, &off);

    PushConstants pc{};
    glm::vec4 tint(dc.tint_color[0], dc.tint_color[1],
                   dc.tint_color[2], dc.tint_color[3]);
    fill_push_constants(&pc, textured, &tint);
    if (lm_active) pc.flags |= 4u;
    vkCmdPushConstants(f.cmd, pipeline_layout_,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, 256, &pc);

    vkCmdDraw(f.cmd, tvb.vertex_count, 1, 0, 0);

    depth_bias_constant_ = 0;
    depth_bias_slope_    = 0;
}

}  // namespace plce::vk3

// ===================================================================
// FACTORY
// ===================================================================

std::unique_ptr<rp::IRenderPath> make_vulkan_render_path(SDL_Window* window) {
    return std::make_unique<plce::vk3::Renderer>(window);
}
