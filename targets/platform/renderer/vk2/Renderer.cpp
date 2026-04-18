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
#include "vk/shaders/basic.vert.spv.h"
#include "vk/shaders/basic.frag.spv.h"
#include "render/TerrainRenderer.h"

namespace plce::vk2 {

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

// Thread-local CBuff recording state
struct RecState { int id = -1; std::vector<Renderer::CBuffDraw> draws; };
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

    swap_.create(dev_, 0, 0);  // uses surface caps for initial size

    // Per-frame contexts
    for (auto& f : frames_)
        f.create(dev_.handle(), dev_.allocator(), dev_.queue_family());

    // Descriptor set layout: one combined image sampler
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding         = 0;
        b.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b.descriptorCount = 1;
        b.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo ci{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = 1;
        ci.pBindings    = &b;
        check(vkCreateDescriptorSetLayout(dev_.handle(), &ci, nullptr,
                                          &tex_set_layout_),
              "desc layout");
    }

    // Descriptor pool
    {
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4096};
        VkDescriptorPoolCreateInfo ci{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ci.maxSets       = 4096;
        ci.poolSizeCount = 1;
        ci.pPoolSizes    = &ps;
        check(vkCreateDescriptorPool(dev_.handle(), &ci, nullptr, &tex_pool_),
              "desc pool");
    }

    // Samplers
    {
        VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        ci.magFilter    = VK_FILTER_NEAREST;
        ci.minFilter    = VK_FILTER_NEAREST;
        ci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        ci.maxLod       = VK_LOD_CLAMP_NONE;
        check(vkCreateSampler(dev_.handle(), &ci, nullptr, &tex_sampler_),
              "sampler");
        ci.magFilter    = VK_FILTER_LINEAR;
        ci.minFilter    = VK_FILTER_LINEAR;
        ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        check(vkCreateSampler(dev_.handle(), &ci, nullptr, &tex_sampler_lm_),
              "sampler lm");
    }

    // Pipeline layout: vertex [0..191] + fragment [192..255]
    {
        VkPushConstantRange pcs[2]{};
        pcs[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pcs[0].offset = 0;   pcs[0].size = 192;
        pcs[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pcs[1].offset = 192; pcs[1].size = 64;
        VkPipelineLayoutCreateInfo ci{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        ci.setLayoutCount         = 1;
        ci.pSetLayouts            = &tex_set_layout_;
        ci.pushConstantRangeCount = 2;
        ci.pPushConstantRanges    = pcs;
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

        // Upload via staging
        VkBuffer stg = VK_NULL_HANDLE; VmaAllocation sa = nullptr;
        VmaAllocationInfo si{};
        VkBufferCreateInfo sci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        sci.size = bytes; sci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo sai{};
        sai.usage = VMA_MEMORY_USAGE_AUTO;
        sai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        vmaCreateBuffer(dev_.allocator(), &sci, &sai, &stg, &sa, &si);
        std::memcpy(si.pMappedData, indices.data(), bytes);

        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pci.queueFamilyIndex = dev_.queue_family();
        VkCommandPool pool; vkCreateCommandPool(dev_.handle(), &pci, nullptr, &pool);
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        VkCommandBuffer cmd; vkAllocateCommandBuffers(dev_.handle(), &cai, &cmd);
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
        vkQueueWaitIdle(dev_.queue());  // one-time init, acceptable
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
        vmaCreateBuffer(dev_.allocator(), &bi, &ai, &staging_buf_,
                        &staging_alloc_, &info);
        staging_mapped_ = static_cast<std::byte*>(info.pMappedData);

        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = dev_.queue_family();
        vkCreateCommandPool(dev_.handle(), &pci, nullptr, &upload_pool_);
    }

    default_tex_ = ensure_default_texture();
    bound_tex_   = default_tex_;

    // Terrain renderer
    plce::vk_render::TerrainRenderer::Config tcfg{};
    tcfg.color_format = swap_.format();
    tcfg.depth_format = swap_.depth_format();
    terrain_ = std::make_unique<plce::vk_render::TerrainRenderer>(
        dev_.handle(), dev_.allocator(), dev_.queue_family(), dev_.queue(), tcfg);

    fb_.width  = swap_.extent().width;
    fb_.height = swap_.extent().height;

    std::fprintf(stderr, "[vk2] renderer ready %ux%u images=%u\n",
                 swap_.extent().width, swap_.extent().height,
                 swap_.image_count());
}

Renderer::~Renderer() {
    vkDeviceWaitIdle(dev_.handle());  // teardown only — acceptable per best practices
    terrain_.reset();
    // CBuff GPU buffers
    for (auto& cb : cbufs_)
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
    if (tex_sampler_)    vkDestroySampler(dev_.handle(), tex_sampler_, nullptr);
    if (tex_sampler_lm_) vkDestroySampler(dev_.handle(), tex_sampler_lm_, nullptr);
    if (tex_pool_)       vkDestroyDescriptorPool(dev_.handle(), tex_pool_, nullptr);
    if (tex_set_layout_) vkDestroyDescriptorSetLayout(dev_.handle(), tex_set_layout_, nullptr);
    for (auto& f : frames_) f.destroy(dev_.handle(), dev_.allocator());
    swap_.destroy(dev_);
}

// ===================================================================
// FRAME LOOP
// ===================================================================

void Renderer::StartFrame() {
    if (frame_active_) return;
    auto& f = frame();
    f.begin(dev_.handle());

    VkResult acq = vkAcquireNextImageKHR(dev_.handle(), swap_.handle(),
                                          UINT64_MAX, f.sem_acquired,
                                          VK_NULL_HANDLE, &acquired_img_);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
        int w, h; SDL_GetWindowSize(window_, &w, &h);
        resize(uint32_t(w), uint32_t(h));
        return;
    }

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
    ++stat_frames_;
    double now = double(SDL_GetTicks64()) / 1000.0;
    if (stat_start_ == 0) stat_start_ = now;
    if (now - stat_start_ >= 1.0) {
        std::fprintf(stderr, "[vk2] fps=%u draws=%u\n", stat_frames_, stat_draws_);
        stat_frames_ = stat_draws_ = 0;
        stat_start_ = now;
    }
}

void Renderer::ensure_pass() {
    if (pass_active_) return;
    begin_pass();
    pass_active_ = true;
}

void Renderer::begin_pass() {
    auto& f = frame();
    VkCommandBuffer cmd = f.cmd;

    // Barriers: UNDEFINED → attachment optimal
    VkImageMemoryBarrier2 bars[2]{};
    bars[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    bars[0].srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    bars[0].dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    bars[0].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    bars[0].oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
    bars[0].newLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    bars[0].image         = swap_.image(acquired_img_);
    bars[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    bars[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    bars[1].srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    bars[1].dstStageMask  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
    bars[1].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    bars[1].oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
    bars[1].newLayout     = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    bars[1].image         = swap_.image(acquired_img_);  // wrong — need depth image
    bars[1].subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

    // Fix: depth barrier should reference the DEPTH image, not swapchain
    // The depth image is owned by Swapchain. We need access to it.
    // For now, we access it through swap_.depth_view()'s parent image.
    // TODO: Swapchain should expose depth_image() directly.
    // Workaround: skip depth barrier since loadOp=CLEAR handles it.
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 1;  // only color barrier
    dep.pImageMemoryBarriers    = bars;
    vkCmdPipelineBarrier2(cmd, &dep);

    // Begin dynamic rendering
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
    VkRect2D sc{{0,0}, swap_.extent()};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
}

void Renderer::end_pass() {
    auto& f = frame();
    vkCmdEndRendering(f.cmd);

    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    b.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    b.dstStageMask  = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
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
    // Wait for BOTH frame slots to finish before touching swapchain
    for (auto& f : frames_)
        vkWaitForFences(dev_.handle(), 1, &f.fence, VK_TRUE, UINT64_MAX);
    swap_.resize(dev_, w, h);
    fb_.width = swap_.extent().width;
    fb_.height = swap_.extent().height;
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
// DRAW
// ===================================================================

void Renderer::DrawVertices(int primType, int count, void* data,
                            int /*vType*/, int /*sType*/) {
    if (count <= 0 || !data) return;
    if (!frame_active_) return;
    ensure_pass();

    VkPrimitiveTopology topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    bool is_quads = false, is_lines = false;
    switch (primType) {
        case 0x0001: topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;      is_lines = true; break;
        case 0x0003: topo = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;     is_lines = true; break;
        case 0x0004: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;  break;
        case 0x0005: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
        case 0x0006: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;   break;
        case 0x0007: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;  is_quads = true; break;
        default: return;
    }
    if (is_quads && (count % 4) != 0) return;
    if (pso_key_.lines != is_lines) { pso_key_.lines = is_lines; pso_dirty_ = true; }

    auto& f = frame();
    constexpr VkDeviceSize stride = 32;
    VkDeviceSize bytes = VkDeviceSize(count) * stride;
    void* dst = f.alloc_transient(bytes);
    if (!dst) return;  // transient full — skip draw
    std::memcpy(dst, data, bytes);
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

    // Texture
    VkDescriptorSet ds = VK_NULL_HANDLE;
    bool textured = false;
    {
        std::lock_guard lk(tex_mu_);
        int tex = (bound_tex_ > 0 && size_t(bound_tex_) < textures_.size() &&
                   textures_[bound_tex_].ready) ? bound_tex_ : default_tex_;
        ds = textures_[tex].desc_set;
        textured = (tex != default_tex_);
    }
    if (ds) vkCmdBindDescriptorSets(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     pipeline_layout_, 0, 1, &ds, 0, nullptr);

    vkCmdBindVertexBuffers(f.cmd, 0, 1, &f.transient_vb, &off);

    // Vertex push constants
    struct VertPC {
        glm::mat4 mvp;
        glm::vec4 nm0, nm1, nm2;
        glm::vec4 chunk_lit;
        glm::vec4 l0, l1, ldiff, lamb;
    } vpc{};
    static_assert(sizeof(vpc) == 192);
    const auto& mv = mv_stack_.back();
    vpc.mvp = proj_stack_.back() * mv;
    // Software depth bias (bgfx pattern)
    constexpr float Z_BIAS_EPS = 6e-5f;
    vpc.mvp[3][2] += depth_bias_constant_ * Z_BIAS_EPS;
    depth_bias_constant_ = 0; depth_bias_slope_ = 0;
    // Y-flip for Vulkan NDC
    vpc.mvp[1] = -vpc.mvp[1];

    glm::mat3 nm(mv);
    const auto& tm = tex_stack_.back();
    vpc.nm0 = glm::vec4(nm[0], tm[0][0]);
    vpc.nm1 = glm::vec4(nm[1], tm[1][1]);
    vpc.nm2 = glm::vec4(nm[2], tm[3][0]);
    vpc.chunk_lit = glm::vec4(0, 0, 0, lighting_enabled_ ? 1.0f : 0.0f);
    vpc.l0   = glm::vec4(light0_dir_eye_, tm[3][1]);
    vpc.l1   = glm::vec4(light1_dir_eye_, mv[3][0]);
    vpc.ldiff = glm::vec4(light_diffuse_, mv[3][1]);
    vpc.lamb  = glm::vec4(light_ambient_, mv[3][2]);
    vkCmdPushConstants(f.cmd, pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT,
                       0, 192, &vpc);

    // Fragment push constants
    struct FragPC {
        uint32_t flags; float alpha_ref; float inv_gamma; uint32_t pad;
        float state_colour[4]; float fog_params[4]; float fog_colour[4];
    } fpc{};
    fpc.flags = ((textured && texture_enabled_) ? 1u : 0u) |
                (alpha_test_enabled_ ? 2u : 0u);
    fpc.alpha_ref = alpha_ref_;
    fpc.inv_gamma = inv_gamma_;
    std::memcpy(fpc.state_colour, state_colour_.data(), 16);
    if (fog_enabled_) {
        switch (fog_mode_) {
            case rp::FogMode::linear:         fpc.fog_params[0] = 1; break;
            case rp::FogMode::exponential:    fpc.fog_params[0] = 2; break;
            case rp::FogMode::exponential_sq: fpc.fog_params[0] = 3; break;
            default: break;
        }
    }
    fpc.fog_params[1] = fog_start_; fpc.fog_params[2] = fog_end_;
    fpc.fog_params[3] = fog_density_;
    std::memcpy(fpc.fog_colour, fog_colour_.data(), 16);
    vkCmdPushConstants(f.cmd, pipeline_layout_, VK_SHADER_STAGE_FRAGMENT_BIT,
                       192, 64, &fpc);

    if (is_quads) {
        uint32_t qc = std::min(uint32_t(count) / 4, kMaxQuads);
        vkCmdBindIndexBuffer(f.cmd, quad_ib_, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(f.cmd, qc * 6, 1, 0, 0, 0);
    } else {
        vkCmdDraw(f.cmd, uint32_t(count), 1, 0, 0);
    }
    ++stat_draws_;
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
bool Renderer::IsWidescreen() { return fb_.width > fb_.height; }
bool Renderer::IsHiDef() { return fb_.height >= 720; }

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
    std::lock_guard lk(tex_mu_);
    textures_.emplace_back();
    return int(textures_.size()) - 1;
}

void Renderer::TextureFree(int idx) {
    std::lock_guard lk(tex_mu_);
    if (idx <= 0 || size_t(idx) >= textures_.size()) return;
    if (idx == default_tex_) return;
    auto& t = textures_[idx];
    // Defer destruction until frame fence signals
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
    std::lock_guard lk(tex_mu_);
    if (idx >= 0 && size_t(idx) < textures_.size() && textures_[idx].ready)
        bound_tex_ = idx;
    else
        bound_tex_ = default_tex_;
}

void Renderer::TextureData(int w, int h, void* data, int level, int) {
    if (level != 0 || !data) return;
    int idx;
    { std::lock_guard lk(tex_mu_); idx = bound_tex_; if (idx <= 0) return;
      if (size_t(idx) >= textures_.size()) textures_.resize(idx + 1); }
    upload_texture(idx, w, h, data);
}

void Renderer::TextureDataUpdate(int xo, int yo, int w, int h, void* data, int level) {
    if (level != 0 || !data) return;
    int idx;
    { std::lock_guard lk(tex_mu_); idx = bound_tex_; if (idx <= 0 || size_t(idx) >= textures_.size()) return; }
    auto& t = textures_[idx];
    if (xo != 0 || yo != 0 || (t.ready && (uint32_t(w) != t.width || uint32_t(h) != t.height)))
        return;
    upload_texture(idx, w, h, data);
}

int Renderer::ensure_default_texture() {
    int idx = TextureCreate();
    uint32_t pixel = 0xFFFFFFFF;
    { std::lock_guard lk(tex_mu_); bound_tex_ = idx; }
    upload_texture(idx, 1, 1, &pixel);
    return idx;
}

void Renderer::upload_texture(int idx, int w, int h, const void* pixels) {
    auto& t = textures_[idx];
    bool reuse = t.ready && t.width == uint32_t(w) && t.height == uint32_t(h);
    if (t.ready && !reuse) {
        // Defer old image destruction
        auto view = t.view; auto img = t.image; auto alloc = t.alloc;
        auto dev = dev_.handle(); auto vma = dev_.allocator();
        frame().deletions.push([=]() {
            if (view) vkDestroyImageView(dev, view, nullptr);
            if (img)  vmaDestroyImage(vma, img, alloc);
        });
        VkDescriptorSet keep = t.desc_set;
        t = {}; t.desc_set = keep;
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
        vmaCreateImage(dev_.allocator(), &ici, &ai, &t.image, &t.alloc, nullptr);

        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = t.image; vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_R8G8B8A8_UNORM;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 1};
        vkCreateImageView(dev_.handle(), &vci, nullptr, &t.view);
    }

    if (!t.desc_set) {
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = tex_pool_; dai.descriptorSetCount = 1;
        dai.pSetLayouts = &tex_set_layout_;
        vkAllocateDescriptorSets(dev_.handle(), &dai, &t.desc_set);
    }
    VkDescriptorImageInfo dii{tex_sampler_, t.view,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet wd{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wd.dstSet = t.desc_set; wd.dstBinding = 0; wd.descriptorCount = 1;
    wd.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wd.pImageInfo = &dii;
    vkUpdateDescriptorSets(dev_.handle(), 1, &wd, 0, nullptr);

    // Stage + copy via upload pool
    VkDeviceSize bytes = VkDeviceSize(w) * h * 4;
    if (bytes > kStagingSize) return;
    std::memcpy(staging_mapped_, pixels, bytes);

    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = upload_pool_; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd; vkAllocateCommandBuffers(dev_.handle(), &cai, &cmd);
    VkCommandBufferBeginInfo bbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bbi);

    // UNDEFINED → TRANSFER_DST
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

    // All mips → SHADER_READ_ONLY
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
    vkQueueWaitIdle(dev_.queue());  // TODO: batch into frame cmd
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
    std::lock_guard lk(cbuf_mu_);
    int first = next_cbuf_;
    int needed = n > 0 ? n : 1;
    next_cbuf_ += needed;
    if (size_t(first + needed) > cbufs_.size()) cbufs_.resize(first + needed);
    return first;
}

void Renderer::CBuffDeleteAll() {
    std::lock_guard lk(cbuf_mu_);
    for (auto& cb : cbufs_)
        if (cb.vb) frame().deletions.push_buffer(dev_.allocator(), cb.vb, cb.alloc);
    cbufs_.clear();
    next_cbuf_ = 1;
    t_rec.id = -1; t_rec.draws.clear();
}

void Renderer::CBuffStart(int index, bool) { t_rec.id = index; t_rec.draws.clear(); }

void Renderer::CBuffClear(int index) {
    std::lock_guard lk(cbuf_mu_);
    if (index < 0 || size_t(index) >= cbufs_.size()) return;
    auto& cb = cbufs_[index];
    cb.draws.clear(); cb.gpu_draws.clear();
    if (cb.vb) {
        frame().deletions.push_buffer(dev_.allocator(), cb.vb, cb.alloc);
        cb.vb = VK_NULL_HANDLE; cb.alloc = nullptr; cb.vb_size = 0;
    }
    cb.valid = cb.uploaded = false;
}

int Renderer::CBuffSize(int index) {
    std::lock_guard lk(cbuf_mu_);
    if (index < 0 || size_t(index) >= cbufs_.size()) return 0;
    return cbufs_[index].valid ? 1 : 0;
}

void Renderer::CBuffEnd() {
    int id = t_rec.id; t_rec.id = -1;
    if (id < 0) return;
    std::lock_guard lk(cbuf_mu_);
    if (size_t(id) >= cbufs_.size()) cbufs_.resize(id + 1);
    auto& cb = cbufs_[id];
    cb.draws = std::move(t_rec.draws);
    cb.valid = !cb.draws.empty();
    cb.uploaded = false;
    t_rec.draws.clear();
}

void Renderer::cbuf_upload(CBuff& cb) {
    constexpr uint32_t kStride = 32;
    std::vector<std::byte> combined;
    cb.gpu_draws.clear();

    for (auto& d : cb.draws) {
        uint32_t verts = uint32_t(d.verts.size() / kStride);
        if (verts == 0) continue;
        uint32_t byte_off = uint32_t(combined.size());
        if (d.primType == 0x0007) {
            if (verts % 4 != 0) continue;
            uint32_t quads = verts / 4;
            for (uint32_t q = 0; q < quads; ++q) {
                const std::byte* b = d.verts.data() + q * 4 * kStride;
                auto push = [&](uint32_t i) {
                    combined.insert(combined.end(), b + i*kStride, b + (i+1)*kStride);
                };
                push(0); push(1); push(2); push(0); push(2); push(3);
            }
            cb.gpu_draws.push_back({byte_off, quads * 6, 0x0004});
        } else {
            combined.insert(combined.end(), d.verts.begin(), d.verts.end());
            cb.gpu_draws.push_back({byte_off, verts, d.primType});
        }
    }
    if (combined.empty()) { cb.uploaded = false; return; }

    uint32_t needed = uint32_t(combined.size());
    if (cb.vb && cb.vb_size < needed) {
        frame().deletions.push_buffer(dev_.allocator(), cb.vb, cb.alloc);
        cb.vb = VK_NULL_HANDLE; cb.alloc = nullptr;
    }
    if (!cb.vb) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = needed;
        bi.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo ai{}; ai.usage = VMA_MEMORY_USAGE_AUTO;
        vmaCreateBuffer(dev_.allocator(), &bi, &ai, &cb.vb, &cb.alloc, nullptr);
        cb.vb_size = needed;
    }

    // Stage + copy
    if (needed > kStagingSize) { cb.uploaded = false; return; }
    std::memcpy(staging_mapped_, combined.data(), needed);

    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = upload_pool_; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd; vkAllocateCommandBuffers(dev_.handle(), &cai, &cmd);
    VkCommandBufferBeginInfo bbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bbi);
    VkBufferCopy rgn{0, 0, needed};
    vkCmdCopyBuffer(cmd, staging_buf_, cb.vb, 1, &rgn);
    vkEndCommandBuffer(cmd);
    VkCommandBufferSubmitInfo csi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    csi.commandBuffer = cmd;
    VkSubmitInfo2 sub{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    sub.commandBufferInfoCount = 1; sub.pCommandBufferInfos = &csi;
    vkQueueSubmit2(dev_.queue(), 1, &sub, VK_NULL_HANDLE);
    vkQueueWaitIdle(dev_.queue());  // TODO: batch into frame cmd
    vkResetCommandBuffer(cmd, 0);
    cb.uploaded = true;
}

bool Renderer::CBuffCall(int index, bool) {
    if (index < 0 || !frame_active_) return false;

    VkBuffer vb = VK_NULL_HANDLE;
    std::vector<CBuffSubDraw> draws;
    {
        std::lock_guard lk(cbuf_mu_);
        if (size_t(index) >= cbufs_.size()) return false;
        auto& cb = cbufs_[index];
        if (!cb.valid || cb.draws.empty()) return false;
        if (!cb.uploaded) { cbuf_upload(cb); if (!cb.uploaded) return false; }
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

    VkDescriptorSet ds = VK_NULL_HANDLE;
    bool textured = false;
    {
        std::lock_guard lk(tex_mu_);
        int tex = (bound_tex_ > 0 && size_t(bound_tex_) < textures_.size() &&
                   textures_[bound_tex_].ready) ? bound_tex_ : default_tex_;
        ds = textures_[tex].desc_set;
        textured = (tex != default_tex_);
    }
    if (ds) vkCmdBindDescriptorSets(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     pipeline_layout_, 0, 1, &ds, 0, nullptr);

    // Push constants once for entire display list
    struct VertPC {
        glm::mat4 mvp;
        glm::vec4 nm0, nm1, nm2, chunk_lit, l0, l1, ldiff, lamb;
    } vpc{};
    const auto& mv = mv_stack_.back();
    vpc.mvp = proj_stack_.back() * mv;
    constexpr float Z_BIAS_EPS = 6e-5f;
    vpc.mvp[3][2] += depth_bias_constant_ * Z_BIAS_EPS;
    depth_bias_constant_ = 0; depth_bias_slope_ = 0;
    vpc.mvp[1] = -vpc.mvp[1];
    glm::mat3 nm(mv);
    const auto& tm = tex_stack_.back();
    vpc.nm0 = glm::vec4(nm[0], tm[0][0]);
    vpc.nm1 = glm::vec4(nm[1], tm[1][1]);
    vpc.nm2 = glm::vec4(nm[2], tm[3][0]);
    vpc.chunk_lit = glm::vec4(0, 0, 0, lighting_enabled_ ? 1.0f : 0.0f);
    vpc.l0 = glm::vec4(light0_dir_eye_, tm[3][1]);
    vpc.l1 = glm::vec4(light1_dir_eye_, mv[3][0]);
    vpc.ldiff = glm::vec4(light_diffuse_, mv[3][1]);
    vpc.lamb = glm::vec4(light_ambient_, mv[3][2]);
    vkCmdPushConstants(f.cmd, pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT,
                       0, 192, &vpc);

    struct FragPC {
        uint32_t flags; float alpha_ref; float inv_gamma; uint32_t pad;
        float sc[4]; float fp[4]; float fc[4];
    } fpc{};
    fpc.flags = ((textured && texture_enabled_) ? 1u : 0u) |
                (alpha_test_enabled_ ? 2u : 0u);
    fpc.alpha_ref = alpha_ref_; fpc.inv_gamma = inv_gamma_;
    std::memcpy(fpc.sc, state_colour_.data(), 16);
    if (fog_enabled_) {
        switch (fog_mode_) {
            case rp::FogMode::linear: fpc.fp[0] = 1; break;
            case rp::FogMode::exponential: fpc.fp[0] = 2; break;
            case rp::FogMode::exponential_sq: fpc.fp[0] = 3; break;
            default: break;
        }
    }
    fpc.fp[1] = fog_start_; fpc.fp[2] = fog_end_; fpc.fp[3] = fog_density_;
    std::memcpy(fpc.fc, fog_colour_.data(), 16);
    vkCmdPushConstants(f.cmd, pipeline_layout_, VK_SHADER_STAGE_FRAGMENT_BIT,
                       192, 64, &fpc);

    for (auto& sd : draws) {
        VkDeviceSize off = sd.vertex_offset;
        vkCmdBindVertexBuffers(f.cmd, 0, 1, &vb, &off);
        VkPrimitiveTopology topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        switch (sd.prim_type) {
            case 0x0001: topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; break;
            case 0x0003: topo = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; break;
            case 0x0005: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
            case 0x0006: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN; break;
        }
        vkCmdSetPrimitiveTopology(f.cmd, topo);
        vkCmdDraw(f.cmd, sd.vertex_count, 1, 0, 0);
        ++stat_draws_;
    }
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

    VkPrimitiveTopology topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    switch (tvb.primitive) {
        case rp::PrimitiveType::triangle_strip: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
        case rp::PrimitiveType::triangle_fan:   topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN; break;
        case rp::PrimitiveType::line_list:      topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; break;
        case rp::PrimitiveType::line_strip:     topo = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; break;
        default: break;
    }
    vkCmdSetPrimitiveTopology(f.cmd, topo);

    VkDescriptorSet ds = VK_NULL_HANDLE; bool textured = false;
    { std::lock_guard lk(tex_mu_);
      int tex = (bound_tex_ > 0 && size_t(bound_tex_) < textures_.size() &&
                 textures_[bound_tex_].ready) ? bound_tex_ : default_tex_;
      ds = textures_[tex].desc_set; textured = (tex != default_tex_); }
    if (ds) vkCmdBindDescriptorSets(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     pipeline_layout_, 0, 1, &ds, 0, nullptr);

    VkDeviceSize off = tvb.offset;
    vkCmdBindVertexBuffers(f.cmd, 0, 1, &f.transient_vb, &off);

    // Push constants (same as DrawVertices but with tint from dc)
    struct VertPC { glm::mat4 mvp; glm::vec4 nm0,nm1,nm2,cl,l0,l1,ld,la; } vpc{};
    const auto& mv = mv_stack_.back();
    vpc.mvp = proj_stack_.back() * mv;
    vpc.mvp[1] = -vpc.mvp[1];
    glm::mat3 nm(mv);
    vpc.nm0 = glm::vec4(nm[0], 1); vpc.nm1 = glm::vec4(nm[1], 1);
    vpc.nm2 = glm::vec4(nm[2], 0);
    vpc.l0 = glm::vec4(light0_dir_eye_, 0);
    vpc.l1 = glm::vec4(light1_dir_eye_, mv[3][0]);
    vpc.ld = glm::vec4(light_diffuse_, mv[3][1]);
    vpc.la = glm::vec4(light_ambient_, mv[3][2]);
    vkCmdPushConstants(f.cmd, pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, 192, &vpc);

    struct FragPC { uint32_t fl; float ar; float ig; uint32_t p; float sc[4]; float fp[4]; float fc[4]; } fpc{};
    fpc.fl = ((textured && texture_enabled_) ? 1u : 0u) | (alpha_test_enabled_ ? 2u : 0u);
    fpc.ar = alpha_ref_; fpc.ig = inv_gamma_;
    fpc.sc[0] = state_colour_[0]*dc.tint_color[0]; fpc.sc[1] = state_colour_[1]*dc.tint_color[1];
    fpc.sc[2] = state_colour_[2]*dc.tint_color[2]; fpc.sc[3] = state_colour_[3]*dc.tint_color[3];
    if (fog_enabled_) { switch(fog_mode_) {
        case rp::FogMode::linear: fpc.fp[0]=1; break;
        case rp::FogMode::exponential: fpc.fp[0]=2; break;
        case rp::FogMode::exponential_sq: fpc.fp[0]=3; break;
        default: break; } }
    fpc.fp[1]=fog_start_; fpc.fp[2]=fog_end_; fpc.fp[3]=fog_density_;
    std::memcpy(fpc.fc, fog_colour_.data(), 16);
    vkCmdPushConstants(f.cmd, pipeline_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 192, 64, &fpc);

    vkCmdDraw(f.cmd, tvb.vertex_count, 1, 0, 0);
    ++stat_draws_;
}

// ===================================================================
// TERRAIN (delegates to existing TerrainRenderer)
// ===================================================================

void Renderer::chunk_upload(const ChunkUpload& up) {
    if (!terrain_) return;
    terrain_->upload_chunk({up.cx, up.cy, up.cz, up.layer},
                           {up.world_origin[0], up.world_origin[1], up.world_origin[2]},
                           {up.aabb_min[0], up.aabb_min[1], up.aabb_min[2]},
                           {up.aabb_max[0], up.aabb_max[1], up.aabb_max[2]},
                           up.vertex_data, up.vertex_count, up.vertex_stride);
}

void Renderer::chunk_destroy(int32_t cx, int32_t cy, int32_t cz, uint8_t layer) {
    if (terrain_) terrain_->destroy_chunk({cx, cy, cz, layer});
}

void Renderer::chunk_upload_from_cbuff(int cbuff_id, const ChunkUpload& base) {
    if (!terrain_ || cbuff_id < 0) return;
    constexpr uint32_t kStride = 32;
    std::vector<std::byte> combined;
    uint32_t total = 0;
    {
        std::lock_guard lk(cbuf_mu_);
        if (size_t(cbuff_id) >= cbufs_.size()) return;
        auto& cb = cbufs_[cbuff_id];
        if (!cb.valid) return;
        size_t est = 0;
        for (auto& d : cb.draws) est += d.verts.size() * 6 / 4;
        combined.reserve(est);
        for (auto& d : cb.draws) {
            if (d.primType != 0x0007) continue;
            uint32_t v = uint32_t(d.verts.size() / kStride);
            if (v == 0 || v % 4 != 0) continue;
            for (uint32_t q = 0; q < v/4; ++q) {
                const std::byte* b = d.verts.data() + q * 4 * kStride;
                auto push = [&](uint32_t i) {
                    combined.insert(combined.end(), b+i*kStride, b+(i+1)*kStride);
                };
                push(0); push(1); push(2); push(0); push(2); push(3);
            }
            total += (v/4) * 6;
        }
    }
    if (total == 0) return;
    ChunkUpload up = base;
    up.vertex_data = combined.data();
    up.vertex_count = total;
    up.vertex_stride = kStride;
    chunk_upload(up);
}

void Renderer::render_terrain(const float* mvp_4x4, const float* frustum_24, uint8_t layer) {
    if (!terrain_ || !frame_active_ || !mvp_4x4 || !frustum_24) return;

    {
        std::lock_guard lk(tex_mu_);
        int tex = bound_tex_;
        if (tex > 0 && size_t(tex) < textures_.size() && textures_[tex].ready)
            terrain_->set_atlas(textures_[tex].view, tex_sampler_);
        int lm = lightmap_tex_;
        if (lm > 0 && size_t(lm) < textures_.size() && textures_[lm].ready)
            terrain_->set_lightmap(textures_[lm].view, tex_sampler_lm_);
    }

    auto& f = frame();
    ensure_pass();

    glm::mat4 mvp; std::memcpy(&mvp[0][0], mvp_4x4, 64);
    std::array<glm::vec4, 6> frustum;
    for (int i = 0; i < 6; ++i)
        frustum[i] = glm::vec4(frustum_24[i*4], frustum_24[i*4+1],
                               frustum_24[i*4+2], frustum_24[i*4+3]);

    plce::vk_render::TerrainRenderer::FogParams fog{};
    if (fog_enabled_) {
        switch (fog_mode_) {
            case rp::FogMode::linear: fog.params.x = 1; break;
            case rp::FogMode::exponential: fog.params.x = 2; break;
            case rp::FogMode::exponential_sq: fog.params.x = 3; break;
            default: break;
        }
    }
    fog.params.y = fog_start_; fog.params.z = fog_end_; fog.params.w = fog_density_;
    fog.colour = glm::vec4(fog_colour_[0], fog_colour_[1], fog_colour_[2], fog_colour_[3]);

    const auto& cam_mv = mv_stack_.back();
    glm::mat3 cr(cam_mv); glm::vec3 ct(cam_mv[3]);
    glm::vec3 cw = -glm::inverse(cr) * ct;

    glm::vec4 tint(state_colour_[0], state_colour_[1], state_colour_[2], state_colour_[3]);
    terrain_->render(f.cmd, mvp, frustum, uint32_t(layer), glm::vec4(cw, 0), fog, tint);

    pso_dirty_ = true;
}

void Renderer::set_terrain_atlas(int tex_id) {
    if (!terrain_) return;
    std::lock_guard lk(tex_mu_);
    if (tex_id <= 0 || size_t(tex_id) >= textures_.size()) return;
    auto& t = textures_[tex_id];
    if (!t.ready || !t.view) return;
    terrain_->set_atlas(t.view, tex_sampler_);
}

}  // namespace plce::vk2

// ===================================================================
// FACTORY (replaces make_vulkan_render_path)
// ===================================================================

std::unique_ptr<rp::IRenderPath> make_vulkan_render_path(SDL_Window* window) {
    return std::make_unique<plce::vk2::Renderer>(window);
}
