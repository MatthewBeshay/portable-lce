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

#include "VkCheck.h"
#include "VertexFormats.h"

#include "vk/shaders/basic.vert.spv.h"  // kBasicVertSpv
#include "vk/shaders/basic.frag.spv.h"  // kBasicFragSpv

namespace plce::vk {

namespace {
constexpr const char* kPipelineCachePath = "pipeline_cache.bin";
}

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
#ifdef ENABLE_VSYNC
    swap_.create(dev_, 0, 0, VK_PRESENT_MODE_FIFO_KHR);
#else
    swap_.create(dev_, 0, 0, VK_PRESENT_MODE_MAILBOX_KHR);
#endif

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
        pipelines_.load_cache(kPipelineCachePath);
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

    tex_mgr_.init(dev_.handle(), dev_.allocator(), dev_.queue(), dev_.queue_family(),
                  tex_set_layout_, tex_pool_);
    dl_mgr_.init();

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
    pipelines_.save_cache(kPipelineCachePath);
    for (auto& pd : pending_destroys_)
        vmaDestroyBuffer(dev_.allocator(), pd.buf, pd.alloc);
    pending_destroys_.clear();
    tex_mgr_.destroy(dev_.handle(), dev_.allocator());
    if (quad_ib_) vmaDestroyBuffer(dev_.allocator(), quad_ib_, quad_ib_alloc_);
    pipelines_.destroy();
    if (pipeline_layout_) vkDestroyPipelineLayout(dev_.handle(), pipeline_layout_, nullptr);
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

    // Poll async texture uploads and mark completed ones ready.
    tex_mgr_.poll_uploads();

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

    f.begin(dev_.handle(), dev_.allocator());

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

    // Y-flip via negative viewport height (VK_KHR_maintenance1, core in 1.1).
    // Geometry in game space (Y-up) renders with CCW front faces and back-face
    // culling — the canonical Vulkan configuration. No MVP row negation needed.
    const float w = float(swap_.extent().width);
    const float h = float(swap_.extent().height);
    VkViewport vp{0.0f, h, w, -h, 0.0f, 1.0f};
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
        std::fprintf(stderr, "[vk] matrix stack overflow at depth %u\n", s.depth);
        return;
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
void Renderer::StateSetWriteEnable(bool r, bool g, bool b, bool a) {
    uint8_t m = (r?1:0)|(g?2:0)|(b?4:0)|(a?8:0);
    if (pso_key_.color_mask() != m) { pso_key_.set_color_mask(m); pso_dirty_ = true; }
}
void Renderer::StateSetDepthTestEnable(bool e) { if (pso_key_.depth_test() != e) { pso_key_.set_depth_test(e); pso_dirty_ = true; } }
void Renderer::StateSetAlphaTestEnable(bool e) { alpha_test_enabled_ = e; }
void Renderer::StateSetDepthSlopeAndBias(float slope, float bias) {
    depth_bias_slope_ = slope; depth_bias_constant_ = bias;
}
void Renderer::StateSetLightDirection(int idx, float x, float y, float z) {
    glm::vec3 d = glm::normalize(glm::mat3(mv_stack_.top()) * glm::vec3(x,y,z));
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
// DRAW
// ===================================================================

void Renderer::DrawVertices(int primType, int count, void* data,
                            int vType, int /*sType*/) {
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
    if (pso_key_.lines() != is_lines) { pso_key_.set_lines(is_lines); pso_dirty_ = true; }

    auto& f = frame();
    constexpr VkDeviceSize stride = 32;
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
    vkCmdSetBlendConstants(f.cmd, blend_constants_.data());

    // Dynamic depth bias
    vkCmdSetDepthBias(f.cmd, depth_bias_constant_, 0.0f, depth_bias_slope_);

    auto [ds, textured, lm_active] = tex_mgr_.bind_textures(f.cmd, pipeline_layout_);

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
// TEXTURES (delegated to TextureManager)
// ===================================================================

int Renderer::TextureCreate() { return tex_mgr_.create(); }
void Renderer::TextureFree(int idx) { tex_mgr_.free(idx, frame().deletions); }
void Renderer::TextureBind(int idx) { tex_mgr_.bind(idx); }
void Renderer::TextureData(int w, int h, void* data, int level, int) { tex_mgr_.data(w, h, data, level); }
void Renderer::TextureDataUpdate(int xo, int yo, int w, int h, void* data, int lvl) { tex_mgr_.data_update(xo, yo, w, h, data, lvl); }
void Renderer::TextureSetParam(int param, int value) { tex_mgr_.set_param(param, value); }
int Renderer::LoadTextureData(const char* fn, void* info, int** out) { return tex_mgr_.load_texture_data(fn, info, out); }
int Renderer::LoadTextureData(uint8_t* data, uint32_t bytes, void* info, int** out) { return tex_mgr_.load_texture_data(data, bytes, info, out); }

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
    if (snap.vb == VK_NULL_HANDLE) return false;

    VkBuffer vb = snap.vb;
    const auto& draws = snap.draws;

    auto& f = frame();
    ensure_pass();

    if (pso_dirty_ || !(pso_key_ == last_bound_pso_)) {
        vkCmdBindPipeline(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipelines_.get(pso_key_));
        last_bound_pso_ = pso_key_; pso_dirty_ = false;
    }
    vkCmdSetBlendConstants(f.cmd, blend_constants_.data());
    vkCmdSetDepthBias(f.cmd, depth_bias_constant_, 0.0f, depth_bias_slope_);

    auto [ds, textured, lm_active] = tex_mgr_.bind_textures(f.cmd, pipeline_layout_);

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

    auto [ds, textured, lm_active] = tex_mgr_.bind_textures(f.cmd, pipeline_layout_);

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

}  // namespace plce::vk

// ===================================================================
// FACTORY
// ===================================================================

std::unique_ptr<rp::IRenderPath> make_vulkan_render_path(SDL_Window* window) {
    return std::make_unique<plce::vk::Renderer>(window);
}
