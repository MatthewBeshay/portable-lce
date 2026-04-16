#include "TerrainRenderer.h"

#include <array>
#include <cstring>
#include <stdexcept>

#include "vk/shaders/terrain_cull.comp.spv.h"
#include "vk/shaders/terrain.vert.spv.h"
#include "vk/shaders/terrain.frag.spv.h"

namespace plce::vk_render {

namespace {

[[noreturn]] void vk_throw(const char* what, VkResult r) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "TerrainRenderer: %s failed: VkResult=%d",
                  what, int(r));
    throw std::runtime_error(buf);
}

void vk_check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) vk_throw(what, r);
}

VkShaderModule make_module(VkDevice device, const uint32_t* code,
                           size_t bytes) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = bytes;
    ci.pCode = code;
    VkShaderModule m = VK_NULL_HANDLE;
    vk_check(vkCreateShaderModule(device, &ci, nullptr, &m),
             "vkCreateShaderModule");
    return m;
}

}  // namespace

TerrainRenderer::TerrainRenderer(VkDevice device, VmaAllocator allocator,
                                 uint32_t /*graphics_queue_family*/,
                                 VkQueue /*transfer_queue*/,
                                 const Config& cfg)
    : device_(device), allocator_(allocator), cfg_(cfg) {
    arena_    = std::make_unique<ChunkArena>(device_, allocator_,
                                             cfg_.slot_bytes,
                                             cfg_.arena_bytes);
    metadata_ = std::make_unique<ChunkMetadata>(device_, allocator_,
                                                arena_->slot_count());
    staging_  = std::make_unique<StagingRing>(allocator_, cfg_.staging_bytes);
    indirect_ = std::make_unique<IndirectDrawBuffer>(allocator_,
                                                     cfg_.max_visible);

    create_descriptor_layouts();
    create_pipelines();
    create_descriptor_pool_and_sets();
}

TerrainRenderer::~TerrainRenderer() {
    destroy_descriptor_pool();
    destroy_pipelines();
    destroy_descriptor_layouts();
    // Sub-objects clean themselves up via unique_ptr destructors.
}

void TerrainRenderer::create_descriptor_layouts() {
    // Cull set: metadata SSBO (read), draws SSBO (write), count SSBO (RW).
    {
        VkDescriptorSetLayoutBinding b[3]{};
        for (int i = 0; i < 3; ++i) {
            b[i].binding = uint32_t(i);
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo ci{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = 3;
        ci.pBindings = b;
        vk_check(vkCreateDescriptorSetLayout(device_, &ci, nullptr,
                                             &cull_set_layout_),
                 "vkCreateDescriptorSetLayout(cull)");
    }
    // Terrain set: metadata SSBO read in vertex stage (chunk world pos),
    // sampler in fragment.
    {
        VkDescriptorSetLayoutBinding b[2]{};
        b[0].binding = 0;
        b[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[0].descriptorCount = 1;
        b[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        b[1].binding = 1;
        b[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[1].descriptorCount = 1;
        b[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo ci{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = 2;
        ci.pBindings = b;
        vk_check(vkCreateDescriptorSetLayout(device_, &ci, nullptr,
                                             &terrain_set_layout_),
                 "vkCreateDescriptorSetLayout(terrain)");
    }
}

void TerrainRenderer::destroy_descriptor_layouts() {
    if (terrain_set_layout_) {
        vkDestroyDescriptorSetLayout(device_, terrain_set_layout_, nullptr);
        terrain_set_layout_ = VK_NULL_HANDLE;
    }
    if (cull_set_layout_) {
        vkDestroyDescriptorSetLayout(device_, cull_set_layout_, nullptr);
        cull_set_layout_ = VK_NULL_HANDLE;
    }
}

void TerrainRenderer::create_pipelines() {
    // Pipeline objects are created lazily in render() once we have the
    // shaders and the surrounding render-pass formats. Layouts can be
    // built upfront because they only depend on the descriptor layouts
    // and push-constant ranges.
    {
        // Cull pipeline layout: 1 set + push constant
        // (6 frustum vec4 + slot_count u32 + max_draws u32 = 6*16 + 8 = 104B)
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.offset = 0;
        pc.size = 6 * 16 + 8;
        VkPipelineLayoutCreateInfo lci{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        lci.setLayoutCount = 1;
        lci.pSetLayouts = &cull_set_layout_;
        lci.pushConstantRangeCount = 1;
        lci.pPushConstantRanges = &pc;
        vk_check(vkCreatePipelineLayout(device_, &lci, nullptr,
                                        &cull_pipeline_layout_),
                 "vkCreatePipelineLayout(cull)");
    }
    {
        // Terrain pipeline layout: 1 set + push constant (mat4 mvp = 64B)
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pc.offset = 0;
        pc.size = 64;
        VkPipelineLayoutCreateInfo lci{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        lci.setLayoutCount = 1;
        lci.pSetLayouts = &terrain_set_layout_;
        lci.pushConstantRangeCount = 1;
        lci.pPushConstantRanges = &pc;
        vk_check(vkCreatePipelineLayout(device_, &lci, nullptr,
                                        &terrain_pipeline_layout_),
                 "vkCreatePipelineLayout(terrain)");
    }
    // Compute pipeline: terrain_cull.comp
    {
        VkShaderModule mod = make_module(device_, kTerrainCullSpv,
                                         sizeof(kTerrainCullSpv));
        VkPipelineShaderStageCreateInfo stage{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName = "main";

        VkComputePipelineCreateInfo ci{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = stage;
        ci.layout = cull_pipeline_layout_;
        vk_check(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &ci,
                                          nullptr, &cull_pipeline_),
                 "vkCreateComputePipelines(cull)");
        vkDestroyShaderModule(device_, mod, nullptr);
    }

    // Graphics pipeline: terrain.vert + terrain.frag, standard 32-byte
    // vertex layout. Depth tested + written, alpha cutout via discard so
    // blend stays disabled.
    {
        VkShaderModule vs = make_module(device_, kTerrainVertSpv,
                                        sizeof(kTerrainVertSpv));
        VkShaderModule fs = make_module(device_, kTerrainFragSpv,
                                        sizeof(kTerrainFragSpv));

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vs;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fs;
        stages[1].pName = "main";

        VkVertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = 32;
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        std::array<VkVertexInputAttributeDescription, 3> attrs{};
        attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0};
        attrs[1] = {1, 0, VK_FORMAT_R32G32_SFLOAT,    12};
        attrs[2] = {2, 0, VK_FORMAT_R8G8B8A8_UNORM,   20};

        VkPipelineVertexInputStateCreateInfo vi{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &binding;
        vi.vertexAttributeDescriptionCount = uint32_t(attrs.size());
        vi.pVertexAttributeDescriptions = attrs.data();

        VkPipelineInputAssemblyStateCreateInfo ia{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1;
        vp.scissorCount  = 1;

        VkPipelineRasterizationStateCreateInfo rs{
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_BACK_BIT;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable  = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendAttachmentState att{};
        att.blendEnable = VK_FALSE;
        att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                             VK_COLOR_COMPONENT_G_BIT |
                             VK_COLOR_COMPONENT_B_BIT |
                             VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo cb{
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1;
        cb.pAttachments = &att;

        VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                       VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{
            VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dyn.dynamicStateCount = uint32_t(std::size(dyn_states));
        dyn.pDynamicStates = dyn_states;

        VkPipelineRenderingCreateInfo prci{
            VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        prci.colorAttachmentCount    = 1;
        prci.pColorAttachmentFormats = &cfg_.color_format;
        prci.depthAttachmentFormat   = cfg_.depth_format;

        VkGraphicsPipelineCreateInfo gci{
            VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gci.pNext = &prci;
        gci.stageCount = 2;
        gci.pStages = stages;
        gci.pVertexInputState   = &vi;
        gci.pInputAssemblyState = &ia;
        gci.pViewportState      = &vp;
        gci.pRasterizationState = &rs;
        gci.pMultisampleState   = &ms;
        gci.pDepthStencilState  = &ds;
        gci.pColorBlendState    = &cb;
        gci.pDynamicState       = &dyn;
        gci.layout              = terrain_pipeline_layout_;
        vk_check(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gci,
                                           nullptr, &terrain_pipeline_),
                 "vkCreateGraphicsPipelines(terrain)");

        vkDestroyShaderModule(device_, vs, nullptr);
        vkDestroyShaderModule(device_, fs, nullptr);
    }
}

void TerrainRenderer::destroy_pipelines() {
    if (cull_pipeline_)    vkDestroyPipeline(device_, cull_pipeline_, nullptr);
    if (terrain_pipeline_) vkDestroyPipeline(device_, terrain_pipeline_,
                                             nullptr);
    if (cull_pipeline_layout_)
        vkDestroyPipelineLayout(device_, cull_pipeline_layout_, nullptr);
    if (terrain_pipeline_layout_)
        vkDestroyPipelineLayout(device_, terrain_pipeline_layout_, nullptr);
    cull_pipeline_           = VK_NULL_HANDLE;
    terrain_pipeline_        = VK_NULL_HANDLE;
    cull_pipeline_layout_    = VK_NULL_HANDLE;
    terrain_pipeline_layout_ = VK_NULL_HANDLE;
}

void TerrainRenderer::create_descriptor_pool_and_sets() {
    VkDescriptorPoolSize sizes[2]{};
    sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[0].descriptorCount = 4;  // 3 in cull set + 1 in terrain set
    sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[1].descriptorCount = 1;  // atlas in terrain set

    VkDescriptorPoolCreateInfo pci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 2;
    pci.poolSizeCount = 2;
    pci.pPoolSizes = sizes;
    vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &desc_pool_),
             "vkCreateDescriptorPool");

    {
        VkDescriptorSetAllocateInfo ai{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = desc_pool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &cull_set_layout_;
        vk_check(vkAllocateDescriptorSets(device_, &ai, &cull_set_),
                 "vkAllocateDescriptorSets(cull)");
    }
    {
        VkDescriptorSetAllocateInfo ai{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = desc_pool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &terrain_set_layout_;
        vk_check(vkAllocateDescriptorSets(device_, &ai, &terrain_set_),
                 "vkAllocateDescriptorSets(terrain)");
    }

    // Wire the SSBOs into both sets. The atlas image binding is filled in
    // later via set_atlas().
    VkDescriptorBufferInfo meta_buf{metadata_->buffer(),  0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo draws_buf{indirect_->draws_buffer(), 0,
                                     VK_WHOLE_SIZE};
    VkDescriptorBufferInfo count_buf{indirect_->count_buffer(), 0,
                                     VK_WHOLE_SIZE};

    std::array<VkWriteDescriptorSet, 4> w{};
    for (auto& it : w) it.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;

    w[0].dstSet = cull_set_;
    w[0].dstBinding = 0;
    w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[0].pBufferInfo = &meta_buf;

    w[1].dstSet = cull_set_;
    w[1].dstBinding = 1;
    w[1].descriptorCount = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[1].pBufferInfo = &draws_buf;

    w[2].dstSet = cull_set_;
    w[2].dstBinding = 2;
    w[2].descriptorCount = 1;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[2].pBufferInfo = &count_buf;

    w[3].dstSet = terrain_set_;
    w[3].dstBinding = 0;
    w[3].descriptorCount = 1;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[3].pBufferInfo = &meta_buf;

    vkUpdateDescriptorSets(device_, uint32_t(w.size()), w.data(), 0, nullptr);
}

void TerrainRenderer::destroy_descriptor_pool() {
    if (desc_pool_) {
        vkDestroyDescriptorPool(device_, desc_pool_, nullptr);
        desc_pool_ = VK_NULL_HANDLE;
    }
    cull_set_ = VK_NULL_HANDLE;
    terrain_set_ = VK_NULL_HANDLE;
}

void TerrainRenderer::set_atlas(VkImageView view, VkSampler sampler) {
    if (!view || !sampler) return;
    VkDescriptorImageInfo dii{sampler, view,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = terrain_set_;
    w.dstBinding = 1;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &dii;
    vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    atlas_set_ = true;
}

void TerrainRenderer::upload_chunk(const ChunkKey& key,
                                   const glm::vec3& world_origin,
                                   const glm::vec3& aabb_min,
                                   const glm::vec3& aabb_max,
                                   const void* vertex_data,
                                   uint32_t vertex_count,
                                   uint32_t vertex_stride) {
    if (vertex_count == 0 || !vertex_data) return;

    VkDeviceSize bytes = VkDeviceSize(vertex_count) * vertex_stride;
    if (bytes > arena_->slot_size()) return;  // chunk too big for a slot

    // Get-or-allocate the slot for this key.
    ChunkArena::Slot slot;
    {
        std::lock_guard lk(live_slots_mutex_);
        auto it = live_slots_.find(key);
        if (it != live_slots_.end()) {
            slot = it->second;
        } else {
            slot = arena_->allocate();
            if (!slot.valid()) return;  // arena full
            live_slots_.emplace(key, slot);
        }
    }

    // Stage the bytes; the next render() will record the copy.
    auto a = staging_->alloc(bytes, 16);
    if (!a.valid()) return;  // staging ring full this frame
    std::memcpy(a.ptr, vertex_data, bytes);

    // Compute base_vertex from arena byte offset / stride.
    uint32_t base_vertex = uint32_t(arena_->offset_of(slot) / vertex_stride);

    ChunkSlotData md{};
    md.world_pos[0] = world_origin.x;
    md.world_pos[1] = world_origin.y;
    md.world_pos[2] = world_origin.z;
    md.aabb_min[0]  = aabb_min.x;
    md.aabb_min[1]  = aabb_min.y;
    md.aabb_min[2]  = aabb_min.z;
    md.aabb_max[0]  = aabb_max.x;
    md.aabb_max[1]  = aabb_max.y;
    md.aabb_max[2]  = aabb_max.z;
    md.vert_count   = vertex_count;
    md.base_vertex  = base_vertex;
    md.face_mask    = 0x3F;  // all six faces visible (no face cull yet)
    metadata_->update(slot, md);

    PendingCopy pc{};
    pc.staging_offset      = a.buffer_offset;
    pc.arena_offset        = arena_->offset_of(slot);
    pc.bytes               = bytes;
    pc.staging_checkpoint  = a.checkpoint;
    {
        std::lock_guard lk(pending_mutex_);
        pending_copies_.push_back(pc);
    }
}

void TerrainRenderer::destroy_chunk(const ChunkKey& key) {
    ChunkArena::Slot slot;
    {
        std::lock_guard lk(live_slots_mutex_);
        auto it = live_slots_.find(key);
        if (it == live_slots_.end()) return;
        slot = it->second;
        live_slots_.erase(it);
    }
    metadata_->clear(slot);
    arena_->free(slot);
}

void TerrainRenderer::render(VkCommandBuffer cmd, const glm::mat4& mvp,
                             const std::array<glm::vec4, 6>& frustum) {
    if (!atlas_set_) return;  // can't sample without an atlas bound

    // 1. Drain pending staging -> arena copies. These must happen
    //    OUTSIDE any active render pass; the caller's contract is that
    //    render() is invoked before vkCmdBeginRendering for this frame.
    {
        std::vector<PendingCopy> copies;
        {
            std::lock_guard lk(pending_mutex_);
            copies.swap(pending_copies_);
        }
        for (const auto& pc : copies) {
            VkBufferCopy region{pc.staging_offset, pc.arena_offset, pc.bytes};
            vkCmdCopyBuffer(cmd, staging_->buffer(), arena_->buffer(),
                            1, &region);
        }
    }

    // 2. Metadata dirty rows -> SSBO. Same transfer stage as the copies.
    metadata_->flush(cmd);
    indirect_->reset_count(cmd);

    // 3. Barrier: TRANSFER writes are visible to COMPUTE reads (metadata,
    //    count) and to vertex INDEX/ATTRIBUTE reads (arena).
    {
        VkBufferMemoryBarrier2 bb[3]{};
        for (auto& b : bb) {
            b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            b.size = VK_WHOLE_SIZE;
            b.srcStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
            b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        }
        bb[0].buffer = arena_->buffer();
        bb[0].dstStageMask  = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT;
        bb[0].dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
        bb[1].buffer = metadata_->buffer();
        bb[1].dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                              VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
        bb[1].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        bb[2].buffer = indirect_->count_buffer();
        bb[2].dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        bb[2].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                              VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.bufferMemoryBarrierCount = 3;
        dep.pBufferMemoryBarriers = bb;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // 4. Cull dispatch.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cull_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            cull_pipeline_layout_, 0, 1, &cull_set_, 0,
                            nullptr);
    struct CullPC {
        glm::vec4 frustum[6];
        uint32_t  slot_count;
        uint32_t  max_draws;
    } cpc{};
    for (int i = 0; i < 6; ++i) cpc.frustum[i] = frustum[i];
    cpc.slot_count = arena_->slot_count();
    cpc.max_draws  = indirect_->max_draws();
    vkCmdPushConstants(cmd, cull_pipeline_layout_,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cpc), &cpc);
    uint32_t groups = (cpc.slot_count + 63u) / 64u;
    vkCmdDispatch(cmd, groups, 1, 1);

    // 5. Barrier: cull's writes -> indirect command read + draw counter
    //    indirect read.
    {
        VkBufferMemoryBarrier2 bb[2]{};
        for (auto& b : bb) {
            b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            b.size = VK_WHOLE_SIZE;
            b.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            b.dstStageMask  = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
            b.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        }
        bb[0].buffer = indirect_->draws_buffer();
        bb[1].buffer = indirect_->count_buffer();
        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.bufferMemoryBarrierCount = 2;
        dep.pBufferMemoryBarriers = bb;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // 6. Indirect draw. Caller is expected to have already begun a render
    //    pass with the right colour/depth attachments matching cfg_.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, terrain_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            terrain_pipeline_layout_, 0, 1, &terrain_set_, 0,
                            nullptr);
    VkDeviceSize zero = 0;
    VkBuffer arena_buf = arena_->buffer();
    vkCmdBindVertexBuffers(cmd, 0, 1, &arena_buf, &zero);
    vkCmdPushConstants(cmd, terrain_pipeline_layout_,
                       VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &mvp);
    vkCmdDrawIndirectCount(cmd, indirect_->draws_buffer(), 0,
                           indirect_->count_buffer(), 0,
                           indirect_->max_draws(),
                           uint32_t(IndirectDrawBuffer::kStride));
}

void TerrainRenderer::release_staging(uint64_t checkpoint) {
    staging_->release_up_to(checkpoint);
}

}  // namespace plce::vk_render
