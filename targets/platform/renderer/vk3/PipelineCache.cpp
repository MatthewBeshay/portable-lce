#include "PipelineCache.h"

#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace plce::vk3 {

namespace {
void check(VkResult r, const char* msg) {
    if (r != VK_SUCCESS) {
        char buf[128];
        std::snprintf(buf, sizeof buf, "%s: VkResult=%d", msg, int(r));
        throw std::runtime_error(buf);
    }
}

VkShaderModule make_module(VkDevice dev, const uint32_t* code, size_t bytes) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = bytes;
    ci.pCode    = code;
    VkShaderModule m = VK_NULL_HANDLE;
    check(vkCreateShaderModule(dev, &ci, nullptr, &m), "shader module");
    return m;
}
}  // namespace

void PipelineCache::init(const Config& cfg) {
    cfg_ = cfg;
    vert_mod_ = make_module(cfg.device, cfg.vert_spv, cfg.vert_size);
    frag_mod_ = make_module(cfg.device, cfg.frag_spv, cfg.frag_size);

    VkPipelineCacheCreateInfo pcci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    check(vkCreatePipelineCache(cfg.device, &pcci, nullptr, &vk_cache_), "pipeline cache");
}

void PipelineCache::destroy() {
    for (auto& [k, p] : cache_)
        vkDestroyPipeline(cfg_.device, p, nullptr);
    cache_.clear();
    if (vk_cache_) {
        vkDestroyPipelineCache(cfg_.device, vk_cache_, nullptr);
        vk_cache_ = VK_NULL_HANDLE;
    }
    if (frag_mod_) vkDestroyShaderModule(cfg_.device, frag_mod_, nullptr);
    if (vert_mod_) vkDestroyShaderModule(cfg_.device, vert_mod_, nullptr);
    vert_mod_ = frag_mod_ = VK_NULL_HANDLE;
}

VkPipeline PipelineCache::get(const PipelineKey& key) {
    auto it = cache_.find(key);
    if (it != cache_.end()) return it->second;
    VkPipeline p = create(key);
    cache_.emplace(key, p);
    return p;
}

void PipelineCache::warm_up() {
    PipelineKey opaque{};
    get(opaque);

    PipelineKey opaque_cull = opaque;
    opaque_cull.cull_back = 1;
    get(opaque_cull);

    PipelineKey blend{};
    blend.blend_enable = 1;
    blend.depth_write  = 0;
    get(blend);

    PipelineKey depth_off{};
    depth_off.depth_test = 0;
    get(depth_off);

    PipelineKey lines_key{};
    lines_key.lines = 1;
    get(lines_key);

    std::fprintf(stderr, "[vk3] warmed %zu pipelines\n", cache_.size());
}

VkPipeline PipelineCache::create(const PipelineKey& key) {
    // Shader stages
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert_mod_;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag_mod_;
    stages[1].pName  = "main";

    // Vertex input: 32-byte WorldStandardVertex
    VkVertexInputBindingDescription binding{0, 32, VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attrs[5]{
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},   // pos
        {1, 0, VK_FORMAT_R32G32_SFLOAT,    12},  // uv
        {2, 0, VK_FORMAT_R8G8B8A8_UNORM,   20},  // color
        {3, 0, VK_FORMAT_R8G8B8A8_SNORM,   24},  // normal
        {4, 0, VK_FORMAT_R16G16_SINT,      28},  // lightmap UVs
    };
    VkPipelineVertexInputStateCreateInfo vi{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount   = 1;
    vi.pVertexBindingDescriptions      = &binding;
    vi.vertexAttributeDescriptionCount = 5;
    vi.pVertexAttributeDescriptions    = attrs;

    // Input assembly — topology is dynamic
    VkPipelineInputAssemblyStateCreateInfo ia{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // Viewport + scissor — dynamic
    VkPipelineViewportStateCreateInfo vp{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount  = 1;

    // Rasterization — CW front face (Y-flip reverses winding)
    VkPipelineRasterizationStateCreateInfo rs{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode    = VK_POLYGON_MODE_FILL;
    rs.cullMode       = VK_CULL_MODE_NONE;  // TODO: debug — disable culling
    rs.frontFace      = VK_FRONT_FACE_CLOCKWISE;
    rs.lineWidth      = 1.0f;
    rs.depthBiasEnable = VK_TRUE;  // dynamic depth bias via vkCmdSetDepthBias

    // Multisample
    VkPipelineMultisampleStateCreateInfo ms{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Depth/stencil
    VkPipelineDepthStencilStateCreateInfo ds{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable  = key.depth_test  ? VK_TRUE : VK_FALSE;
    ds.depthWriteEnable = key.depth_write ? VK_TRUE : VK_FALSE;
    ds.depthCompareOp   = VkCompareOp(key.depth_func);

    // Blend
    VkPipelineColorBlendAttachmentState att{};
    att.blendEnable         = key.blend_enable ? VK_TRUE : VK_FALSE;
    att.srcColorBlendFactor = VkBlendFactor(key.blend_src);
    att.dstColorBlendFactor = VkBlendFactor(key.blend_dst);
    att.colorBlendOp        = VK_BLEND_OP_ADD;
    att.srcAlphaBlendFactor = VkBlendFactor(key.blend_src);
    att.dstAlphaBlendFactor = VkBlendFactor(key.blend_dst);
    att.alphaBlendOp        = VK_BLEND_OP_ADD;
    att.colorWriteMask =
        ((key.color_mask & 0x1) ? VK_COLOR_COMPONENT_R_BIT : 0) |
        ((key.color_mask & 0x2) ? VK_COLOR_COMPONENT_G_BIT : 0) |
        ((key.color_mask & 0x4) ? VK_COLOR_COMPONENT_B_BIT : 0) |
        ((key.color_mask & 0x8) ? VK_COLOR_COMPONENT_A_BIT : 0);

    VkPipelineColorBlendStateCreateInfo cb{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments    = &att;

    // Dynamic state
    VkDynamicState dyn_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
        VK_DYNAMIC_STATE_BLEND_CONSTANTS,
        VK_DYNAMIC_STATE_DEPTH_BIAS,
    };
    VkPipelineDynamicStateCreateInfo dyn{
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = uint32_t(std::size(dyn_states));
    dyn.pDynamicStates    = dyn_states;

    // Dynamic rendering (no VkRenderPass)
    VkPipelineRenderingCreateInfo prci{
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    prci.colorAttachmentCount    = 1;
    prci.pColorAttachmentFormats = &cfg_.color_format;
    prci.depthAttachmentFormat   = cfg_.depth_format;

    // Assemble
    VkGraphicsPipelineCreateInfo gci{
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gci.pNext               = &prci;
    gci.stageCount          = 2;
    gci.pStages             = stages;
    gci.pVertexInputState   = &vi;
    gci.pInputAssemblyState = &ia;
    gci.pViewportState      = &vp;
    gci.pRasterizationState = &rs;
    gci.pMultisampleState   = &ms;
    gci.pDepthStencilState  = &ds;
    gci.pColorBlendState    = &cb;
    gci.pDynamicState       = &dyn;
    gci.layout              = cfg_.layout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    check(vkCreateGraphicsPipelines(cfg_.device, vk_cache_, 1, &gci,
                                    nullptr, &pipeline),
          "graphics pipeline");
    return pipeline;
}

void PipelineCache::load_cache(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz <= 0) { std::fclose(f); return; }
    std::vector<uint8_t> blob(static_cast<size_t>(sz));
    std::fread(blob.data(), 1, blob.size(), f);
    std::fclose(f);
    if (vk_cache_) {
        vkDestroyPipelineCache(cfg_.device, vk_cache_, nullptr);
        vk_cache_ = VK_NULL_HANDLE;
    }
    VkPipelineCacheCreateInfo pcci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    pcci.initialDataSize = blob.size();
    pcci.pInitialData    = blob.data();
    VkResult r = vkCreatePipelineCache(cfg_.device, &pcci, nullptr, &vk_cache_);
    if (r != VK_SUCCESS) {
        std::fprintf(stderr, "[vk3] pipeline cache load failed (VkResult=%d), using empty cache\n", int(r));
        pcci.initialDataSize = 0;
        pcci.pInitialData    = nullptr;
        check(vkCreatePipelineCache(cfg_.device, &pcci, nullptr, &vk_cache_), "pipeline cache fallback");
    }
}

void PipelineCache::save_cache(const char* path) {
    if (!vk_cache_) return;
    size_t sz = 0;
    VkResult r = vkGetPipelineCacheData(cfg_.device, vk_cache_, &sz, nullptr);
    if (r != VK_SUCCESS || sz == 0) return;
    std::vector<uint8_t> blob(sz);
    r = vkGetPipelineCacheData(cfg_.device, vk_cache_, &sz, blob.data());
    if (r != VK_SUCCESS) return;
    FILE* f = std::fopen(path, "wb");
    if (!f) return;
    std::fwrite(blob.data(), 1, sz, f);
    std::fclose(f);
}

}  // namespace plce::vk3
