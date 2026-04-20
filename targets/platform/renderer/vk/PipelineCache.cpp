#include "PipelineCache.h"
#include "VkCheck.h"

#include <array>
#include <cstdio>
#include <cstdint>
#include <vector>

namespace plce::vk {

namespace {

VkShaderModule make_module(VkDevice dev, const uint32_t* code, size_t bytes) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = bytes;
    ci.pCode    = code;
    VkShaderModule m = VK_NULL_HANDLE;
    check(vkCreateShaderModule(dev, &ci, nullptr, &m), "shader module");
    return m;
}

// All per-pipeline state referenced by VkGraphicsPipelineCreateInfo through
// pointers. Instances are populated by `populate()` and must stay in memory
// until vkCreateGraphicsPipelines returns — that lets us batch several
// pipelines into one driver call without copying vertex-input descriptions
// or dynamic-state arrays into a separate scratch arena.
struct PipelineBuild {
    VkPipelineShaderStageCreateInfo           stages[2]{};
    VkVertexInputBindingDescription           binding_standard{0, 32, VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription         attrs_standard[5]{
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},   // pos
        {1, 0, VK_FORMAT_R32G32_SFLOAT,    12},  // uv
        {2, 0, VK_FORMAT_R8G8B8A8_UNORM,   20},  // color
        {3, 0, VK_FORMAT_R8G8B8A8_SNORM,   24},  // normal
        {4, 0, VK_FORMAT_R16G16_SINT,      28},  // lightmap UVs
    };
    VkVertexInputBindingDescription           binding_compact{0, 16, VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription         attrs_compact[2]{
        {0, 0, VK_FORMAT_R16G16B16A16_SINT, 0},  // (pos.xyz, color 5-6-5)
        {1, 0, VK_FORMAT_R16G16B16A16_SINT, 8},  // (uv.xy,   lm.uv)
    };
    VkPipelineVertexInputStateCreateInfo      vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo    ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    VkPipelineViewportStateCreateInfo         vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    VkPipelineRasterizationStateCreateInfo    rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    VkPipelineMultisampleStateCreateInfo      ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    VkPipelineDepthStencilStateCreateInfo     ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState       att{};
    VkPipelineColorBlendStateCreateInfo       cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    VkDynamicState                            dyn_states[5]{
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
        VK_DYNAMIC_STATE_BLEND_CONSTANTS,
        VK_DYNAMIC_STATE_DEPTH_BIAS,
    };
    VkPipelineDynamicStateCreateInfo          dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    VkPipelineRenderingCreateInfo             prci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    VkGraphicsPipelineCreateInfo              gci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
};

void populate(const PipelineCache::Config& cfg, const PipelineKey& key,
              VkShaderModule vert_mod, VkShaderModule vert_compact_mod,
              VkShaderModule frag_mod, PipelineBuild& b) {
    b.stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    b.stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    b.stages[0].module = key.compact() ? vert_compact_mod : vert_mod;
    b.stages[0].pName  = "main";
    b.stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    b.stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    b.stages[1].module = frag_mod;
    b.stages[1].pName  = "main";

    b.vi.vertexBindingDescriptionCount = 1;
    if (key.compact()) {
        b.vi.pVertexBindingDescriptions      = &b.binding_compact;
        b.vi.vertexAttributeDescriptionCount = 2;
        b.vi.pVertexAttributeDescriptions    = b.attrs_compact;
    } else {
        b.vi.pVertexBindingDescriptions      = &b.binding_standard;
        b.vi.vertexAttributeDescriptionCount = 5;
        b.vi.pVertexAttributeDescriptions    = b.attrs_standard;
    }

    b.ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    b.vp.viewportCount = 1;
    b.vp.scissorCount  = 1;

    b.rs.polygonMode     = VK_POLYGON_MODE_FILL;
    b.rs.cullMode        = key.cull_back() ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
    b.rs.frontFace       = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    b.rs.lineWidth       = 1.0f;
    b.rs.depthBiasEnable = VK_TRUE;  // dynamic depth bias via vkCmdSetDepthBias

    b.ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    b.ds.depthTestEnable  = key.depth_test()  ? VK_TRUE : VK_FALSE;
    b.ds.depthWriteEnable = key.depth_write() ? VK_TRUE : VK_FALSE;
    b.ds.depthCompareOp   = VkCompareOp(key.depth_func());

    const uint8_t cmask = key.color_mask();
    b.att.blendEnable         = key.blend_enable() ? VK_TRUE : VK_FALSE;
    b.att.srcColorBlendFactor = VkBlendFactor(key.blend_src());
    b.att.dstColorBlendFactor = VkBlendFactor(key.blend_dst());
    b.att.colorBlendOp        = VK_BLEND_OP_ADD;
    b.att.srcAlphaBlendFactor = VkBlendFactor(key.blend_src());
    b.att.dstAlphaBlendFactor = VkBlendFactor(key.blend_dst());
    b.att.alphaBlendOp        = VK_BLEND_OP_ADD;
    b.att.colorWriteMask =
        ((cmask & 0x1) ? VK_COLOR_COMPONENT_R_BIT : 0) |
        ((cmask & 0x2) ? VK_COLOR_COMPONENT_G_BIT : 0) |
        ((cmask & 0x4) ? VK_COLOR_COMPONENT_B_BIT : 0) |
        ((cmask & 0x8) ? VK_COLOR_COMPONENT_A_BIT : 0);
    b.cb.attachmentCount = 1;
    b.cb.pAttachments    = &b.att;

    b.dyn.dynamicStateCount = uint32_t(std::size(b.dyn_states));
    b.dyn.pDynamicStates    = b.dyn_states;

    b.prci.colorAttachmentCount    = 1;
    b.prci.pColorAttachmentFormats = &cfg.color_format;
    b.prci.depthAttachmentFormat   = cfg.depth_format;

    b.gci.pNext               = &b.prci;
    b.gci.stageCount          = 2;
    b.gci.pStages             = b.stages;
    b.gci.pVertexInputState   = &b.vi;
    b.gci.pInputAssemblyState = &b.ia;
    b.gci.pViewportState      = &b.vp;
    b.gci.pRasterizationState = &b.rs;
    b.gci.pMultisampleState   = &b.ms;
    b.gci.pDepthStencilState  = &b.ds;
    b.gci.pColorBlendState    = &b.cb;
    b.gci.pDynamicState       = &b.dyn;
    b.gci.layout              = cfg.layout;
}

}  // namespace

void PipelineCache::init(const Config& cfg) {
    cfg_ = cfg;
    vert_mod_ = make_module(cfg.device, cfg.vert_spv, cfg.vert_size);
    frag_mod_ = make_module(cfg.device, cfg.frag_spv, cfg.frag_size);
    if (cfg.vert_compact_spv && cfg.vert_compact_size > 0)
        vert_compact_mod_ = make_module(cfg.device, cfg.vert_compact_spv, cfg.vert_compact_size);

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
    if (frag_mod_)         vkDestroyShaderModule(cfg_.device, frag_mod_, nullptr);
    if (vert_compact_mod_) vkDestroyShaderModule(cfg_.device, vert_compact_mod_, nullptr);
    if (vert_mod_)         vkDestroyShaderModule(cfg_.device, vert_mod_, nullptr);
    vert_mod_ = vert_compact_mod_ = frag_mod_ = VK_NULL_HANDLE;
}

VkPipeline PipelineCache::get(const PipelineKey& key) {
    auto it = cache_.find(key);
    if (it != cache_.end()) return it->second;
    VkPipeline p = create(key);
    cache_.emplace(key, p);
    return p;
}

void PipelineCache::warm_up() {
    // Collect the warm-up keys so we can issue one batched
    // vkCreateGraphicsPipelines call — lets drivers compile the variants
    // in parallel instead of serializing six individual submissions.
    std::vector<PipelineKey> keys;
    keys.reserve(8);

    {
        PipelineKey opaque{};
        keys.push_back(opaque);

        PipelineKey opaque_cull = opaque;
        opaque_cull.set_cull_back(true);
        keys.push_back(opaque_cull);

        PipelineKey blend{};
        blend.set_blend_enable(true);
        blend.set_depth_write(false);
        keys.push_back(blend);

        PipelineKey depth_off{};
        depth_off.set_depth_test(false);
        keys.push_back(depth_off);
    }

    if (vert_compact_mod_) {
        PipelineKey compact_opaque{};
        compact_opaque.set_compact(true);
        keys.push_back(compact_opaque);

        PipelineKey compact_cull = compact_opaque;
        compact_cull.set_cull_back(true);
        keys.push_back(compact_cull);
    }

    // Skip any keys already present (defensive — warm_up is expected to
    // run exactly once, but re-entering it should not crash).
    std::vector<PipelineKey> fresh;
    fresh.reserve(keys.size());
    for (const auto& k : keys)
        if (cache_.find(k) == cache_.end()) fresh.push_back(k);
    if (fresh.empty()) return;

    std::vector<PipelineBuild> builds(fresh.size());
    std::vector<VkGraphicsPipelineCreateInfo> cis(fresh.size());
    for (size_t i = 0; i < fresh.size(); ++i) {
        populate(cfg_, fresh[i], vert_mod_, vert_compact_mod_, frag_mod_, builds[i]);
        cis[i] = builds[i].gci;
    }

    std::vector<VkPipeline> out(fresh.size(), VK_NULL_HANDLE);
    check(vkCreateGraphicsPipelines(cfg_.device, vk_cache_,
                                    uint32_t(cis.size()), cis.data(),
                                    nullptr, out.data()),
          "graphics pipelines (warm up)");
    for (size_t i = 0; i < fresh.size(); ++i)
        cache_.emplace(fresh[i], out[i]);

    std::fprintf(stderr, "[vk] warmed %zu pipelines\n", cache_.size());
}

VkPipeline PipelineCache::create(const PipelineKey& key) {
    PipelineBuild b;
    populate(cfg_, key, vert_mod_, vert_compact_mod_, frag_mod_, b);

    VkPipeline pipeline = VK_NULL_HANDLE;
    check(vkCreateGraphicsPipelines(cfg_.device, vk_cache_, 1, &b.gci,
                                    nullptr, &pipeline),
          "graphics pipeline");
    return pipeline;
}

void PipelineCache::load_cache(std::span<const std::uint8_t> blob) {
    if (vk_cache_) {
        vkDestroyPipelineCache(cfg_.device, vk_cache_, nullptr);
        vk_cache_ = VK_NULL_HANDLE;
    }
    VkPipelineCacheCreateInfo pcci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    pcci.initialDataSize = blob.size();
    pcci.pInitialData    = blob.empty() ? nullptr : blob.data();
    VkResult r = vkCreatePipelineCache(cfg_.device, &pcci, nullptr, &vk_cache_);
    if (r != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] pipeline cache load failed (VkResult=%d), using empty cache\n", int(r));
        pcci.initialDataSize = 0;
        pcci.pInitialData    = nullptr;
        check(vkCreatePipelineCache(cfg_.device, &pcci, nullptr, &vk_cache_), "pipeline cache fallback");
    }
}

std::vector<std::uint8_t> PipelineCache::save_cache() const {
    if (!vk_cache_) return {};
    size_t sz = 0;
    if (vkGetPipelineCacheData(cfg_.device, vk_cache_, &sz, nullptr) != VK_SUCCESS || sz == 0)
        return {};
    std::vector<std::uint8_t> blob(sz);
    if (vkGetPipelineCacheData(cfg_.device, vk_cache_, &sz, blob.data()) != VK_SUCCESS)
        return {};
    return blob;
}

}  // namespace plce::vk
