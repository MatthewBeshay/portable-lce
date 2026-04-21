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
    VkDynamicState                            dyn_states[9]{
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
        VK_DYNAMIC_STATE_BLEND_CONSTANTS,
        VK_DYNAMIC_STATE_DEPTH_BIAS,
        VK_DYNAMIC_STATE_LINE_WIDTH,
        VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
        VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
        VK_DYNAMIC_STATE_STENCIL_REFERENCE,
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
    b.ds.stencilTestEnable = key.stencil_test() ? VK_TRUE : VK_FALSE;
    // Stencil ops: fail=KEEP, pass=REPLACE (writeMask gates the actual
    // write), depthFail=KEEP. compare/write masks and reference are
    // dynamic (vkCmdSetStencilCompareMask / WriteMask / Reference).
    b.ds.front.failOp      = VK_STENCIL_OP_KEEP;
    b.ds.front.passOp      = VK_STENCIL_OP_REPLACE;
    b.ds.front.depthFailOp = VK_STENCIL_OP_KEEP;
    b.ds.front.compareOp   = VkCompareOp(key.stencil_func());
    b.ds.back              = b.ds.front;

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
    b.prci.stencilAttachmentFormat = cfg.stencil_format;
    // Note: b.prci.pNext left null; pipeline rendering struct is
    // chained into VkGraphicsPipelineCreateInfo below via b.gci.pNext.

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
    for (auto& e : cache_slots_) {
        if (e.pipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(cfg_.device, e.pipeline, nullptr);
    }
    cache_slots_.clear();
    cache_mask_  = 0;
    cache_count_ = 0;
    if (vk_cache_) {
        vkDestroyPipelineCache(cfg_.device, vk_cache_, nullptr);
        vk_cache_ = VK_NULL_HANDLE;
    }
    if (frag_mod_)         vkDestroyShaderModule(cfg_.device, frag_mod_, nullptr);
    if (vert_compact_mod_) vkDestroyShaderModule(cfg_.device, vert_compact_mod_, nullptr);
    if (vert_mod_)         vkDestroyShaderModule(cfg_.device, vert_mod_, nullptr);
    vert_mod_ = vert_compact_mod_ = frag_mod_ = VK_NULL_HANDLE;
}

// Linear probe inside the flat table. Returns the slot index for
// `key` — either the entry that contains it, or the first empty slot
// where it would be inserted. Caller checks pipeline handle to
// distinguish.
size_t PipelineCache::probe(const std::vector<Entry>& slots,
                            size_t mask, uint64_t key) {
    // Fibonacci hashing — multiply by 2^64 / phi. The high bits of the
    // product have the best distribution; anding with mask picks a
    // subset that's good enough for our small tables.
    size_t idx = size_t(key * 0x9E3779B97F4A7C15ull) & mask;
    while (slots[idx].pipeline != VK_NULL_HANDLE && slots[idx].key != key) {
        idx = (idx + 1) & mask;
    }
    return idx;
}

void PipelineCache::rehash(size_t new_cap) {
    std::vector<Entry> new_slots(new_cap);
    const size_t new_mask = new_cap - 1;
    for (const auto& e : cache_slots_) {
        if (e.pipeline == VK_NULL_HANDLE) continue;
        size_t idx = probe(new_slots, new_mask, e.key);
        new_slots[idx] = e;
    }
    cache_slots_ = std::move(new_slots);
    cache_mask_  = new_mask;
}

VkPipeline PipelineCache::get(const PipelineKey& key) {
    if (cache_slots_.empty()) rehash(16);  // initial power-of-two capacity
    size_t idx = probe(cache_slots_, cache_mask_, key.bits);
    if (cache_slots_[idx].pipeline != VK_NULL_HANDLE)
        return cache_slots_[idx].pipeline;
    // Miss — grow if needed before the insert (load factor 0.5).
    if ((cache_count_ + 1) * 2 > cache_slots_.size()) {
        rehash(cache_slots_.size() * 2);
        idx = probe(cache_slots_, cache_mask_, key.bits);
    }
    VkPipeline p = create(key);
    cache_slots_[idx] = {key.bits, p};
    ++cache_count_;
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
    if (cache_slots_.empty()) rehash(16);
    std::vector<PipelineKey> fresh;
    fresh.reserve(keys.size());
    for (const auto& k : keys) {
        size_t idx = probe(cache_slots_, cache_mask_, k.bits);
        if (cache_slots_[idx].pipeline == VK_NULL_HANDLE)
            fresh.push_back(k);
    }
    if (fresh.empty()) return;

    // Grow once up-front so probe indices stay valid through the inserts.
    while ((cache_count_ + fresh.size()) * 2 > cache_slots_.size())
        rehash(cache_slots_.size() * 2);

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
    for (size_t i = 0; i < fresh.size(); ++i) {
        size_t idx = probe(cache_slots_, cache_mask_, fresh[i].bits);
        cache_slots_[idx] = {fresh[i].bits, out[i]};
        ++cache_count_;
    }

    std::fprintf(stderr, "[vk] warmed %zu pipelines\n", cache_count_);
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
