#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "platform/renderer/IRenderPath.h"

struct SDL_Window;
struct VmaAllocator_T;
struct VmaAllocation_T;
using VmaAllocator = VmaAllocator_T*;
using VmaAllocation = VmaAllocation_T*;

namespace plce::vk_render { class TerrainRenderer; }

class VulkanRenderPath final : public rp::IRenderPath {
public:
    struct CBuffDraw {
        int primType   = 0;
        int vertexType = 0;
        int shaderType = 0;
        std::vector<std::byte> verts;
    };

    explicit VulkanRenderPath(SDL_Window* window);
    ~VulkanRenderPath() override;

    VulkanRenderPath(const VulkanRenderPath&) = delete;
    VulkanRenderPath& operator=(const VulkanRenderPath&) = delete;

    // -- Lifecycle --
    void StartFrame() override;
    void Present() override;
    void Clear(int flags) override;
    void SetClearColour(const float rgba[4]) override;
    void render_frame(const rp::FrameDesc& frame) override;
    void resize(uint32_t w, uint32_t h) override;
    void GetFramebufferSize(int& w, int& h) override;
    void SetWindowSize(int w, int h) override;
    void SetFullscreen(bool fs) override;
    void Close() override;
    [[nodiscard]] bool ShouldClose() override;
    [[nodiscard]] const rp::FrameFramebuffer& framebuffer() const override;
    [[nodiscard]] bool IsWidescreen() override;
    [[nodiscard]] bool IsHiDef() override;

    // -- Matrix stack --
    void MatrixMode(rp::MatrixStack stack) override;
    void MatrixSetIdentity() override;
    void MatrixTranslate(float x, float y, float z) override;
    void MatrixRotate(float angle, float x, float y, float z) override;
    void MatrixScale(float x, float y, float z) override;
    void MatrixPerspective(float fovy, float aspect, float zNear,
                           float zFar) override;
    void MatrixOrthogonal(float left, float right, float bottom, float top,
                          float zNear, float zFar) override;
    void MatrixPop() override;
    void MatrixPush() override;
    void MatrixMult(float* mat) override;
    [[nodiscard]] const float* MatrixGet(rp::MatrixStack stack) override;

    // -- Draw --
    void DrawVertices(int primitiveType, int count, void* data, int vertexType,
                      int shaderType) override;

    // -- Resource handles (stubbed) --
    [[nodiscard]] rp::MeshHandle create_mesh(const rp::MeshDesc&) override {
        return rp::kInvalidMesh;
    }
    void update_mesh(rp::MeshHandle, const rp::MeshDesc&) override {}
    void destroy_mesh(rp::MeshHandle) override {}

    [[nodiscard]] rp::TextureHandle create_texture(
        const rp::TextureDesc&) override {
        return rp::kInvalidTexture;
    }
    void update_texture(rp::TextureHandle, const rp::TextureRegion&) override {}
    void destroy_texture(rp::TextureHandle) override {}

    [[nodiscard]] rp::MaterialHandle create_material(
        const rp::MaterialDesc&) override {
        return rp::kInvalidMaterial;
    }
    void update_material(rp::MaterialHandle, const rp::MaterialDesc&) override {
    }
    void destroy_material(rp::MaterialHandle) override {}

    [[nodiscard]] std::pair<rp::TransientVertexBuffer, std::span<std::byte>>
    alloc_transient_vertices(uint32_t, rp::VertexLayout,
                             rp::PrimitiveType) override {
        return {{}, {}};
    }

    void read_framebuffer(const rp::TextureReadback&) override {}
    [[nodiscard]] rp::ResourceFootprint query_resource_footprint()
        const override {
        return {};
    }
    void seal_static_resource_tier() override {}
    void begin_atomic_resource_batch() override {}
    void end_atomic_resource_batch() override {}
    void push_debug_event(const char*) override {}
    void pop_debug_event() override {}
    void tick() override {}

    // -- CBuff*: simple record-on-CPU / replay-as-DrawVertices --
    [[nodiscard]] int CBuffCreate(int n) override;
    void CBuffDelete(int, int) override {}
    void CBuffDeleteAll() override;
    void CBuffStart(int index, bool full) override;
    void CBuffClear(int index) override;
    [[nodiscard]] int CBuffSize(int index) override;
    void CBuffEnd() override;
    [[nodiscard]] bool CBuffCall(int index, bool full) override;
    void CBuffDeferredModeStart() override {}
    void CBuffDeferredModeEnd() override {}

    [[nodiscard]] int TextureCreate() override;
    void TextureFree(int idx) override;
    void TextureBind(int idx) override;
    // Lightmap binding for chunks (TerrainRenderer's set 0 binding 2).
    // Doesn't affect the main bound_texture_ used by basic-pipeline draws.
    void TextureBindVertex(int idx, bool) override {
        lightmap_texture_ = idx;
    }
    void TextureSetTextureLevels(int) override {}
    void TextureData(int width, int height, void* data, int level,
                     int format) override;
    void TextureDataUpdate(int, int, int, int, void*, int) override {}
    void TextureSetParam(int, int) override {}
    [[nodiscard]] int TextureGetTextureLevels() override { return 1; }

    void StateSetColour(float r, float g, float b, float a) override;
    void StateSetDepthMask(bool e) override;
    void StateSetBlendEnable(bool e) override;
    void StateSetBlendFunc(rp::BlendFactor s, rp::BlendFactor d) override;
    void StateSetBlendFactor(unsigned int) override {}
    void StateSetAlphaFunc(rp::AlphaTest f, float ref) override {
        alpha_test_func_ = f;
        alpha_ref_ = ref;
    }
    void StateSetDepthFunc(rp::DepthTest f) override;
    void StateSetFaceCull(bool e) override;
    void StateSetLineWidth(float) override {}
    void StateSetWriteEnable(bool, bool, bool, bool) override {}
    void StateSetDepthTestEnable(bool e) override;
    void StateSetAlphaTestEnable(bool e) override { alpha_test_enabled_ = e; }
    void StateSetDepthSlopeAndBias(float, float) override {}
    void StateSetFogEnable(bool e) override { fog_enabled_ = e; }
    void StateSetFogMode(rp::FogMode m) override { fog_mode_ = m; }
    void StateSetFogNearDistance(float d) override { fog_start_ = d; }
    void StateSetFogFarDistance(float d) override  { fog_end_   = d; }
    void StateSetFogDensity(float d) override      { fog_density_ = d; }
    void StateSetFogColour(float r, float g, float b) override {
        fog_colour_ = {r, g, b, 1.0f};
    }
    void StateSetLightingEnable(bool e) override { lighting_enabled_ = e; }
    void StateSetLightColour(int, float, float, float) override {}
    void StateSetLightAmbientColour(float, float, float) override {}
    void StateSetLightDirection(int, float, float, float) override {}
    void StateSetLightEnable(int, bool) override {}
    void StateSetViewport(int) override {}
    void StateSetEnableViewportClipPlanes(bool) override {}
    void StateSetStencil(int, uint8_t, uint8_t, uint8_t) override {}
    void StateSetForceLOD(int) override {}
    void StateSetTextureEnable(bool) override {}
    void StateSetActiveTexture(int) override {}
    void StateSetVertexTextureUV(float, float) override {}

    // Chunk offset support requires per-CBuff state capture (Tesselator
    // emits chunk-local vertices but the offset is set on the main thread,
    // outside the recording). Wiring that up properly is Phase 4 work; for
    // now ignore the offset rather than apply a stale main-thread value to
    // every replayed chunk draw.
    void SetChunkOffset(float, float, float) override {}

    void ReadPixels(int, int, int, int, void*) override {}
    [[nodiscard]] int LoadTextureData(const char* filename, void* srcInfo,
                                      int** dataOut) override;
    [[nodiscard]] int LoadTextureData(uint8_t* data, uint32_t bytes,
                                      void* srcInfo, int** dataOut) override;

    void Set_matrixDirty() override {}
    void CBuffLockStaticCreations() override {}
    void UpdateGamma(unsigned short) override {}
    void Suspend() override {}
    [[nodiscard]] bool Suspended() override { return false; }
    void Resume() override {}
    void BeginEvent(const char*) override {}
    void EndEvent() override {}
    void submit_immediate(const rp::DrawCall&) override {}

    void chunk_upload(const ChunkUpload&) override;
    void chunk_destroy(int32_t cx, int32_t cy, int32_t cz,
                       uint8_t layer) override;
    void chunk_upload_from_cbuff(int cbuff_id,
                                 const ChunkUpload& base) override;
    void render_terrain(const float* mvp_4x4,
                        const float* frustum_24) override;
    void set_terrain_atlas(int texture_id) override;

private:
    // 1 frame in flight = serialise CPU/GPU. We share the chunk arena
    // between frames; with 2 in flight, frame N+1's staging->arena copy
    // can race frame N's vertex read of the same slot, producing visible
    // flicker. Serialising fixes the race; the throughput cost is small
    // for the kind of scenes we render.
    static constexpr uint32_t kFramesInFlight = 1;
    static constexpr VkDeviceSize kTransientVbSize = 16ull * 1024 * 1024;

    struct PerFrame {
        VkCommandPool   pool   = VK_NULL_HANDLE;
        VkCommandBuffer cmd    = VK_NULL_HANDLE;
        VkSemaphore     image_acquired = VK_NULL_HANDLE;
        VkSemaphore     render_done    = VK_NULL_HANDLE;
        VkFence         in_flight      = VK_NULL_HANDLE;
        VkBuffer        transient_vb   = VK_NULL_HANDLE;
        VmaAllocation   transient_alloc = nullptr;
        std::byte*      transient_mapped = nullptr;
        VkDeviceSize    transient_offset = 0;
    };

    void create_instance(bool enable_validation);
    void create_surface();
    void pick_physical_device();
    void create_device();
    void create_allocator();
    void create_swapchain(uint32_t width, uint32_t height);
    void destroy_swapchain();
    void create_pipeline_layout();
    void destroy_pipeline_layout();
    void destroy_all_pipelines();
    void ensure_render_pass(PerFrame& f);
    void create_quad_index_buffer();
    void destroy_quad_index_buffer();
    void create_depth_image(uint32_t width, uint32_t height);
    void destroy_depth_image();
    void create_per_frame();
    void destroy_per_frame();
    void create_texture_resources();
    void destroy_texture_resources();
    int  ensure_default_texture();
    void upload_texture(int idx, int width, int height, const void* pixels);
    void begin_render_pass(PerFrame& f);
    void end_render_pass(PerFrame& f);

public:
    // PSO key - one pipeline per state combination, cached on first use.
    struct PsoKey {
        bool depth_test    = true;
        bool depth_write   = true;
        bool blend_enable  = false;
        bool cull_back     = false;
        bool lines         = false;  // line-class topology (LINE_LIST/STRIP)
        // VK_COMPARE_OP_LESS_OR_EQUAL = 3. Matches the bgfx renderer's
        // default and the legacy GL state the game targets.
        uint8_t depth_func = 3;
        // Default blend factors: SRC_ALPHA / ONE_MINUS_SRC_ALPHA. Standard
        // alpha blending; only takes effect when blend_enable is true.
        uint8_t blend_src  = 6;   // VK_BLEND_FACTOR_SRC_ALPHA
        uint8_t blend_dst  = 7;   // VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA
        bool operator==(const PsoKey&) const = default;
    };
    struct PsoKeyHash {
        size_t operator()(const PsoKey& k) const noexcept {
            uint64_t v = uint64_t(k.depth_test)
                       | (uint64_t(k.depth_write)  << 1)
                       | (uint64_t(k.blend_enable) << 2)
                       | (uint64_t(k.cull_back)    << 3)
                       | (uint64_t(k.lines)        << 4)
                       | (uint64_t(k.depth_func)   << 8)
                       | (uint64_t(k.blend_src)    << 16)
                       | (uint64_t(k.blend_dst)    << 24);
            return std::hash<uint64_t>{}(v);
        }
    };

private:
    VkPipeline ensure_pipeline(const PsoKey& key);

    SDL_Window* window_       = nullptr;
    bool        should_close_ = false;

    VkInstance       instance_  = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    VkSurfaceKHR     surface_   = VK_NULL_HANDLE;
    VkPhysicalDevice phys_      = VK_NULL_HANDLE;
    VkDevice         device_    = VK_NULL_HANDLE;
    uint32_t         graphics_family_ = 0;
    VkQueue          graphics_queue_  = VK_NULL_HANDLE;
    VmaAllocator     allocator_       = nullptr;

    VkSwapchainKHR   swapchain_       = VK_NULL_HANDLE;
    VkFormat         swapchain_format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D       swapchain_extent_ = {0, 0};
    std::vector<VkImage>     swapchain_images_;
    std::vector<VkImageView> swapchain_views_;

    // Pipeline cache (one PSO per state combination).
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    std::unordered_map<PsoKey, VkPipeline, PsoKeyHash> pipeline_cache_;
    PsoKey current_pso_key_{};
    PsoKey last_bound_pso_key_{};
    bool   pso_key_dirty_ = true;

    // Depth attachment for the swapchain render pass.
    VkImage         depth_image_ = VK_NULL_HANDLE;
    VmaAllocation   depth_alloc_ = nullptr;
    VkImageView     depth_view_  = VK_NULL_HANDLE;
    VkFormat        depth_format_ = VK_FORMAT_D32_SFLOAT;

    // GL-state colour - the modulation colour set via StateSetColour, used
    // when Tesselator emits the 0x00000000 sentinel as the per-vertex colour.
    std::array<float, 4> state_colour_{1.0f, 1.0f, 1.0f, 1.0f};

    // Per-chunk world-space offset. Chunks submit their vertices in
    // chunk-local space and rely on the shader to add this offset.
    std::array<float, 3> chunk_offset_{0.0f, 0.0f, 0.0f};

    // Lighting toggle (set via StateSetLightingEnable). Hard-coded sun
    // direction in the vertex shader for now; lights are not yet wired
    // through StateSetLightDirection / StateSetLightColour.
    bool                 lighting_enabled_ = false;

    // Alpha-test (cutout) state for grass / leaves / fences.
    bool                 alpha_test_enabled_ = false;
    rp::AlphaTest        alpha_test_func_    = rp::AlphaTest::greater;
    float                alpha_ref_          = 0.1f;

    // Fog state (legacy GL-style). Pushed via fragment push constant.
    bool                 fog_enabled_ = false;
    rp::FogMode          fog_mode_    = rp::FogMode::linear;
    float                fog_start_   = 0.0f;
    float                fog_end_     = 1.0f;
    float                fog_density_ = 1.0f;
    std::array<float, 4> fog_colour_{0.5f, 0.7f, 1.0f, 1.0f};

    // Static index buffer that expands GL_QUADS (4 verts) into two triangles
    // per quad (6 indices). Sized for the maximum quad batch we expect.
    VkBuffer         quad_index_buffer_ = VK_NULL_HANDLE;
    VmaAllocation    quad_index_alloc_  = nullptr;
    static constexpr uint32_t kMaxQuadsPerDraw = 16384;

    // Texture system. One descriptor set per texture (combined image
    // sampler), one shared linear-filtering sampler.
    VkDescriptorSetLayout tex_set_layout_     = VK_NULL_HANDLE;
    VkDescriptorPool      tex_pool_           = VK_NULL_HANDLE;
    VkSampler             tex_sampler_        = VK_NULL_HANDLE;  // nearest+repeat (atlas)
    VkSampler             tex_sampler_lm_     = VK_NULL_HANDLE;  // linear+clamp  (lightmap)
    struct TextureSlot {
        VkImage         image    = VK_NULL_HANDLE;
        VmaAllocation   alloc    = nullptr;
        VkImageView     view     = VK_NULL_HANDLE;
        VkDescriptorSet desc_set = VK_NULL_HANDLE;
        uint32_t        width    = 0;
        uint32_t        height   = 0;
        bool            ready    = false;
    };
    std::vector<TextureSlot> textures_;
    int default_texture_ = 0;
    int bound_texture_   = 0;

    std::array<PerFrame, kFramesInFlight> frames_{};
    uint32_t frame_index_    = 0;
    uint32_t acquired_image_ = 0;
    bool     frame_active_   = false;
    bool     pass_active_    = false;
    bool     pipeline_bound_ = false;

    rp::FrameFramebuffer fb_{};
    std::array<float, 4> clear_color_{0.05f, 0.05f, 0.10f, 1.0f};

    // Software matrix stack
    rp::MatrixStack matrix_mode_ = rp::MatrixStack::modelview;
    std::vector<glm::mat4> modelview_stack_{glm::mat4(1.0f)};
    std::vector<glm::mat4> projection_stack_{glm::mat4(1.0f)};
    std::vector<glm::mat4> texture_stack_{glm::mat4(1.0f)};
    glm::mat4 cached_matrix_get_{1.0f};

    int next_handle_ = 0;
    int next_cbuff_  = 1;
    int lightmap_texture_ = 0;

    std::unique_ptr<plce::vk_render::TerrainRenderer> terrain_;

    // Per-frame stats reported once per second to stderr.
    uint32_t stat_draws_total_       = 0;
    uint32_t stat_draws_textured_    = 0;
    uint32_t stat_tex_creates_       = 0;
    uint32_t stat_tex_uploads_       = 0;
    uint32_t stat_tex_binds_         = 0;
    uint32_t stat_frames_            = 0;
    double   stat_window_start_secs_ = 0.0;

    // CBuff* (display list) record/replay store. Recording state is
    // per-thread (chunk meshers run on worker threads); the shared pool is
    // mutex-protected. Same model as the bgfx backend.
    struct CBuff {
        std::vector<CBuffDraw> draws;
        bool valid = false;
    };
    std::vector<CBuff> cbuffs_;
    mutable std::mutex cbuffs_mutex_;
    mutable std::mutex textures_mutex_;
};
