#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <mutex>
#include <vector>

#include "platform/renderer/IRenderPath.h"
#include "Device.h"
#include "DeletionQueue.h"
#include "DisplayListManager.h"
#include "FrameContext.h"
#include "PipelineCache.h"
#include "Swapchain.h"
#include "TextureManager.h"

struct SDL_Window;

namespace plce::vk {

/// Vulkan renderer (vk) — uses dynamic depth bias instead of MVP Z hack,
/// per-texture samplers, and fan-to-list CPU conversion.
class Renderer final : public rp::IRenderPath {
public:
    explicit Renderer(SDL_Window* window);
    ~Renderer() override;

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // -- Lifecycle --
    void StartFrame() override;
    void Present() override;
    void Clear(int flags) override;
    void SetClearColour(const float rgba[4]) override;
    void render_frame(const rp::FrameDesc&) override {}
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
    void MatrixPerspective(float fovy, float aspect, float zNear, float zFar) override;
    void MatrixOrthogonal(float l, float r, float b, float t, float zn, float zf) override;
    void MatrixPop() override;
    void MatrixPush() override;
    void MatrixMult(float* mat) override;
    [[nodiscard]] const float* MatrixGet(rp::MatrixStack stack) override;

    // -- Draw --
    void DrawVertices(int primType, int count, void* data, int vType, int sType) override;

    // -- Resources --
    [[nodiscard]] rp::MeshHandle create_mesh(const rp::MeshDesc&) override { return rp::kInvalidMesh; }
    void update_mesh(rp::MeshHandle, const rp::MeshDesc&) override {}
    void destroy_mesh(rp::MeshHandle) override {}

    [[nodiscard]] rp::TextureHandle create_texture(const rp::TextureDesc&) override { return rp::kInvalidTexture; }
    void update_texture(rp::TextureHandle, const rp::TextureRegion&) override {}
    void destroy_texture(rp::TextureHandle) override {}

    [[nodiscard]] rp::MaterialHandle create_material(const rp::MaterialDesc&) override;
    void update_material(rp::MaterialHandle, const rp::MaterialDesc&) override {}
    void destroy_material(rp::MaterialHandle) override {}

    [[nodiscard]] std::pair<rp::TransientVertexBuffer, std::span<std::byte>>
    alloc_transient_vertices(uint32_t count, rp::VertexLayout layout,
                             rp::PrimitiveType prim) override;

    void read_framebuffer(const rp::TextureReadback&) override {}
    [[nodiscard]] rp::ResourceFootprint query_resource_footprint() const override { return {}; }
    void seal_static_resource_tier() override {}
    void begin_atomic_resource_batch() override {}
    void end_atomic_resource_batch() override {}
    void push_debug_event(const char*) override;
    void pop_debug_event() override;
    void tick() override {}
    void submit_immediate(const rp::DrawCall&) override;

    // -- CBuff (display list) --
    [[nodiscard]] int CBuffCreate(int n) override;
    void CBuffDelete(int, int) override {}
    void CBuffDeleteAll() override;
    void CBuffStart(int index, bool full = true) override;
    void CBuffClear(int index) override;
    [[nodiscard]] int CBuffSize(int index) override;
    void CBuffEnd() override;
    [[nodiscard]] bool CBuffCall(int index, bool full = true) override;
    void CBuffDeferredModeStart() override {}
    void CBuffDeferredModeEnd() override {}

    // -- Textures --
    [[nodiscard]] int TextureCreate() override;
    void TextureFree(int idx) override;
    void TextureBind(int idx) override;
    void TextureBindVertex(int idx, bool) override { tex_mgr_.bind_vertex(idx); }
    void TextureSetTextureLevels(int) override {}
    void TextureData(int w, int h, void* data, int level, int format) override;
    void TextureDataUpdate(int xo, int yo, int w, int h, void* data, int lvl) override;
    void TextureSetParam(int param, int value) override;
    [[nodiscard]] int TextureGetTextureLevels() override { return 1; }

    // -- State --
    void StateSetColour(float r, float g, float b, float a) override;
    void StateSetDepthMask(bool e) override;
    void StateSetBlendEnable(bool e) override;
    void StateSetBlendFunc(rp::BlendFactor s, rp::BlendFactor d) override;
    void StateSetBlendFactor(unsigned int argb) override;
    void StateSetAlphaFunc(rp::AlphaTest f, float ref) override;
    void StateSetDepthFunc(rp::DepthTest f) override;
    void StateSetFaceCull(bool e) override;
    void StateSetLineWidth(float) override {}
    void StateSetWriteEnable(bool r, bool g, bool b, bool a) override;
    void StateSetDepthTestEnable(bool e) override;
    void StateSetAlphaTestEnable(bool e) override;
    void StateSetDepthSlopeAndBias(float slope, float bias) override;
    void StateSetFogEnable(bool e) override { fog_enabled_ = e; }
    void StateSetFogMode(rp::FogMode m) override { fog_mode_ = m; }
    void StateSetFogNearDistance(float d) override { fog_start_ = d; }
    void StateSetFogFarDistance(float d) override  { fog_end_ = d; }
    void StateSetFogDensity(float d) override      { fog_density_ = d; }
    void StateSetFogColour(float r, float g, float b) override { fog_colour_ = {r,g,b,1}; }
    void StateSetLightingEnable(bool e) override { lighting_enabled_ = e; }
    void StateSetLightColour(int, float r, float g, float b) override { light_diffuse_ = {r,g,b}; }
    void StateSetLightAmbientColour(float r, float g, float b) override { light_ambient_ = {r,g,b}; }
    void StateSetLightDirection(int idx, float x, float y, float z) override;
    void StateSetLightEnable(int, bool) override {}
    void StateSetViewport(int) override {}
    void StateSetEnableViewportClipPlanes(bool) override {}
    void StateSetStencil(int, uint8_t, uint8_t, uint8_t) override {}
    void StateSetForceLOD(int) override {}
    void StateSetTextureEnable(bool e) override;
    void StateSetActiveTexture(int gl_enum) override;
    void StateSetVertexTextureUV(float u, float v) override { global_lm_uv_ = {u,v}; }

    void SetChunkOffset(float x, float y, float z) override { chunk_offset_ = {x,y,z}; }
    void ReadPixels(int, int, int, int, void*) override {}
    [[nodiscard]] int LoadTextureData(const char* fn, void* info, int** out) override;
    [[nodiscard]] int LoadTextureData(uint8_t* data, uint32_t bytes, void* info, int** out) override;

    void Set_matrixDirty() override {}
    void CBuffLockStaticCreations() override {}
    void UpdateGamma(unsigned short g) override;
    void Suspend() override {}
    [[nodiscard]] bool Suspended() override { return false; }
    void Resume() override {}
    void BeginEvent(const char*) override {}
    void EndEvent() override {}

private:
    static constexpr uint32_t kFramesInFlight = 2;

    // -- Subsystems --
    Device     dev_;
    Swapchain  swap_;
    std::array<FrameContext, kFramesInFlight> frames_;
    PipelineCache pipelines_;
    TextureManager tex_mgr_;
    DisplayListManager dl_mgr_;

    // -- Pipeline layout + descriptors --
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout tex_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool tex_pool_ = VK_NULL_HANDLE;

    // Quad index buffer (GL_QUADS -> 2 triangles)
    VkBuffer      quad_ib_       = VK_NULL_HANDLE;
    VmaAllocation quad_ib_alloc_ = nullptr;
    static constexpr uint32_t kMaxQuads = 16384;

    // -- Per-frame state --
    uint32_t frame_idx_     = 0;
    uint32_t acquired_img_  = 0;
    bool     frame_active_  = false;
    bool     pass_active_   = false;
    bool     should_close_  = false;
    bool     transient_overflow_warned_ = false;

    // -- Render state tracking --
    PipelineKey   pso_key_{};
    PipelineKey   last_bound_pso_{};
    bool     pso_dirty_ = true;

    std::array<float, 4> clear_color_{0.05f, 0.05f, 0.10f, 1.0f};
    std::array<float, 4> state_colour_{1, 1, 1, 1};
    std::array<float, 4> blend_constants_{1, 1, 1, 1};
    std::array<float, 3> chunk_offset_{};

    bool lighting_enabled_  = false;
    bool alpha_test_enabled_ = false;
    rp::AlphaTest alpha_test_func_ = rp::AlphaTest::greater;
    float alpha_ref_ = 0.1f;
    float inv_gamma_ = 1.0f;

    bool fog_enabled_ = false;
    rp::FogMode fog_mode_ = rp::FogMode::linear;
    float fog_start_ = 0, fog_end_ = 1, fog_density_ = 1;
    std::array<float, 4> fog_colour_{0.5f, 0.7f, 1.0f, 1.0f};

    glm::vec3 light0_dir_eye_{0.174f, 0.870f, -0.609f};
    glm::vec3 light1_dir_eye_{-0.174f, 0.870f, 0.609f};
    glm::vec3 light_diffuse_{0.6f, 0.6f, 0.6f};
    glm::vec3 light_ambient_{0.4f, 0.4f, 0.4f};

    float depth_bias_constant_ = 0;
    float depth_bias_slope_    = 0;

    int  active_tex_unit_  = 0;
    bool texture_enabled_  = true;
    std::array<float, 2> global_lm_uv_{240, 240};

    // Matrix stacks
    rp::MatrixStack matrix_mode_ = rp::MatrixStack::modelview;
    std::vector<glm::mat4> mv_stack_{glm::mat4(1)};
    std::vector<glm::mat4> proj_stack_{glm::mat4(1)};
    std::vector<glm::mat4> tex_stack_{glm::mat4(1)};

    uint32_t next_material_id_ = 0;

    // Thread-safe deferred buffer destruction. Worker threads push here
    // instead of accessing frame().deletions (which is main-thread only).
    std::vector<PendingDestroy> pending_destroys_;
    std::mutex pending_destroy_mutex_;

    // Framebuffer info
    rp::FrameFramebuffer fb_{};
    SDL_Window* window_ = nullptr;

    // -- Internal helpers --
    FrameContext& frame() { return frames_[frame_idx_]; }
    std::vector<glm::mat4>& stack();
    void ensure_pass();
    void begin_pass();
    void end_pass();
    void fill_push_constants(void* out, bool textured, const glm::vec4* tint = nullptr);
};

}  // namespace plce::vk
