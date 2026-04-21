#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace rp {

/// Decoded image returned from IRenderPath::load_texture_data.
/// argb_pixels contains width*height 32-bit ARGB values (A in high byte);
/// this matches BufferedImage's internal storage and uploads directly via
/// IRenderPath::TextureData.
struct LoadedImage {
    int width  = 0;
    int height = 0;
    std::vector<int> argb_pixels;
};

// ---------------------------------------------------------------------------
// Handle types
// ---------------------------------------------------------------------------

struct MeshHandle {
    uint32_t index = 0;
    uint32_t generation = 0;

    bool operator==(const MeshHandle&) const = default;
    explicit operator bool() const { return generation != 0; }
};

struct TextureHandle {
    uint32_t index = 0;
    uint32_t generation = 0;

    bool operator==(const TextureHandle&) const = default;
    explicit operator bool() const { return generation != 0; }
};

struct MaterialHandle {
    uint32_t index = 0;
    uint32_t generation = 0;

    bool operator==(const MaterialHandle&) const = default;
    explicit operator bool() const { return generation != 0; }
};

inline constexpr MeshHandle kInvalidMesh{};
inline constexpr TextureHandle kInvalidTexture{};
inline constexpr MaterialHandle kInvalidMaterial{};

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

enum class VertexLayout : uint8_t {
    chunk_compact,
    world_standard,
    world_texgen,
};

enum class PrimitiveType : uint8_t {
    triangle_list,
    triangle_strip,
    triangle_fan,
    line_list,
    line_strip,
};

enum class ShaderPath : uint8_t {
    standard,
    projected_texture,
};

enum class BlendMode : uint8_t {
    opaque,
    alpha,
    additive,
    multiply,
    premultiplied,
    custom,
};

enum class BlendFactor : uint8_t {
    zero,
    one,
    src_color,
    one_minus_src_color,
    src_alpha,
    one_minus_src_alpha,
    dst_color,
    one_minus_dst_color,
    dst_alpha,
    one_minus_dst_alpha,
    constant_alpha,
    one_minus_constant_alpha,
};

enum class AlphaTest : uint8_t {
    off,
    greater,
    greater_equal,
    equal,
};

enum class DepthTest : uint8_t {
    off,
    less,
    less_equal,
    equal,
    greater,
    greater_equal,
    always,
};

enum class CullMode : uint8_t {
    none,
    back_ccw,
    back_cw,
    front,
};

enum class FogMode : uint8_t {
    disabled,
    linear,
    exponential,
    exponential_sq,
};

enum class MatrixStack : uint8_t {
    modelview,
    projection,
    texture,
};

enum class ViewportLayout : uint8_t {
    fullscreen,
    split_top,
    split_bottom,
    split_left,
    split_right,
    quadrant_top_left,
    quadrant_top_right,
    quadrant_bottom_left,
    quadrant_bottom_right,
};

enum class VertexSource : uint8_t {
    mesh,
    transient,
};

enum ClearFlags : uint8_t {
    CLEAR_NONE = 0,
    CLEAR_COLOR = 1 << 0,
    CLEAR_DEPTH = 1 << 1,
    CLEAR_STENCIL = 1 << 2,
};

// ---------------------------------------------------------------------------
// Descriptors
// ---------------------------------------------------------------------------

struct TransientVertexBuffer {
    uint32_t frame_index = 0;
    uint32_t offset = 0;
    uint32_t vertex_count = 0;
    VertexLayout layout = VertexLayout::world_standard;
    PrimitiveType primitive = PrimitiveType::triangle_list;
};

struct MaterialDesc {
    ShaderPath shader = ShaderPath::standard;
    BlendMode blend = BlendMode::opaque;
    BlendFactor blend_src_custom = BlendFactor::one;
    BlendFactor blend_dst_custom = BlendFactor::zero;
    AlphaTest alpha_test = AlphaTest::off;
    float alpha_ref = 0.0f;
    DepthTest depth_test = DepthTest::less_equal;
    bool depth_write = true;
    CullMode cull = CullMode::back_ccw;
    bool lit = true;
    bool textured = true;
    bool fog_enabled = true;
    TextureHandle texture_slots[4] = {};
    const char* debug_name = nullptr;
};

struct StencilOp {
    DepthTest func = DepthTest::always;
    uint8_t ref = 0;
    uint8_t func_mask = 0xFF;
    uint8_t write_mask = 0xFF;
};

struct DrawCall {
    VertexSource source = VertexSource::mesh;
    union {
        MeshHandle mesh;
        TransientVertexBuffer transient;
    };

    DrawCall() : mesh{} {}

    MaterialHandle material;
    float transform[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    float tint_color[4] = {1, 1, 1, 1};
    TextureHandle texture_override;
    uint8_t fog_profile_idx = 0;
    int8_t forced_lod = 0;
    float depth_slope = 0;
    float depth_bias = 0;
    float line_width = 0;
    uint32_t blend_constant_factor = 0xFFFFFFFF;
    const StencilOp* stencil = nullptr;
    bool lit_override_off = false;
};

struct ChunkDrawCall {
    MeshHandle mesh;
    MaterialHandle material;
    float chunk_offset[3] = {0, 0, 0};
    uint8_t fog_profile_idx = 0;
    int8_t forced_lod = 0;
};

struct FogProfile {
    FogMode mode = FogMode::disabled;
    float color[3] = {0, 0, 0};
    float start = 0;
    float end = 0;
    float density = 0;
};

struct DirectionalLight {
    bool enabled = false;
    float direction[3] = {0, -1, 0};
    float color[3] = {1, 1, 1};
};

struct LightingEnv {
    float ambient_color[3] = {0.2f, 0.2f, 0.2f};
    DirectionalLight directional[2] = {};
    TextureHandle lightmap;
};

struct ViewCamera {
    float projection[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    float view[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
};

struct ViewClear {
    uint8_t flags = CLEAR_NONE;
    float color[4] = {0, 0, 0, 1};
    float depth = 1.0f;
    uint8_t stencil = 0;
};

struct ViewColorMask {
    bool r = true, g = true, b = true, a = false;
};

struct ViewDesc {
    ViewportLayout viewport_layout = ViewportLayout::fullscreen;
    ViewCamera camera;
    ViewClear clear;
    ViewColorMask color_mask;
    bool scissor_from_viewport = false;

    FogProfile fog_profiles[4] = {};
    uint8_t fog_profile_count = 1;

    LightingEnv lighting;

    std::span<const ChunkDrawCall> chunk_opaque;
    std::span<const ChunkDrawCall> chunk_alpha_test;
    std::span<const ChunkDrawCall> chunk_transparent;
    std::span<const DrawCall> world_opaque;
    std::span<const DrawCall> world_alpha_test;
    std::span<const DrawCall> world_transparent;
    std::span<const DrawCall> debug_overlay;
};

struct FrameFramebuffer {
    uint32_t width = 0;
    uint32_t height = 0;
    float aspect = 0;
    bool is_widescreen = false;
    bool is_hi_def = false;
};

struct FrameDesc {
    FrameFramebuffer framebuffer;
    double current_time_seconds = 0;
    float delta_time_seconds = 0;
    uint64_t frame_index = 0;
    std::span<const ViewDesc> views;
    std::span<const DrawCall> ui_overlay;
};

// ---------------------------------------------------------------------------
// IRenderPath
// ---------------------------------------------------------------------------

class IRenderPath {
public:
    virtual ~IRenderPath() = default;

    // -- Persistent resources (thread-transparent) --------------------------

    [[nodiscard]] virtual MaterialHandle create_material(
        const MaterialDesc& desc) = 0;

    // -- Transient vertex buffer (thread-transparent, frame-scoped) ---------

    [[nodiscard]] virtual std::pair<TransientVertexBuffer, std::span<std::byte>>
    alloc_transient_vertices(uint32_t vertex_count, VertexLayout layout,
                             PrimitiveType primitive) = 0;

    // -- Frame submission (main thread only) --------------------------------

    virtual void render_frame(const FrameDesc& frame) = 0;
    virtual void resize(uint32_t w, uint32_t h) = 0;

    // -- Queries ------------------------------------------------------------

    [[nodiscard]] virtual const FrameFramebuffer& framebuffer() const = 0;

    // -- Debug markers ------------------------------------------------------

    virtual void push_debug_event(const char* name) = 0;
    virtual void pop_debug_event() = 0;

    // -- Host lifecycle (main thread only) ----------------------------------

    virtual void tick() = 0;

    // =======================================================================
    // [[deprecated]] Legacy methods - shrinks as subsystems migrate.
    //
    // These exist so call sites can move from `the old renderer` to
    // `render_path->foo()` one subsystem at a time. Each method forwards
    // to the underlying backend in LegacyGLRenderPath. Every caller
    // produces a compiler warning. When a method has zero callers, delete
    // it from this section.
    // =======================================================================

    // Matrix stack
    virtual void MatrixMode(MatrixStack stack) = 0;
    virtual void MatrixSetIdentity() = 0;
    virtual void MatrixTranslate(float x, float y, float z) = 0;
    virtual void MatrixRotate(float angle, float x, float y, float z) = 0;
    virtual void MatrixScale(float x, float y, float z) = 0;
    virtual void MatrixPerspective(float fovy, float aspect, float zNear,
                                   float zFar) = 0;
    virtual void MatrixOrthogonal(float left, float right, float bottom,
                                  float top, float zNear, float zFar) = 0;
    virtual void MatrixPop() = 0;
    virtual void MatrixPush() = 0;
    virtual void MatrixMult(float* mat) = 0;
    [[nodiscard]] virtual const float* MatrixGet(MatrixStack stack) = 0;

    // Draw. `vertexType` is 0 = WorldStandardVertex (32 B), 1 = compact
    // chunk format (16 B). The previous `shaderType` parameter was unused
    // by every backend — when a projected-texture / alt shader path is
    // genuinely needed, add a typed `ShaderPath` argument rather than
    // reviving the silent int.
    virtual void DrawVertices(int primitiveType, int count, void* data,
                              int vertexType) = 0;

    // Command buffers
    [[nodiscard]] virtual int CBuffCreate(int count) = 0;
    virtual void CBuffDelete(int first, int count) = 0;
    virtual void CBuffDeleteAll() = 0;
    virtual void CBuffStart(int index, bool full = false) = 0;
    virtual void CBuffClear(int index) = 0;
    [[nodiscard]] virtual int CBuffSize(int index) = 0;
    virtual void CBuffEnd() = 0;
    [[nodiscard]] virtual bool CBuffCall(int index, bool full = true) = 0;
    virtual void CBuffDeferredModeStart() = 0;
    virtual void CBuffDeferredModeEnd() = 0;

    // Textures
    [[nodiscard]] virtual int TextureCreate() = 0;
    virtual void TextureFree(int idx) = 0;
    virtual void TextureBind(int idx) = 0;
    virtual void TextureBindVertex(int idx, bool scaleLight = false) = 0;
    virtual void TextureSetTextureLevels(int levels) = 0;
    virtual void TextureData(int width, int height, void* data, int level,
                             int format = 0) = 0;
    virtual void TextureDataUpdate(int xoff, int yoff, int w, int h, void* data,
                                   int level) = 0;
    virtual void TextureSetParam(int param, int value) = 0;

    // Render state
    virtual void StateSetColour(float r, float g, float b, float a) = 0;
    virtual void StateSetDepthMask(bool enable) = 0;
    virtual void StateSetBlendEnable(bool enable) = 0;
    virtual void StateSetBlendFunc(BlendFactor src, BlendFactor dst) = 0;
    virtual void StateSetBlendFactor(unsigned int colour) = 0;
    virtual void StateSetAlphaFunc(AlphaTest func, float param) = 0;
    virtual void StateSetDepthFunc(DepthTest func) = 0;
    virtual void StateSetFaceCull(bool enable) = 0;
    virtual void StateSetLineWidth(float width) = 0;
    virtual void StateSetWriteEnable(bool r, bool g, bool b, bool a) = 0;
    virtual void StateSetDepthTestEnable(bool enable) = 0;
    virtual void StateSetAlphaTestEnable(bool enable) = 0;
    virtual void StateSetDepthSlopeAndBias(float slope, float bias) = 0;

    // Fog
    virtual void StateSetFogEnable(bool enable) = 0;
    virtual void StateSetFogMode(FogMode mode) = 0;
    virtual void StateSetFogNearDistance(float dist) = 0;
    virtual void StateSetFogFarDistance(float dist) = 0;
    virtual void StateSetFogDensity(float density) = 0;
    virtual void StateSetFogColour(float r, float g, float b) = 0;

    // Lighting
    virtual void StateSetLightingEnable(bool enable) = 0;
    virtual void StateSetLightColour(int light, float r, float g, float b) = 0;
    virtual void StateSetLightAmbientColour(float r, float g, float b) = 0;
    virtual void StateSetLightDirection(int light, float x, float y,
                                        float z) = 0;
    virtual void StateSetLightEnable(int light, bool enable) = 0;

    // Viewport
    virtual void StateSetViewport(int viewportType) = 0;
    virtual void StateSetEnableViewportClipPlanes(bool enable) = 0;
    virtual void StateSetStencil(int func, uint8_t ref, uint8_t funcMask,
                                 uint8_t writeMask) = 0;
    virtual void StateSetForceLOD(int lod) = 0;
    virtual void StateSetTextureEnable(bool enable) = 0;
    virtual void StateSetActiveTexture(int tex) = 0;

    // Chunks
    virtual void SetChunkOffset(float x, float y, float z) = 0;

    // Texture queries
    [[nodiscard]] virtual int TextureGetTextureLevels() = 0;
    virtual void ReadPixels(int x, int y, int w, int h, void* buf) = 0;
    /// Load texture data from a file path (PNG etc. via stb_image).
    /// Returns std::nullopt on failure; never throws.
    [[nodiscard]] virtual std::optional<LoadedImage>
    load_texture_data(const char* filename) = 0;
    /// Load texture data from an in-memory byte span.
    [[nodiscard]] virtual std::optional<LoadedImage>
    load_texture_data(std::span<const uint8_t> bytes) = 0;

    // Lighting state
    virtual void StateSetVertexTextureUV(float u, float v) = 0;

    // Frame lifecycle
    virtual void StartFrame() = 0;
    virtual void Present() = 0;
    virtual void Clear(int flags) = 0;
    virtual void SetClearColour(const float rgba[4]) = 0;
    virtual void Set_matrixDirty() = 0;
    virtual void CBuffLockStaticCreations() = 0;

    // Window queries (migrated to FrameDesc::framebuffer in new path)
    virtual void GetFramebufferSize(int& w, int& h) = 0;
    [[nodiscard]] virtual bool IsWidescreen() = 0;
    [[nodiscard]] virtual bool IsHiDef() = 0;
    virtual void Close() = 0;
    [[nodiscard]] virtual bool ShouldClose() = 0;
    virtual void SetWindowSize(int w, int h) = 0;
    virtual void SetFullscreen(bool fs) = 0;
    virtual void UpdateGamma(unsigned short gamma) = 0;
    virtual void Suspend() = 0;
    [[nodiscard]] virtual bool Suspended() = 0;
    virtual void Resume() = 0;

    // Events
    virtual void BeginEvent(const char* name) = 0;
    virtual void EndEvent() = 0;

    // Immediate single-draw submission
    virtual void submit_immediate(const DrawCall& dc) = 0;

    // GPU-driven terrain hooks (Phase 4). Default no-op so existing
    // backends (bgfx) ignore them; the Vulkan backend wires them through
    // to its TerrainRenderer when PLCE_VK_GPU_CHUNKS is enabled.
    struct ChunkUpload {
        int32_t  cx, cy, cz;
        uint8_t  layer;
        float    world_origin[3];
        float    aabb_min[3];
        float    aabb_max[3];
        const void* vertex_data = nullptr;
        uint32_t vertex_count   = 0;
        uint32_t vertex_stride  = 0;
    };
    virtual void chunk_upload(const ChunkUpload&) {}
    virtual void chunk_destroy(int32_t /*cx*/, int32_t /*cy*/,
                               int32_t /*cz*/, uint8_t /*layer*/) {}
    // Drains a recorded CBuff (built via the existing Tesselator + CBuff
    // path) into one contiguous chunk upload. Lets us reuse the legacy
    // tessellation pipeline without modifying Tesselator.
    virtual void chunk_upload_from_cbuff(int /*cbuff_id*/,
                                         const ChunkUpload& /*base*/) {}
    // Records the prepare phase (staging copies + cull dispatch) before
    // the next draw, then the indirect draw inside the active render pass.
    virtual void render_terrain(const float* /*mvp_4x4*/,
                                const float* /*frustum_24*/,
                                uint8_t /*layer*/ = 0) {}
    // Marks `texture_id` (from TextureCreate / glGenTextures_4J) as the
    // terrain atlas; binds it into the TerrainRenderer's descriptor set.
    virtual void set_terrain_atlas(int /*texture_id*/) {}
};

// ---------------------------------------------------------------------------
// Vertex format matching the Tesselator world_standard layout (32 bytes)
// ---------------------------------------------------------------------------

struct WorldStandardVertex {
    float pos[3];
    float uv[2];
    uint32_t color;
    uint32_t normal;
    uint32_t tex2;
};
static_assert(sizeof(WorldStandardVertex) == 32);

inline TextureHandle texture_handle_from_gl_id(int gl_id) {
    return {static_cast<uint32_t>(gl_id), 1};
}

// ---------------------------------------------------------------------------
// Global render path accessor
// ---------------------------------------------------------------------------

namespace render_path_internal {
void set_active(IRenderPath* path);
IRenderPath& get_active();
}  // namespace render_path_internal

}  // namespace rp

#define RenderPath (::rp::render_path_internal::get_active())

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

struct SDL_Window;
std::unique_ptr<rp::IRenderPath> make_bgfx_render_path(SDL_Window* window);
std::unique_ptr<rp::IRenderPath> make_vulkan_render_path(SDL_Window* window);
