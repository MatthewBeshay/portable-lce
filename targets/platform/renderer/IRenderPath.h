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
/// `packed_pixels_argb32` contains width*height 32-bit ARGB values (A in
/// the high byte) packed into unsigned 32-bit ints. Matches
/// BufferedImage's internal storage and uploads directly via
/// IRenderPath::TextureData.
struct LoadedImage {
    int width  = 0;
    int height = 0;
    std::vector<uint32_t> packed_pixels_argb32;
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

enum class TextureFilter : uint8_t {
    nearest,
    linear,
};

enum class TextureWrap : uint8_t {
    repeat,
    clamp_to_edge,
    mirrored_repeat,
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

// Description of a persistent mesh upload. Used to register a long-
// lived vertex buffer that DrawCalls can reference via MeshHandle
// (DrawCall::source == VertexSource::mesh). The renderer copies
// `vertex_data` into a device-local GPU buffer and keeps it resident
// until destroy_mesh is called. For short-lived per-frame geometry
// use alloc_transient_vertices + VertexSource::transient instead.
struct MeshDesc {
    const void*   vertex_data   = nullptr;
    uint32_t      vertex_count  = 0;
    VertexLayout  layout        = VertexLayout::world_standard;
    PrimitiveType primitive     = PrimitiveType::triangle_list;
    const char*   debug_name    = nullptr;
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
    // Per-draw alpha test threshold override. Materials share a single
    // alpha_ref, but particle batches and a handful of entity renderers
    // need finer-grained thresholds (0.01 / 0.1 / 1/255) on the same
    // base material. Negative means "inherit from material".
    float alpha_ref_override = -1.0f;
    // Texture transform (scale + offset) applied to sampled UVs in the
    // shader: v_uv = a_uv * uv_scale + uv_offset. Default identity.
    // Captures what the legacy TextureMatrix stack would otherwise carry;
    // modern DrawCall-path callers fill it explicitly so the draw stays
    // self-describing (record_draw_call overrides the live tex_stack).
    float uv_scale[2]  = {1.0f, 1.0f};
    float uv_offset[2] = {0.0f, 0.0f};

    // Snapshot of the modelview matrix at push time, used to derive the
    // normal matrix for per-vertex lighting. `transform` already carries
    // the full live proj*mv the shader applies to vertex positions; the
    // rotation part of that is what lights need for `mat3(mv) * normal`.
    // Default identity — UI paths that pass through unlit materials
    // (fonts, fills, lines) ignore it. Column-major, same convention as
    // `transform`.
    float mv_transform[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

    // World-space chunk origin added in the vertex shader after the
    // chunk-relative position is read. Lets chunk meshes stay stored at
    // local (0..16, 0..128, 0..16) while each DrawCall places them at
    // their world origin, without touching the live SetChunkOffset
    // state. Defaults to zero for non-chunk draws.
    float chunk_offset[3] = {0, 0, 0};

    // When true, `transform` / `mv_transform` / `uv_scale` / `uv_offset`
    // / `tint_color` are authoritative snapshots captured at push time
    // and the renderer must use them instead of the live legacy state
    // (proj_stack_, mv_stack_, tex_stack_, state_colour_, etc.). Used
    // by the deferred drain paths (rp::ui_overlay / rp::world_draws)
    // that run at end-of-frame when live state is no longer valid.
    //
    // When false, the DrawCall is a passthrough recorded synchronously
    // under live legacy state (MeshBuilder mid-frame, matching the
    // submit_immediate ordering the legacy Tesselator had). The
    // snapshot fields stay at their identity defaults and the renderer
    // skips the overrides so lighting / projection / textures inherit
    // whatever the caller set up.
    bool self_describing = false;
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

// Per-view scissor override. When set, replaces the default (full
// viewport rect) scissor for draws inside this view. Leave
// width == 0 to keep the viewport default.
struct ViewScissor {
    int32_t  x      = 0;
    int32_t  y      = 0;
    uint32_t width  = 0;
    uint32_t height = 0;
};

struct ViewDesc {
    ViewportLayout viewport_layout = ViewportLayout::fullscreen;
    ViewCamera camera;
    ViewClear clear;
    ViewColorMask color_mask;
    bool scissor_from_viewport = false;
    ViewScissor scissor;

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

    /// Register a persistent mesh with the renderer. The vertex data is
    /// copied into a device-local GPU buffer and kept resident until
    /// destroy_mesh is called on the returned handle. Default impl
    /// returns an invalid handle — new backends must override to
    /// participate in the persistent-mesh DrawCall path.
    [[nodiscard]] virtual MeshHandle create_mesh(const MeshDesc&) {
        return {};
    }
    virtual void destroy_mesh(MeshHandle /*handle*/) {}

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

    // -- GPU timestamps (per-pass) ------------------------------------------
    //
    // Wrap a render pass with a begin/end pair. The renderer records
    // GPU timestamps at the top/bottom of pipeline and logs the delta
    // per-tag in the 1-Hz breakdown. Pairs may nest; `tag` is stored
    // by pointer so it must outlive the frame (string literals are
    // fine). Default impl is a no-op for backends that don't support
    // timestamps.
    virtual void push_timestamp(const char* /*tag*/) {}
    virtual void pop_timestamp() {}

    // -- Host lifecycle (main thread only) ----------------------------------

    virtual void tick() = 0;

    // =======================================================================
    // [[deprecated]] Legacy methods - shrinks as subsystems migrate.
    //
    // These exist so call sites can move from `the old renderer` to
    // `render_path->foo()` one subsystem at a time. Each method forwards
    // to the underlying backend in LegacyGLRenderPath. When a method has
    // zero callers, delete it from this section.
    //
    // A subset of these (Matrix*, StateSet*, CBuff*, TextureBind,
    // DrawVertices, submit_immediate, SetChunkOffset, SetClearColour)
    // now carry [[deprecated]] — it's a catch-net so every remaining
    // raw-renderer call site shows up in the build log as a Tier D
    // cleanup target. MatrixGet / framebuffer() / StartFrame / Present /
    // render_frame / create_mesh / alloc_transient_vertices /
    // submit_draw_call are NOT marked — they're the modern API.
    // =======================================================================

#define PLCE_DEPRECATED_LEGACY \
    [[deprecated("use plce::ui::draw_* for 2D, plce::world::MeshBuilder / " \
                 "submit_draw_call for 3D, create_mesh for persistent meshes")]]

    // Matrix stack
    PLCE_DEPRECATED_LEGACY virtual void MatrixMode(MatrixStack stack) = 0;
    PLCE_DEPRECATED_LEGACY virtual void MatrixSetIdentity() = 0;
    PLCE_DEPRECATED_LEGACY virtual void MatrixTranslate(float x, float y, float z) = 0;
    PLCE_DEPRECATED_LEGACY virtual void MatrixRotate(float angle, float x, float y, float z) = 0;
    PLCE_DEPRECATED_LEGACY virtual void MatrixScale(float x, float y, float z) = 0;
    PLCE_DEPRECATED_LEGACY virtual void MatrixPerspective(float fovy, float aspect, float zNear,
                                   float zFar) = 0;
    PLCE_DEPRECATED_LEGACY virtual void MatrixOrthogonal(float left, float right, float bottom,
                                  float top, float zNear, float zFar) = 0;
    PLCE_DEPRECATED_LEGACY virtual void MatrixPop() = 0;
    PLCE_DEPRECATED_LEGACY virtual void MatrixPush() = 0;
    PLCE_DEPRECATED_LEGACY virtual void MatrixMult(float* mat) = 0;
    [[nodiscard]] virtual const float* MatrixGet(MatrixStack stack) = 0;

    // Draw. `vertexType` is 0 = WorldStandardVertex (32 B), 1 = compact
    // chunk format (16 B). The previous `shaderType` parameter was unused
    // by every backend — when a projected-texture / alt shader path is
    // genuinely needed, add a typed `ShaderPath` argument rather than
    // reviving the silent int.
    PLCE_DEPRECATED_LEGACY virtual void DrawVertices(int primitiveType, int count, void* data,
                              int vertexType) = 0;

    // Command buffers
    [[nodiscard]] PLCE_DEPRECATED_LEGACY virtual int CBuffCreate(int count) = 0;
    PLCE_DEPRECATED_LEGACY virtual void CBuffDelete(int first, int count) = 0;
    PLCE_DEPRECATED_LEGACY virtual void CBuffDeleteAll() = 0;
    PLCE_DEPRECATED_LEGACY virtual void CBuffStart(int index, bool full = false) = 0;
    PLCE_DEPRECATED_LEGACY virtual void CBuffClear(int index) = 0;
    [[nodiscard]] PLCE_DEPRECATED_LEGACY virtual int CBuffSize(int index) = 0;
    PLCE_DEPRECATED_LEGACY virtual void CBuffEnd() = 0;
    [[nodiscard]] PLCE_DEPRECATED_LEGACY virtual bool CBuffCall(int index, bool full = true) = 0;
    PLCE_DEPRECATED_LEGACY virtual void CBuffDeferredModeStart() = 0;
    PLCE_DEPRECATED_LEGACY virtual void CBuffDeferredModeEnd() = 0;

    // Textures
    [[nodiscard]] virtual int TextureCreate() = 0;
    virtual void TextureFree(int idx) = 0;
    PLCE_DEPRECATED_LEGACY virtual void TextureBind(int idx) = 0;
    virtual void TextureBindVertex(int idx, bool scaleLight = false) = 0;
    /// Currently-bound diffuse texture id at the backend level — what
    /// the next draw would sample if it didn't carry a texture_override.
    /// Needed by Tier-C modern draws (MeshBuilder / plce::ui) to snapshot
    /// the texture at push time so the drained DrawCall doesn't inherit
    /// whichever texture the legacy pipeline happens to have bound when
    /// render_frame runs.
    [[nodiscard]] virtual int TextureGetBoundId() const { return 0; }
    virtual void TextureSetTextureLevels(int levels) = 0;
    virtual void TextureData(int width, int height, void* data, int level,
                             int format = 0) = 0;
    virtual void TextureDataUpdate(int xoff, int yoff, int w, int h, void* data,
                                   int level) = 0;

    // Render state
    PLCE_DEPRECATED_LEGACY virtual void StateSetColour(float r, float g, float b, float a) = 0;
    PLCE_DEPRECATED_LEGACY virtual void StateSetDepthMask(bool enable) = 0;
    PLCE_DEPRECATED_LEGACY virtual void StateSetBlendEnable(bool enable) = 0;
    PLCE_DEPRECATED_LEGACY virtual void StateSetBlendFunc(BlendFactor src, BlendFactor dst) = 0;
    PLCE_DEPRECATED_LEGACY virtual void StateSetBlendFactor(unsigned int colour) = 0;
    PLCE_DEPRECATED_LEGACY virtual void StateSetAlphaFunc(AlphaTest func, float param) = 0;
    PLCE_DEPRECATED_LEGACY virtual void StateSetDepthFunc(DepthTest func) = 0;
    PLCE_DEPRECATED_LEGACY virtual void StateSetFaceCull(bool enable) = 0;
    PLCE_DEPRECATED_LEGACY virtual void StateSetLineWidth(float width) = 0;
    PLCE_DEPRECATED_LEGACY virtual void StateSetWriteEnable(bool r, bool g, bool b, bool a) = 0;
    PLCE_DEPRECATED_LEGACY virtual void StateSetDepthTestEnable(bool enable) = 0;
    PLCE_DEPRECATED_LEGACY virtual void StateSetAlphaTestEnable(bool enable) = 0;
    PLCE_DEPRECATED_LEGACY virtual void StateSetDepthSlopeAndBias(float slope, float bias) = 0;

    // Fog
    PLCE_DEPRECATED_LEGACY virtual void StateSetFogEnable(bool enable) = 0;
    PLCE_DEPRECATED_LEGACY virtual void StateSetFogMode(FogMode mode) = 0;
    PLCE_DEPRECATED_LEGACY virtual void StateSetFogNearDistance(float dist) = 0;
    PLCE_DEPRECATED_LEGACY virtual void StateSetFogFarDistance(float dist) = 0;
    PLCE_DEPRECATED_LEGACY virtual void StateSetFogDensity(float density) = 0;
    PLCE_DEPRECATED_LEGACY virtual void StateSetFogColour(float r, float g, float b) = 0;

    // Lighting
    PLCE_DEPRECATED_LEGACY virtual void StateSetLightingEnable(bool enable) = 0;
    PLCE_DEPRECATED_LEGACY virtual void StateSetLightColour(int light, float r, float g, float b) = 0;
    PLCE_DEPRECATED_LEGACY virtual void StateSetLightAmbientColour(float r, float g, float b) = 0;
    PLCE_DEPRECATED_LEGACY virtual void StateSetLightDirection(int light, float x, float y,
                                        float z) = 0;
    PLCE_DEPRECATED_LEGACY virtual void StateSetLightEnable(int light, bool enable) = 0;

    // Viewport
    PLCE_DEPRECATED_LEGACY virtual void StateSetViewport(int viewportType) = 0;
    PLCE_DEPRECATED_LEGACY virtual void StateSetEnableViewportClipPlanes(bool enable) = 0;
    PLCE_DEPRECATED_LEGACY virtual void StateSetStencil(int func, uint8_t ref, uint8_t funcMask,
                                 uint8_t writeMask) = 0;
    PLCE_DEPRECATED_LEGACY virtual void StateSetForceLOD(int lod) = 0;
    PLCE_DEPRECATED_LEGACY virtual void StateSetTextureEnable(bool enable) = 0;
    /// Enable/disable the lightmap sampler. Replaces the legacy GL pattern
    /// that toggled `GL_TEXTURE_2D` on texture unit 1.
    PLCE_DEPRECATED_LEGACY virtual void StateSetLightmapEnable(bool enable) = 0;
    /// Set the min/mag sampler filter for the currently-bound texture.
    /// Replaces the legacy TextureSetParam(GL_TEXTURE_MIN_FILTER, ...) pair.
    PLCE_DEPRECATED_LEGACY virtual void StateSetTextureFilter(TextureFilter min, TextureFilter mag) = 0;
    /// Set the S/T wrap modes for the currently-bound texture. Replaces the
    /// legacy TextureSetParam(GL_TEXTURE_WRAP_S/T, ...) pair.
    PLCE_DEPRECATED_LEGACY virtual void StateSetTextureWrap(TextureWrap s, TextureWrap t) = 0;

    // Chunks
    PLCE_DEPRECATED_LEGACY virtual void SetChunkOffset(float x, float y, float z) = 0;

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
    PLCE_DEPRECATED_LEGACY virtual void StateSetVertexTextureUV(float u, float v) = 0;

    // Frame lifecycle
    virtual void StartFrame() = 0;
    virtual void Present() = 0;
    virtual void Clear(int flags) = 0;
    // Variant that sets the clear colour atomically for this one call
    // without mutating persistent clear-colour state. Preferred over
    // SetClearColour + Clear, which is deprecated.
    virtual void Clear(int flags, const float rgba[4]) = 0;
    PLCE_DEPRECATED_LEGACY virtual void SetClearColour(const float rgba[4]) = 0;
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
    PLCE_DEPRECATED_LEGACY virtual void submit_immediate(const DrawCall& dc) = 0;
    /// Draw a DrawCall synchronously using the material-driven record
    /// path (pc.mvp = live proj*mv * dc.transform; material controls
    /// pipeline state). Lets mid-frame producers like
    /// plce::world::MeshBuilder keep correct ordering with legacy
    /// draws (terrain writes depth, entities depth-test against it,
    /// then legacy HUD clears depth for GUI) without queueing to
    /// world_draws and draining after the HUD depth-clear.
    virtual void submit_draw_call(const DrawCall& /*dc*/) {}

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
// UI overlay collection — frame-scoped DrawCall buffer.
//
// Subsystems that have migrated off the legacy stateful API (matrix pushes,
// StateSet*, submit_immediate) push declarative DrawCalls here during their
// normal draw flow. The main loop clears the buffer once per frame, then
// assigns it into `FrameDesc::ui_overlay` right before calling
// `render_frame`. The renderer iterates those DrawCalls and emits real
// draws via its material-aware record path.
//
// Not thread-safe — UI draws happen on the main thread.
// ---------------------------------------------------------------------------
namespace ui_overlay {

void push(const DrawCall& dc);
void clear();
[[nodiscard]] std::span<const DrawCall> get();

}  // namespace ui_overlay

// ---------------------------------------------------------------------------
// World draw collection — frame-scoped DrawCall / ChunkDrawCall buffers
// that feed ViewDesc's six world_/chunk_ spans. Mirrors ui_overlay but
// split by material kind (opaque / alpha_test / transparent) and by
// vertex source (world_ uses DrawCall + MeshHandle-or-transient;
// chunk_ uses ChunkDrawCall + chunk_offset).
//
// The main loop clears the buffers once per frame, then assigns spans
// over them into `FrameDesc::views[0]`. Subsystems that have migrated
// off the legacy stateful API push into the appropriate bucket during
// their normal draw flow.
//
// Not thread-safe for world_* draws — entity / tile-entity / particle
// migration all runs on the main thread. The chunk_* pushes will come
// from main thread too: worker-thread chunk meshing produces a
// MeshHandle asynchronously but the ChunkDrawCall submission remains
// on the main thread's render loop.
// ---------------------------------------------------------------------------
namespace world_draws {

void push_opaque(const DrawCall& dc);
void push_alpha_test(const DrawCall& dc);
void push_transparent(const DrawCall& dc);
void push_chunk_opaque(const ChunkDrawCall& dc);
void push_chunk_alpha_test(const ChunkDrawCall& dc);
void push_chunk_transparent(const ChunkDrawCall& dc);

void clear();

[[nodiscard]] std::span<const DrawCall> opaque();
[[nodiscard]] std::span<const DrawCall> alpha_test();
[[nodiscard]] std::span<const DrawCall> transparent();
[[nodiscard]] std::span<const ChunkDrawCall> chunk_opaque();
[[nodiscard]] std::span<const ChunkDrawCall> chunk_alpha_test();
[[nodiscard]] std::span<const ChunkDrawCall> chunk_transparent();

}  // namespace world_draws

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
