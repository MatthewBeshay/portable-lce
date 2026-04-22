#include "platform/renderer/ui/UiDraw.h"

#include <cstring>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "platform/renderer/IRenderPath.h"
#include "platform/renderer/renderer.h"

namespace plce::ui {

namespace {

// Module-private material handles. Populated by init(), consumed by
// every draw_* function. Zero-initialised handles are invalid and
// record_draw_call rejects them, so callers that forget to call init()
// just produce no visible output (plus will eventually tripped an
// assert when we add one).
struct MaterialTable {
    rp::MaterialHandle untextured_alpha{};
    rp::MaterialHandle textured_alpha{};
    rp::MaterialHandle font_glyph{};
    rp::MaterialHandle fullscreen_overlay{};
    rp::MaterialHandle vignette{};
};

MaterialTable s_materials;
bool          s_initialised = false;

// Capture the live proj * mv from the matrix stacks into a 16-float
// array suitable for DrawCall::transform. render_frame uses identity
// matrices when processing ui_overlay, so each DrawCall's transform
// carries the full screen-to-clip mapping. Any MatrixPush/Scale/
// Translate scope the caller wrapped around this draw is preserved.
void snapshot_transform(float out[16]) {
    const float* proj = RenderPath.MatrixGet(rp::MatrixStack::projection);
    const float* mv   = RenderPath.MatrixGet(rp::MatrixStack::modelview);
    glm::mat4 p(1.0f), m(1.0f);
    std::memcpy(&p[0][0], proj, sizeof(float) * 16);
    std::memcpy(&m[0][0], mv,   sizeof(float) * 16);
    const glm::mat4 pm = p * m;
    std::memcpy(out, &pm[0][0], sizeof(float) * 16);
}

// Unpack 0xAARRGGBB-in-uint32 (R in low byte, A in MSB) into
// normalised float RGBA.
void unpack_rgba(uint32_t rgba, float out[4]) {
    out[0] = float((rgba      ) & 0xFFu) / 255.0f;
    out[1] = float((rgba >>  8) & 0xFFu) / 255.0f;
    out[2] = float((rgba >> 16) & 0xFFu) / 255.0f;
    out[3] = float((rgba >> 24) & 0xFFu) / 255.0f;
}

// Reinterpret the packed-colour field in WorldStandardVertex as the
// caller's rgba. The 0xfe00fe00 sentinel the shader understands as
// "use the PC tint, ignore vertex colour" — used when the draw wants
// a uniform tint via DrawCall::tint_color.
constexpr uint32_t kVertexColorSentinel = 0xfe00fe00u;

}  // namespace

void init() {
    if (s_initialised) return;

    rp::MaterialDesc untex{};
    untex.shader      = rp::ShaderPath::standard;
    untex.blend       = rp::BlendMode::alpha;
    untex.textured    = false;
    untex.lit         = false;
    untex.fog_enabled = false;
    untex.depth_test  = rp::DepthTest::less_equal;
    untex.depth_write = true;
    untex.cull        = rp::CullMode::none;
    s_materials.untextured_alpha = RenderPath.create_material(untex);

    rp::MaterialDesc textured{};
    textured.shader      = rp::ShaderPath::standard;
    textured.blend       = rp::BlendMode::alpha;
    textured.textured    = true;
    textured.lit         = false;
    textured.fog_enabled = false;
    textured.depth_test  = rp::DepthTest::less_equal;
    textured.depth_write = true;
    textured.cull        = rp::CullMode::none;
    s_materials.textured_alpha = RenderPath.create_material(textured);

    rp::MaterialDesc font{};
    font.shader      = rp::ShaderPath::standard;
    font.blend       = rp::BlendMode::alpha;
    font.textured    = true;
    font.lit         = false;
    font.fog_enabled = false;
    // Glyph atlas is alpha-tested to kill the bg sprite pixels;
    // depth off so text lays on top of the current frame.
    font.alpha_test  = rp::AlphaTest::greater;
    font.alpha_ref   = 0.1f;
    font.depth_test  = rp::DepthTest::off;
    font.depth_write = false;
    font.cull        = rp::CullMode::none;
    s_materials.font_glyph = RenderPath.create_material(font);

    rp::MaterialDesc overlay{};
    overlay.shader      = rp::ShaderPath::standard;
    overlay.blend       = rp::BlendMode::alpha;
    overlay.textured    = true;
    overlay.lit         = false;
    overlay.fog_enabled = false;
    overlay.depth_test  = rp::DepthTest::off;
    overlay.depth_write = false;
    overlay.alpha_test  = rp::AlphaTest::off;
    overlay.cull        = rp::CullMode::none;
    s_materials.fullscreen_overlay = RenderPath.create_material(overlay);

    rp::MaterialDesc vig{};
    vig.shader           = rp::ShaderPath::standard;
    vig.blend            = rp::BlendMode::custom;
    vig.blend_src_custom = rp::BlendFactor::zero;
    vig.blend_dst_custom = rp::BlendFactor::one_minus_src_color;
    vig.textured         = true;
    vig.lit              = false;
    vig.fog_enabled      = false;
    vig.depth_test       = rp::DepthTest::off;
    vig.depth_write      = false;
    vig.cull             = rp::CullMode::none;
    s_materials.vignette = RenderPath.create_material(vig);

    s_initialised = true;
}

void draw_fill(int x0, int y0, int x1, int y1, uint32_t rgba, float z) {
    auto [tvb, span] = RenderPath.alloc_transient_vertices(
        4, rp::VertexLayout::world_standard, rp::PrimitiveType::triangle_fan);
    if (span.empty()) return;

    auto* v = reinterpret_cast<rp::WorldStandardVertex*>(span.data());
    // Match the legacy GuiComponent::fill vertex order (triangle fan
    // wound so the quad faces forward under the GUI ortho).
    v[0] = {{(float)x0, (float)y1, z}, {0, 0}, 0, 0, kVertexColorSentinel};
    v[1] = {{(float)x1, (float)y1, z}, {0, 0}, 0, 0, kVertexColorSentinel};
    v[2] = {{(float)x1, (float)y0, z}, {0, 0}, 0, 0, kVertexColorSentinel};
    v[3] = {{(float)x0, (float)y0, z}, {0, 0}, 0, 0, kVertexColorSentinel};

    rp::DrawCall dc{};
    dc.source    = rp::VertexSource::transient;
    dc.transient = tvb;
    dc.material  = s_materials.untextured_alpha;
    unpack_rgba(rgba, dc.tint_color);
    snapshot_transform(dc.transform);

    rp::ui_overlay::push(dc);
}

void draw_fill_gradient(int x0, int y0, int x1, int y1,
                        uint32_t top_rgba, uint32_t bottom_rgba, float z) {
    auto [tvb, span] = RenderPath.alloc_transient_vertices(
        4, rp::VertexLayout::world_standard, rp::PrimitiveType::triangle_fan);
    if (span.empty()) return;

    auto* v = reinterpret_cast<rp::WorldStandardVertex*>(span.data());
    // Per-vertex colour carries the gradient; tint stays 1,1,1,1 so
    // the shader passes per-vertex through unchanged. Vertex order
    // matches the legacy GuiComponent::fillGradient.
    v[0] = {{(float)x1, (float)y0, z}, {0, 0}, top_rgba,    0, kVertexColorSentinel};
    v[1] = {{(float)x0, (float)y0, z}, {0, 0}, top_rgba,    0, kVertexColorSentinel};
    v[2] = {{(float)x0, (float)y1, z}, {0, 0}, bottom_rgba, 0, kVertexColorSentinel};
    v[3] = {{(float)x1, (float)y1, z}, {0, 0}, bottom_rgba, 0, kVertexColorSentinel};

    rp::DrawCall dc{};
    dc.source    = rp::VertexSource::transient;
    dc.transient = tvb;
    dc.material  = s_materials.untextured_alpha;
    // tint passthrough — per-vertex colour does the shading.
    dc.tint_color[0] = dc.tint_color[1] = dc.tint_color[2] = dc.tint_color[3] = 1.0f;
    snapshot_transform(dc.transform);

    rp::ui_overlay::push(dc);
}

void draw_textured_quad(int x, int y, int w, int h,
                        float u0, float v0, float u1, float v1,
                        int texture_id, uint32_t tint_rgba) {
    draw_textured_quad(float(x), float(y), float(x + w), float(y + h), 0.0f,
                       u0, v0, u1, v1, texture_id, tint_rgba);
}

void draw_textured_quad(float x0, float y0, float x1, float y1, float z,
                        float u0, float v0, float u1, float v1,
                        int texture_id, uint32_t tint_rgba) {
    auto [tvb, span] = RenderPath.alloc_transient_vertices(
        4, rp::VertexLayout::world_standard, rp::PrimitiveType::triangle_fan);
    if (span.empty()) return;

    auto* v = reinterpret_cast<rp::WorldStandardVertex*>(span.data());
    v[0] = {{x0, y1, z}, {u0, v1}, 0, 0, kVertexColorSentinel};
    v[1] = {{x1, y1, z}, {u1, v1}, 0, 0, kVertexColorSentinel};
    v[2] = {{x1, y0, z}, {u1, v0}, 0, 0, kVertexColorSentinel};
    v[3] = {{x0, y0, z}, {u0, v0}, 0, 0, kVertexColorSentinel};

    rp::DrawCall dc{};
    dc.source                 = rp::VertexSource::transient;
    dc.transient              = tvb;
    dc.material               = s_materials.textured_alpha;
    dc.texture_override.index = uint32_t(texture_id);
    unpack_rgba(tint_rgba, dc.tint_color);
    snapshot_transform(dc.transform);

    rp::ui_overlay::push(dc);
}

namespace {

// Build a screen-space ortho covering [0, w] × [0, h] with a [-100,
// 100] depth range. Used by the fullscreen-overlay helpers, which
// paint the entire window and don't want to depend on whatever
// matrix state the legacy path last left live — the caller's
// (w, h) is the authoritative source of size.
glm::mat4 fullscreen_ortho(int w, int h) {
    return glm::ortho(0.0f, float(w), float(h), 0.0f, -100.0f, 100.0f);
}

// Emit a 4-vertex triangle-fan covering (0, 0)..(w, h) with a given
// UV rect. The caller-provided (w, h) ortho is baked into the
// DrawCall, not snapshot from the live matrix stack — legacy
// overlay helpers were always synthesising their own ortho here
// because the GUI ortho set up elsewhere doesn't necessarily match
// the fullscreen (w, h).
void push_fullscreen_quad(int w, int h, int texture_id,
                          rp::MaterialHandle material,
                          const float rgba[4],
                          float u0, float v0, float u1, float v1) {
    auto [tvb, span] = RenderPath.alloc_transient_vertices(
        4, rp::VertexLayout::world_standard, rp::PrimitiveType::triangle_fan);
    if (span.empty()) return;

    auto* v = reinterpret_cast<rp::WorldStandardVertex*>(span.data());
    v[0] = {{0.0f,     float(h), -90.0f}, {u0, v1}, 0, 0, kVertexColorSentinel};
    v[1] = {{float(w), float(h), -90.0f}, {u1, v1}, 0, 0, kVertexColorSentinel};
    v[2] = {{float(w), 0.0f,     -90.0f}, {u1, v0}, 0, 0, kVertexColorSentinel};
    v[3] = {{0.0f,     0.0f,     -90.0f}, {u0, v0}, 0, 0, kVertexColorSentinel};

    rp::DrawCall dc{};
    dc.source                 = rp::VertexSource::transient;
    dc.transient              = tvb;
    dc.material               = material;
    dc.texture_override.index = uint32_t(texture_id);
    dc.tint_color[0] = rgba[0];
    dc.tint_color[1] = rgba[1];
    dc.tint_color[2] = rgba[2];
    dc.tint_color[3] = rgba[3];

    const glm::mat4 proj = fullscreen_ortho(w, h);
    std::memcpy(dc.transform, &proj[0][0], sizeof(float) * 16);

    rp::ui_overlay::push(dc);
}

}  // namespace

void draw_fullscreen_overlay(int w, int h, int texture_id,
                             const float rgba[4],
                             float u0, float v0, float u1, float v1) {
    push_fullscreen_quad(w, h, texture_id,
                         s_materials.fullscreen_overlay, rgba,
                         u0, v0, u1, v1);
}

void draw_vignette(int w, int h, int texture_id, const float rgba[4]) {
    push_fullscreen_quad(w, h, texture_id,
                         s_materials.vignette, rgba,
                         0.0f, 0.0f, 1.0f, 1.0f);
}

// draw_glyph_quad lands in the next commit alongside the Font migration.

}  // namespace plce::ui
