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
    rp::MaterialHandle untextured_alpha_no_depth{};  // for fullscreen overlays
    rp::MaterialHandle line_alpha{};                 // for untextured line-list draws
    rp::MaterialHandle textured_alpha{};
    rp::MaterialHandle font_glyph{};
    rp::MaterialHandle fullscreen_overlay{};
    rp::MaterialHandle vignette{};
    rp::MaterialHandle item_in_hand{};         // listItem / listTerrain 3D mesh
    rp::MaterialHandle item_in_hand_glint{};   // listGlint enchant overlay
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

// Capture the live texture-matrix stack into the scale + translation
// form the vertex shader reads (v_uv = a_uv * scale + offset). The
// basic shader only uses the diagonal + xy translation of the texture
// matrix — rotation/shear on the matrix were silently ignored on the
// legacy path too, so mirroring just scale+translate here is not a
// behaviour change.
void snapshot_uv_transform(float scale[2], float offset[2]) {
    const float* tm = RenderPath.MatrixGet(rp::MatrixStack::texture);
    // Column-major 4x4. scale = (tm[0][0], tm[1][1]);
    // offset = (tm[3][0], tm[3][1]).
    scale[0]  = tm[0];
    scale[1]  = tm[5];
    offset[0] = tm[12];
    offset[1] = tm[13];
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

    // Same as untextured_alpha but with depth test + write off. Used
    // for fullscreen colour overlays (sleep, death, damage flash) that
    // need to land on top of the existing HUD regardless of each HUD
    // sprite's blitOffset z value.
    rp::MaterialDesc untex_nd{};
    untex_nd.shader      = rp::ShaderPath::standard;
    untex_nd.blend       = rp::BlendMode::alpha;
    untex_nd.textured    = false;
    untex_nd.lit         = false;
    untex_nd.fog_enabled = false;
    untex_nd.depth_test  = rp::DepthTest::off;
    untex_nd.depth_write = false;
    untex_nd.cull        = rp::CullMode::none;
    s_materials.untextured_alpha_no_depth = RenderPath.create_material(untex_nd);

    // Line primitive material — untextured, alpha blend, depth off.
    // Separate from untextured_alpha_no_depth because record_draw_call
    // already branches on DrawCall.transient.primitive for topology;
    // the material just needs to keep every other state right.
    rp::MaterialDesc line{};
    line.shader      = rp::ShaderPath::standard;
    line.blend       = rp::BlendMode::alpha;
    line.textured    = false;
    line.lit         = false;
    line.fog_enabled = false;
    line.depth_test  = rp::DepthTest::off;
    line.depth_write = false;
    line.cull        = rp::CullMode::none;
    s_materials.line_alpha = RenderPath.create_material(line);

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

    // 3D held-item mesh material. Lit via the frame's directional
    // lights, alpha tested to kill the item atlas's transparent pixels.
    // Depth test + write on so it composites correctly with world
    // geometry in first-person view.
    rp::MaterialDesc item{};
    item.shader      = rp::ShaderPath::standard;
    item.blend       = rp::BlendMode::alpha;
    item.textured    = true;
    item.lit         = true;
    item.fog_enabled = false;
    item.alpha_test  = rp::AlphaTest::greater;
    item.alpha_ref   = 0.1f;
    item.depth_test  = rp::DepthTest::less_equal;
    item.depth_write = true;
    item.cull        = rp::CullMode::back_ccw;
    s_materials.item_in_hand = RenderPath.create_material(item);

    // Glint overlay material. depth=equal so it only paints on top of
    // the item it was drawn after; src_color * one gives an additive
    // tint that matches the legacy RenderPath.StateSetBlendFunc(
    // src_color, one) path. Lighting off — the glint mesh bakes its
    // own per-vertex tint.
    rp::MaterialDesc glint{};
    glint.shader           = rp::ShaderPath::standard;
    glint.blend            = rp::BlendMode::custom;
    glint.blend_src_custom = rp::BlendFactor::src_color;
    glint.blend_dst_custom = rp::BlendFactor::one;
    glint.textured         = true;
    glint.lit              = false;
    glint.fog_enabled      = false;
    glint.alpha_test       = rp::AlphaTest::off;
    glint.depth_test       = rp::DepthTest::equal;
    glint.depth_write      = true;
    glint.cull             = rp::CullMode::back_ccw;
    s_materials.item_in_hand_glint = RenderPath.create_material(glint);

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

void draw_untextured_lines(const LineVertex* vertices, size_t vertex_count) {
    if (!vertices || vertex_count < 2) return;

    auto [tvb, span] = RenderPath.alloc_transient_vertices(
        uint32_t(vertex_count), rp::VertexLayout::world_standard,
        rp::PrimitiveType::line_list);
    if (span.empty()) return;

    auto* v = reinterpret_cast<rp::WorldStandardVertex*>(span.data());
    for (size_t i = 0; i < vertex_count; ++i) {
        v[i] = {{vertices[i].x, vertices[i].y, 0.0f},
                {0.0f, 0.0f},
                vertices[i].rgba,
                0,
                0};
    }

    rp::DrawCall dc{};
    dc.source    = rp::VertexSource::transient;
    dc.transient = tvb;
    dc.material  = s_materials.line_alpha;
    // Per-vertex colour drives the line shading. The shader treats
    // rgb=0 as a sentinel meaning "use pc.state_colour instead"; set
    // tint to opaque black so 0xFF000000 vertices (common in the F3
    // graph at the leftmost data point) still render black rather
    // than snap to white.
    dc.tint_color[0] = 0.0f;
    dc.tint_color[1] = 0.0f;
    dc.tint_color[2] = 0.0f;
    dc.tint_color[3] = 1.0f;
    snapshot_transform(dc.transform);

    rp::ui_overlay::push(dc);
}

void draw_fullscreen_fill(int w, int h, uint32_t rgba) {
    auto [tvb, span] = RenderPath.alloc_transient_vertices(
        4, rp::VertexLayout::world_standard, rp::PrimitiveType::triangle_fan);
    if (span.empty()) return;

    auto* v = reinterpret_cast<rp::WorldStandardVertex*>(span.data());
    v[0] = {{0.0f,     float(h), 0.0f}, {0, 0}, 0, 0, kVertexColorSentinel};
    v[1] = {{float(w), float(h), 0.0f}, {0, 0}, 0, 0, kVertexColorSentinel};
    v[2] = {{float(w), 0.0f,     0.0f}, {0, 0}, 0, 0, kVertexColorSentinel};
    v[3] = {{0.0f,     0.0f,     0.0f}, {0, 0}, 0, 0, kVertexColorSentinel};

    rp::DrawCall dc{};
    dc.source    = rp::VertexSource::transient;
    dc.transient = tvb;
    dc.material  = s_materials.untextured_alpha_no_depth;
    unpack_rgba(rgba, dc.tint_color);

    // Synthesize the (w, h) ortho — callers pass logical UI size, same
    // convention as draw_fullscreen_overlay.
    const glm::mat4 proj = glm::ortho(0.0f, float(w), float(h), 0.0f,
                                      -100.0f, 100.0f);
    std::memcpy(dc.transform, &proj[0][0], sizeof(float) * 16);

    rp::ui_overlay::push(dc);
}

void draw_item_in_hand_mesh(rp::MeshHandle mesh, int texture_id,
                            const float tint_rgba[4], int forced_lod) {
    if (!mesh) return;

    rp::DrawCall dc{};
    dc.source                 = rp::VertexSource::mesh;
    dc.mesh                   = mesh;
    dc.material               = s_materials.item_in_hand;
    dc.texture_override.index = uint32_t(texture_id);
    dc.tint_color[0] = tint_rgba[0];
    dc.tint_color[1] = tint_rgba[1];
    dc.tint_color[2] = tint_rgba[2];
    dc.tint_color[3] = tint_rgba[3];
    if (forced_lod >= 0) dc.forced_lod = int8_t(forced_lod);
    snapshot_transform(dc.transform);
    snapshot_uv_transform(dc.uv_scale, dc.uv_offset);

    rp::ui_overlay::push(dc);
}

void draw_item_in_hand_glint_mesh(rp::MeshHandle mesh, int texture_id) {
    if (!mesh) return;

    rp::DrawCall dc{};
    dc.source                 = rp::VertexSource::mesh;
    dc.mesh                   = mesh;
    dc.material               = s_materials.item_in_hand_glint;
    dc.texture_override.index = uint32_t(texture_id);
    // Tint is baked into the mesh vertex colours; leave dc.tint_color
    // at 1,1,1,1 (the default) so fill_push_constants's
    // state_colour * tint passthrough reproduces the legacy behaviour.
    snapshot_transform(dc.transform);
    snapshot_uv_transform(dc.uv_scale, dc.uv_offset);

    rp::ui_overlay::push(dc);
}

void draw_glyph_quad(float x, float y, float w, float h,
                     float u0, float v0, float u1, float v1,
                     int texture_id, const float rgba[4]) {
    auto [tvb, span] = RenderPath.alloc_transient_vertices(
        4, rp::VertexLayout::world_standard, rp::PrimitiveType::triangle_fan);
    if (span.empty()) return;

    auto* v = reinterpret_cast<rp::WorldStandardVertex*>(span.data());
    v[0] = {{x,     y + h, 0.0f}, {u0, v1}, 0, 0, kVertexColorSentinel};
    v[1] = {{x + w, y + h, 0.0f}, {u1, v1}, 0, 0, kVertexColorSentinel};
    v[2] = {{x + w, y,     0.0f}, {u1, v0}, 0, 0, kVertexColorSentinel};
    v[3] = {{x,     y,     0.0f}, {u0, v0}, 0, 0, kVertexColorSentinel};

    rp::DrawCall dc{};
    dc.source                 = rp::VertexSource::transient;
    dc.transient              = tvb;
    dc.material               = s_materials.font_glyph;
    dc.texture_override.index = uint32_t(texture_id);
    dc.tint_color[0] = rgba[0];
    dc.tint_color[1] = rgba[1];
    dc.tint_color[2] = rgba[2];
    dc.tint_color[3] = rgba[3];
    snapshot_transform(dc.transform);

    rp::ui_overlay::push(dc);
}

}  // namespace plce::ui
