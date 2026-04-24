#include "platform/renderer/world/WorldDraw.h"

#include <cstring>
#include <vector>

#include <glm/glm.hpp>

#include "platform/renderer/IRenderPath.h"

namespace plce::world {

namespace {

// Module-private material handles, populated by init() and consumed by
// every MeshBuilder flush. Zero-initialised handles are invalid; the
// Vulkan record_draw_call path rejects DrawCalls that carry one, so a
// caller who forgets to call init() just gets no visible output (plus
// eventually a validation assert once we add one).
struct MaterialTable {
    rp::MaterialHandle opaque{};
    rp::MaterialHandle alpha_test{};
    rp::MaterialHandle transparent{};
    rp::MaterialHandle sky_gradient{};
    rp::MaterialHandle sky_additive{};
    rp::MaterialHandle weather{};
};

MaterialTable s_materials;
bool          s_initialised = false;

rp::MaterialHandle material_for_kind(MaterialKind k) {
    switch (k) {
        case MaterialKind::opaque:       return s_materials.opaque;
        case MaterialKind::alpha_test:   return s_materials.alpha_test;
        case MaterialKind::transparent:  return s_materials.transparent;
        case MaterialKind::sky_gradient: return s_materials.sky_gradient;
        case MaterialKind::sky_additive: return s_materials.sky_additive;
        case MaterialKind::weather:      return s_materials.weather;
    }
    return {};
}
}  // namespace

rp::MaterialHandle world_material(MaterialKind kind) {
    return material_for_kind(kind);
}

namespace {

// Pack an XYZ float normal into the R8G8B8A8_SNORM encoding the basic
// vertex shader expects. Shared with UiDraw's item-in-hand mesh build
// but kept per-module to avoid a cross-include.
uint32_t pack_normal_snorm(float x, float y, float z) {
    auto clamp_snorm = [](float v) -> int8_t {
        if (v >  1.0f) v =  1.0f;
        if (v < -1.0f) v = -1.0f;
        return static_cast<int8_t>(v * 127.0f);
    };
    const uint8_t xx = static_cast<uint8_t>(clamp_snorm(x));
    const uint8_t yy = static_cast<uint8_t>(clamp_snorm(y));
    const uint8_t zz = static_cast<uint8_t>(clamp_snorm(z));
    return uint32_t(xx) | (uint32_t(yy) << 8) | (uint32_t(zz) << 16);
}

// The basic shader treats a_lm_raw.x <= -500 as "use global lightmap"
// — 0xfe00fe00 sign-extends to -512 on the x component, matching the
// legacy Tesselator sentinel. Entity / particle draws don't carry
// per-vertex lightmap coords, so every vertex gets this value.
constexpr uint32_t kNoLightmapSentinel = 0xfe00fe00u;

void snapshot_transform(float out[16]) {
    const float* proj = RenderPath.MatrixGet(rp::MatrixStack::projection);
    const float* mv   = RenderPath.MatrixGet(rp::MatrixStack::modelview);
    glm::mat4 p(1.0f), m(1.0f);
    std::memcpy(&p[0][0], proj, sizeof(float) * 16);
    std::memcpy(&m[0][0], mv,   sizeof(float) * 16);
    const glm::mat4 pm = p * m;
    std::memcpy(out, &pm[0][0], sizeof(float) * 16);
}

void snapshot_uv_transform(float scale[2], float offset[2]) {
    const float* tm = RenderPath.MatrixGet(rp::MatrixStack::texture);
    scale[0]  = tm[0];
    scale[1]  = tm[5];
    offset[0] = tm[12];
    offset[1] = tm[13];
}

// Capture just the modelview matrix — shader's normal matrix comes from
// mat3(dc.mv_transform). Kept separate from snapshot_transform's
// proj*mv so per-vertex lighting can rotate model normals into world
// space without the projection leaking into the normal path.
void snapshot_mv_transform(float out[16]) {
    const float* mv = RenderPath.MatrixGet(rp::MatrixStack::modelview);
    std::memcpy(out, mv, sizeof(float) * 16);
}

void push_to_bucket(MaterialKind kind, const rp::DrawCall& dc) {
    switch (kind) {
        case MaterialKind::opaque:      rp::world_draws::push_opaque(dc);      break;
        case MaterialKind::alpha_test:  rp::world_draws::push_alpha_test(dc);  break;
        case MaterialKind::transparent: rp::world_draws::push_transparent(dc); break;
    }
}

}  // namespace

void init() {
    if (s_initialised) return;

    // Opaque world draws: solid mobs, painting backs, chest bodies, etc.
    // Alpha-test off, blend off, depth<=, cull CCW back, lit (picks up
    // frame directional lights via the shader's per-vertex lighting path).
    rp::MaterialDesc op{};
    op.shader      = rp::ShaderPath::standard;
    op.blend       = rp::BlendMode::opaque;
    op.textured    = true;
    op.lit         = true;
    op.fog_enabled = true;
    op.alpha_test  = rp::AlphaTest::off;
    op.depth_test  = rp::DepthTest::less_equal;
    op.depth_write = true;
    op.cull        = rp::CullMode::none;
    s_materials.opaque = RenderPath.create_material(op);

    // Cutout: foliage, hair tufts, entity texture holes. Alpha-tested
    // against 0.1 so the background atlas pixels discard without
    // needing a blend state.
    rp::MaterialDesc ct{};
    ct.shader      = rp::ShaderPath::standard;
    ct.blend       = rp::BlendMode::opaque;
    ct.textured    = true;
    ct.lit         = true;
    ct.fog_enabled = true;
    ct.alpha_test  = rp::AlphaTest::greater;
    ct.alpha_ref   = 0.1f;
    ct.depth_test  = rp::DepthTest::less_equal;
    ct.depth_write = true;
    ct.cull        = rp::CullMode::none;
    s_materials.alpha_test = RenderPath.create_material(ct);

    // Transparent: glass panes, water, particle billboards, portal.
    // Alpha-blend with depth test but no depth write so pixels behind
    // a transparent face still sort correctly.
    rp::MaterialDesc tr{};
    tr.shader      = rp::ShaderPath::standard;
    tr.blend       = rp::BlendMode::alpha;
    tr.textured    = true;
    tr.lit         = true;
    tr.fog_enabled = true;
    tr.alpha_test  = rp::AlphaTest::off;
    tr.depth_test  = rp::DepthTest::less_equal;
    tr.depth_write = false;
    tr.cull        = rp::CullMode::none;
    s_materials.transparent = RenderPath.create_material(tr);

    // Sky gradient: overworld sky dome, dark dome, end-dim cube, sunrise
    // cone. Alpha-blend (vertex colour carries per-vertex sky colour +
    // alpha for horizon fade); unlit; fog off (we ARE the sky); no
    // depth write so terrain cut-in stays correct; texture off — most
    // sky geometry is vertex-coloured only, texture variants can set
    // texture_override on the DrawCall explicitly.
    rp::MaterialDesc sg{};
    sg.shader      = rp::ShaderPath::standard;
    sg.blend       = rp::BlendMode::alpha;
    sg.textured    = false;
    sg.lit         = false;
    sg.fog_enabled = false;
    sg.alpha_test  = rp::AlphaTest::off;
    sg.depth_test  = rp::DepthTest::less_equal;
    sg.depth_write = false;
    sg.cull        = rp::CullMode::none;
    s_materials.sky_gradient = RenderPath.create_material(sg);

    // Sky additive: stars, halo ring. src_alpha * one additive blend
    // so stars brighten the sky dome under them. Depth on, depth write
    // off. Texture off for stars (vertex colour only); halo ring also
    // vertex-coloured.
    rp::MaterialDesc sa{};
    sa.shader           = rp::ShaderPath::standard;
    sa.blend            = rp::BlendMode::custom;
    sa.blend_src_custom = rp::BlendFactor::src_alpha;
    sa.blend_dst_custom = rp::BlendFactor::one;
    sa.textured         = false;
    sa.lit              = false;
    sa.fog_enabled      = false;
    sa.alpha_test       = rp::AlphaTest::off;
    sa.depth_test       = rp::DepthTest::less_equal;
    sa.depth_write      = false;
    sa.cull             = rp::CullMode::none;
    s_materials.sky_additive = RenderPath.create_material(sa);

    // Weather: rain + snow billboards. Alpha-tested (soft edge on the
    // sprite), lit (picks up the frame ambient so rain dims correctly
    // at night), depth test on + depth write off (so rain doesn't
    // occlude translucent water behind it). Uses per-vertex lightmap
    // via tex2() for biome / block-light-influenced colour.
    rp::MaterialDesc wx{};
    wx.shader      = rp::ShaderPath::standard;
    wx.blend       = rp::BlendMode::alpha;
    wx.textured    = true;
    wx.lit         = true;
    wx.fog_enabled = true;
    wx.alpha_test  = rp::AlphaTest::greater;
    wx.alpha_ref   = 0.1f;
    wx.depth_test  = rp::DepthTest::less_equal;
    wx.depth_write = false;
    wx.cull        = rp::CullMode::none;
    s_materials.weather = RenderPath.create_material(wx);

    s_initialised = true;
}

// --------------------------------------------------------------------------

struct MeshBuilder::Impl {
    MaterialKind kind = MaterialKind::opaque;
    Topology     topology = Topology::quads;
    int          texture_id = 0;
    int          forced_lod = -1;
    float        tint[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    // Sticky per-vertex state, mirroring Tesselator.
    float    cur_u = 0.0f, cur_v = 0.0f;
    uint32_t cur_color = 0x00000000u;  // 0 = sentinel "use state_colour"
    uint32_t cur_normal = 0u;
    uint32_t cur_tex2 = kNoLightmapSentinel;  // global-lightmap fallback
    float    ox = 0.0f, oy = 0.0f, oz = 0.0f;

    std::vector<rp::WorldStandardVertex> verts;  // input vertices
};

MeshBuilder::MeshBuilder(MaterialKind kind, int texture_id)
    : impl_(new Impl) {
    impl_->kind       = kind;
    impl_->texture_id = texture_id;
}

MeshBuilder::~MeshBuilder() { delete impl_; }

void MeshBuilder::set_material(MaterialKind kind) { impl_->kind = kind; }
void MeshBuilder::set_texture(int atlas_id)       { impl_->texture_id = atlas_id; }
void MeshBuilder::set_topology(Topology t) {
    impl_->topology = t;
    // Switching topology discards any pending vertices — the caller
    // is expected to have flushed first if they cared about them.
    impl_->verts.clear();
}
void MeshBuilder::set_forced_lod(int lod)         { impl_->forced_lod = lod; }
void MeshBuilder::set_tint(float r, float g, float b, float a) {
    impl_->tint[0] = r; impl_->tint[1] = g; impl_->tint[2] = b; impl_->tint[3] = a;
}

void MeshBuilder::normal(float nx, float ny, float nz) {
    impl_->cur_normal = pack_normal_snorm(nx, ny, nz);
}

void MeshBuilder::tex(float u, float v) { impl_->cur_u = u; impl_->cur_v = v; }

void MeshBuilder::color(float r, float g, float b) {
    color(r, g, b, 1.0f);
}
void MeshBuilder::color(float r, float g, float b, float a) {
    auto to255 = [](float c) -> int {
        int i = int(c * 255.0f);
        if (i < 0)   i = 0;
        if (i > 255) i = 255;
        return i;
    };
    color(uint8_t(to255(r)), uint8_t(to255(g)),
          uint8_t(to255(b)), uint8_t(to255(a)));
}
void MeshBuilder::color(uint8_t r, uint8_t g, uint8_t b) {
    color(r, g, b, uint8_t(255));
}
void MeshBuilder::color(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    impl_->cur_color = uint32_t(r) |
                       (uint32_t(g) << 8) |
                       (uint32_t(b) << 16) |
                       (uint32_t(a) << 24);
}
void MeshBuilder::color(int packed_rgb) { color(packed_rgb, 255); }
void MeshBuilder::color(int packed_rgb, int alpha) {
    auto clamp = [](int v) -> int { return v < 0 ? 0 : v > 255 ? 255 : v; };
    color(uint8_t(clamp((packed_rgb >> 16) & 0xff)),
          uint8_t(clamp((packed_rgb >>  8) & 0xff)),
          uint8_t(clamp((packed_rgb >>  0) & 0xff)),
          uint8_t(clamp(alpha)));
}
void MeshBuilder::color_packed(uint32_t rgba) { impl_->cur_color = rgba; }

void MeshBuilder::tex2(uint32_t packed_uv) { impl_->cur_tex2 = packed_uv; }

void MeshBuilder::offset(float xo, float yo, float zo) {
    impl_->ox = xo; impl_->oy = yo; impl_->oz = zo;
}

void MeshBuilder::addOffset(float dx, float dy, float dz) {
    impl_->ox += dx; impl_->oy += dy; impl_->oz += dz;
}

void MeshBuilder::vertex(float x, float y, float z) {
    vertexUV(x, y, z, impl_->cur_u, impl_->cur_v);
}

void MeshBuilder::vertexUV(float x, float y, float z, float u, float v) {
    impl_->cur_u = u;
    impl_->cur_v = v;
    rp::WorldStandardVertex wv;
    wv.pos[0] = x + impl_->ox;
    wv.pos[1] = y + impl_->oy;
    wv.pos[2] = z + impl_->oz;
    wv.uv[0]  = u;
    wv.uv[1]  = v;
    wv.color  = impl_->cur_color;
    wv.normal = impl_->cur_normal;
    wv.tex2   = impl_->cur_tex2;
    impl_->verts.push_back(wv);
}

void MeshBuilder::quad(const QuadVertex corners[4]) {
    for (int i = 0; i < 4; ++i) {
        const auto& c = corners[i];
        rp::WorldStandardVertex wv;
        wv.pos[0] = c.x + impl_->ox;
        wv.pos[1] = c.y + impl_->oy;
        wv.pos[2] = c.z + impl_->oz;
        wv.uv[0]  = c.u;
        wv.uv[1]  = c.v;
        wv.color  = c.packed_color;
        wv.normal = impl_->cur_normal;
        wv.tex2   = impl_->cur_tex2;
        impl_->verts.push_back(wv);
    }
}

void MeshBuilder::flush() {
    auto& v = impl_->verts;
    if (v.empty()) return;

    // Choose the output DrawCall primitive + vertex count based on
    // input topology. Quads expand to triangle_list (6 verts per 4);
    // every other input topology passes straight through.
    rp::PrimitiveType out_prim = rp::PrimitiveType::triangle_list;
    uint32_t          out_count = 0;
    size_t            quads = 0;

    switch (impl_->topology) {
        case Topology::quads: {
            quads = v.size() / 4;
            if (quads == 0) { v.clear(); return; }
            out_prim  = rp::PrimitiveType::triangle_list;
            out_count = uint32_t(quads * 6);
            break;
        }
        case Topology::triangles:
            out_prim  = rp::PrimitiveType::triangle_list;
            out_count = uint32_t(v.size() - v.size() % 3);
            break;
        case Topology::triangle_strip:
            if (v.size() < 3) { v.clear(); return; }
            out_prim  = rp::PrimitiveType::triangle_strip;
            out_count = uint32_t(v.size());
            break;
        case Topology::triangle_fan:
            if (v.size() < 3) { v.clear(); return; }
            out_prim  = rp::PrimitiveType::triangle_fan;
            out_count = uint32_t(v.size());
            break;
        case Topology::line_list:
            out_prim  = rp::PrimitiveType::line_list;
            out_count = uint32_t(v.size() - v.size() % 2);
            break;
        case Topology::line_strip:
            if (v.size() < 2) { v.clear(); return; }
            out_prim  = rp::PrimitiveType::line_strip;
            out_count = uint32_t(v.size());
            break;
    }
    if (out_count == 0) { v.clear(); return; }

    auto [tvb, span] = RenderPath.alloc_transient_vertices(
        out_count, rp::VertexLayout::world_standard, out_prim);
    if (span.empty()) { v.clear(); return; }

    auto* out = reinterpret_cast<rp::WorldStandardVertex*>(span.data());
    if (impl_->topology == Topology::quads) {
        for (size_t q = 0; q < quads; ++q) {
            const size_t i = q * 4;
            out[q * 6 + 0] = v[i + 0];
            out[q * 6 + 1] = v[i + 1];
            out[q * 6 + 2] = v[i + 2];
            out[q * 6 + 3] = v[i + 0];
            out[q * 6 + 4] = v[i + 2];
            out[q * 6 + 5] = v[i + 3];
        }
    } else {
        std::memcpy(out, v.data(), size_t(out_count) * sizeof(rp::WorldStandardVertex));
    }

    rp::DrawCall dc{};
    dc.source                 = rp::VertexSource::transient;
    dc.transient              = tvb;
    dc.material               = material_for_kind(impl_->kind);
    // Sync flush: caller is in the middle of the legacy world pass
    // (ModelPart::render / entity renderer / tile-entity). Live legacy
    // state is authoritative — leave dc.transform / mv_transform /
    // uv_scale / uv_offset / tint_color / self_describing at defaults
    // so record_draw_call doesn't override what fill_push_constants
    // pulled from the live matrix stacks + state_colour_ +
    // lighting_enabled_ + chunk_offset_ + tex_stack_. The legacy caller
    // already bound the right texture via RenderPath.TextureBind, set
    // state_colour_ via StateSetColour, etc.
    if (impl_->texture_id > 0) {
        dc.texture_override.index = uint32_t(impl_->texture_id);
    }
    if (impl_->forced_lod >= 0) dc.forced_lod = int8_t(impl_->forced_lod);
    dc.self_describing = false;

    RenderPath.submit_draw_call(dc);

    v.clear();
}

void MeshBuilder::reset() {
    impl_->forced_lod = -1;
    impl_->tint[0] = impl_->tint[1] = impl_->tint[2] = impl_->tint[3] = 1.0f;
    impl_->verts.clear();
}

size_t MeshBuilder::vertex_count() const { return impl_->verts.size(); }

std::vector<rp::WorldStandardVertex> MeshBuilder::take_vertices() {
    auto& v = impl_->verts;
    if (v.empty()) return {};

    // Expand quad input to triangle_list so callers get a mesh that
    // Renderer::create_mesh can feed straight to vkCmdDraw without an
    // intermediate index buffer. Non-quad topologies pass through.
    if (impl_->topology != Topology::quads) {
        std::vector<rp::WorldStandardVertex> out;
        out.swap(v);
        return out;
    }
    const size_t quads = v.size() / 4;
    std::vector<rp::WorldStandardVertex> out;
    out.reserve(quads * 6);
    for (size_t q = 0; q < quads; ++q) {
        const size_t i = q * 4;
        out.push_back(v[i + 0]);
        out.push_back(v[i + 1]);
        out.push_back(v[i + 2]);
        out.push_back(v[i + 0]);
        out.push_back(v[i + 2]);
        out.push_back(v[i + 3]);
    }
    v.clear();
    return out;
}

}  // namespace plce::world
