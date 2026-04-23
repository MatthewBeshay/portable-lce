#pragma once

#include <cstdint>

// Modern 3D world-space draw primitives. Callers (entity renderers,
// tile-entity renderers, particle subclasses, sky / clouds / weather)
// feed geometry through plce::world::MeshBuilder — the module turns
// accumulated quad vertices into rp::DrawCall records pushed onto the
// relevant rp::world_draws bucket (opaque / alpha_test / transparent).
//
// The builder's per-vertex API intentionally mirrors the legacy
// Tesselator's shape (vertex, vertexUV, tex, color, normal, offset)
// so migration of the 90+ call sites is mechanical.
//
// No rp:: / Vk / bgfx vocabulary leaks through this header. The
// MeshBuilder owns DrawCall assembly, material lookup, and the matrix
// snapshot, same way plce::ui::draw_* does for the UI layer.

namespace plce::world {

// Input topology for MeshBuilder. The default (`quads`) matches the
// legacy Tesselator mode 0x0007 and gets expanded to a triangle_list
// on flush so the underlying DrawCall can feed the plain vkCmdDraw
// path (no shared quad index buffer). The others pass straight
// through — use them for leash strips, line graphs, fan-centred
// geometry etc.
enum class Topology : uint8_t {
    quads,
    triangles,
    triangle_strip,
    triangle_fan,
    line_list,
    line_strip,
};

// Material families the builder can emit into. Every legacy entity /
// particle draw reduces to one of these three buckets:
//   - opaque:    no alpha_test, no blend (grass, wool, solid mobs).
//   - alpha_test: discard-below-threshold (leaves, foliage, crosses,
//                 holes in entity skins like zombie eyes).
//   - transparent: alpha blend (glass, water, portal, fire, particles).
enum class MaterialKind : uint8_t {
    opaque,
    alpha_test,
    transparent,
};

// Register the module's internal materials with the active render
// path. Idempotent — only the first call creates materials; further
// calls are cheap no-ops. Must be called on the main thread after
// the RenderPath global is set (i.e. after Renderer construction).
// Companion to plce::ui::init().
void init();

// Accumulates 3D world-space geometry and flushes it as a single
// DrawCall into rp::world_draws. Construct per draw (or reuse across
// flushes), call tex / color / normal / offset to set persistent
// state, push quads via vertex / vertexUV, then flush() to emit.
//
// Input topology is quads (4 vertices per face, same as Tesselator's
// mode 0x0007). flush() expands each quad into two triangles before
// allocating the transient vertex buffer, so the underlying
// rp::DrawCall uses triangle_list — the Vulkan record path has no
// quad topology.
//
// The builder snapshots live proj*mv + texture transform at flush()
// time, so any MatrixPush / Scale / Translate / Rotate scope the
// caller wrapped around its draws is preserved exactly the way the
// legacy CBuffCall path did.
class MeshBuilder {
public:
    MeshBuilder(MaterialKind kind = MaterialKind::opaque, int texture_id = 0);

    // Material + texture can change between flushes without
    // destructing the builder. Changing them mid-quad (between
    // partial vertex pushes) is undefined.
    void set_material(MaterialKind kind);
    void set_texture(int atlas_id);

    // Switch input topology. Clears any buffered vertices from an
    // earlier topology (flush() first if you need those emitted).
    void set_topology(Topology t);

    // Optional per-draw overrides. Applied to the DrawCall emitted
    // on the next flush().
    void set_forced_lod(int lod);
    void set_tint(float r, float g, float b, float a);

    // Persistent per-vertex state, mirroring the Tesselator API so
    // entity render code migrates mechanically. All of these stick
    // to vertices written after the call until overridden.
    void normal(float nx, float ny, float nz);
    void tex(float u, float v);
    void color(float r, float g, float b);
    void color(float r, float g, float b, float a);
    void color(uint8_t r, uint8_t g, uint8_t b);
    void color(uint8_t r, uint8_t g, uint8_t b, uint8_t a);
    // Matches legacy Tesselator::color(int c): `c` is 0x00RRGGBB,
    // alpha defaults to 255.
    void color(int packed_rgb);
    // Matches legacy Tesselator::color(int c, int alpha): `c` is
    // 0x00RRGGBB; `alpha` is a 0..255 byte.
    void color(int packed_rgb, int alpha);
    void color_packed(uint32_t rgba);
    // Sticky per-vertex lightmap coord. Packed as `u | (v << 16)`
    // (16-bit each, matching Tesselator::tex2(int)). Every vertex
    // written after this call carries the given packed lightmap
    // until overridden. Default is the sentinel `0xfe00fe00` which
    // the vertex shader treats as "use the global fallback".
    void tex2(uint32_t packed_uv);
    // Legacy Tesselator shim — mipmap filtering on migrated call
    // sites is now driven by the material / forced_lod, not a
    // per-batch Tesselator flag. Kept as a no-op returning the
    // previous value (always false) so the save-restore idiom
    // that TileRenderer uses in the cross/cactus paths compiles.
    bool setMipmapEnable(bool /*enable*/) { return false; }
    void offset(float xo, float yo, float zo);
    // Incremental offset — Tesselator::addOffset parity. Shifts vertex
    // positions by (dx, dy, dz) on top of the current sticky offset.
    void addOffset(float dx, float dy, float dz);

    // Push a vertex. vertexUV sets the current (u, v) before writing;
    // vertex uses whatever (u, v) is current. Four consecutive vertex
    // calls form one quad.
    void vertex(float x, float y, float z);
    void vertexUV(float x, float y, float z, float u, float v);

    // Push a complete quad in one call — helper for renderers that
    // already have the 4 corners handy.
    struct QuadVertex {
        float x, y, z;
        float u, v;
        uint32_t packed_color;  // 0 = use material state_colour
    };
    void quad(const QuadVertex corners[4]);

    // Submit the accumulated geometry as one DrawCall on the kind's
    // bucket. Vertex state (current u/v/color/normal/offset) survives
    // flush so the builder can be reused across multiple submissions
    // in the same draw pass. Accumulated vertices are consumed.
    void flush();

    // Reset the vertex accumulator and all per-draw overrides back to
    // defaults. Kind + texture + persistent vertex state (color /
    // normal / offset / u / v) are preserved.
    void reset();

    // Non-copyable (internal std::vector, no gain from copies).
    MeshBuilder(const MeshBuilder&) = delete;
    MeshBuilder& operator=(const MeshBuilder&) = delete;

private:
    // Opaque pimpl — the implementation pulls in std::vector and the
    // internal WorldStandardVertex layout, which shouldn't infect
    // headers included by every entity renderer.
    struct Impl;
    Impl* impl_;
public:
    ~MeshBuilder();
};

}  // namespace plce::world
