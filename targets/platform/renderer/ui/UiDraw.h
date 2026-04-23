#pragma once

#include <cstdint>

// Modern 2D UI draw primitives. Callers (Font, Gui, GuiComponent, ...)
// issue typed draws here — the module turns them into rp::DrawCall
// records pushed onto rp::ui_overlay, which the active backend drains
// during render_frame. No rp:: / Vk / bgfx vocabulary leaks through
// this header.
//
// Coordinate space: whatever the caller's current proj * mv matrix
// defines, captured at call time. In practice this is the GUI ortho
// set up by Gui::render (logical pixels × guiScale). Any
// MatrixPush / Scale / Translate scope the caller wraps around a
// block of draws is preserved — the live matrix is snapshotted into
// each DrawCall's transform.
//
// Colours: uint32_t `rgba` is packed with R in the low byte, A in the
// high byte (matches the 0xAARRGGBB convention Gui code uses for its
// `int col` arguments after masking). Float-RGBA variants take a
// straight `[r, g, b, a]` array.

namespace plce::ui {

// Register the module's internal materials with the active render
// path. Idempotent — only the first call creates materials; further
// calls are cheap no-ops. Must be called on the main thread after
// the RenderPath global is set (i.e. after Renderer construction).
void init();

// Solid filled rectangle, alpha-blended. `z` is the vertex depth the
// legacy GuiComponent::blitOffset controls — 0 for most UI, negative
// values (e.g. -90) for elements that need to sit in front of other
// equally-projected draws.
void draw_fill(int x0, int y0, int x1, int y1, uint32_t rgba,
               float z = 0.0f);

// Two-colour vertical gradient from `top_rgba` at y0 to `bottom_rgba`
// at y1, alpha-blended. `z` same as draw_fill.
void draw_fill_gradient(int x0, int y0, int x1, int y1,
                        uint32_t top_rgba, uint32_t bottom_rgba,
                        float z = 0.0f);

// Textured quad sampling an axis-aligned region of the atlas at
// `texture_id`. UVs are in the atlas's 0..1 space. `tint_rgba`
// multiplies the sampled colour; pass 0xFFFFFFFFu for no tint.
void draw_textured_quad(int x, int y, int w, int h,
                        float u0, float v0, float u1, float v1,
                        int texture_id,
                        uint32_t tint_rgba = 0xFFFFFFFFu);

// Float-coordinate variant with explicit z. Used by GuiComponent::blit,
// which pixel-snaps and sub-pixel-shifts coords on its own and needs
// to pass the corner positions directly.
void draw_textured_quad(float x0, float y0, float x1, float y1, float z,
                        float u0, float v0, float u1, float v1,
                        int texture_id,
                        uint32_t tint_rgba = 0xFFFFFFFFu);

// Text glyph quad. Float positions (Font uses sub-pixel math),
// alpha-tested material (discards pixels below alpha_ref so the
// glyph sprite's bg doesn't bleed), and float RGBA to match Font's
// internal color tracking.
void draw_glyph_quad(float x, float y, float w, float h,
                     float u0, float v0, float u1, float v1,
                     int texture_id, const float rgba[4]);

// Fullscreen textured overlay (pumpkin-head blur, portal tint,
// post-effect). UVs default to 0..1 but can be overridden when the
// texture is an atlas region (e.g. portalTile's UV subrect).
void draw_fullscreen_overlay(int w, int h, int texture_id,
                             const float rgba[4],
                             float u0 = 0.0f, float v0 = 0.0f,
                             float u1 = 1.0f, float v1 = 1.0f);

// Fullscreen solid colour fill with depth test disabled. Used for
// sleep / death / damage-flash overlays that need to sit on top of
// the existing HUD regardless of per-element blitOffset. Alpha-blended
// so partially-transparent fills darken rather than replace the frame.
void draw_fullscreen_fill(int w, int h, uint32_t rgba);

// 2D line-list vertex. Used by draw_untextured_lines. Not a general
// vertex format — the underlying DrawCall still uses
// WorldStandardVertex internally; this is the caller-visible shape.
struct LineVertex {
    float    x, y;
    uint32_t rgba;
};

// Batched line-list draw in screen-pixel coordinates. Snapshots the
// live proj*mv at call time (same convention as draw_fill) so any
// MatrixPush/Translate scope the caller wrapped is preserved.
// `vertices` must contain an even number of entries — each consecutive
// pair defines one line segment. Pass 0xFFFFFFFFu for white lines, or
// per-vertex colour for gradients.
void draw_untextured_lines(const LineVertex* vertices, size_t vertex_count);

// Fullscreen vignette with the custom zero / one_minus_src_color
// blend the legacy renderVignette used.
void draw_vignette(int w, int h, int texture_id,
                   const float rgba[4]);

}  // namespace plce::ui
