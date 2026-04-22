#include "GuiComponent.h"

#include <math.h>

#include "minecraft/client/Minecraft.h"
#include "minecraft/client/gui/Font.h"
#include "minecraft/client/gui/Gui.h"
#include "minecraft/client/renderer/Tesselator.h"
#include "platform/renderer/IRenderPath.h"
#include "platform/renderer/renderer.h"
#include "platform/renderer/ui/UiDraw.h"
#include "platform/stubs.h"


void GuiComponent::hLine(int x0, int x1, int y, int col) {
    if (x1 < x0) {
        int tmp = x0;
        x0 = x1;
        x1 = tmp;
    }
    fill(x0, y, x1 + 1, y + 1, col);
}

void GuiComponent::vLine(int x, int y0, int y1, int col) {
    if (y1 < y0) {
        int tmp = y0;
        y0 = y1;
        y1 = tmp;
    }
    fill(x, y0 + 1, x + 1, y1, col);
}

void GuiComponent::fill(int x0, int y0, int x1, int y1, int col) {
    if (x0 < x1) { int tmp = x0; x0 = x1; x1 = tmp; }
    if (y0 < y1) { int tmp = y0; y0 = y1; y1 = tmp; }
    // `col` is an int with byte layout 0xAARRGGBB (A high, R low-ish)
    // but the modern UiDraw API takes 0xAABBGGRR (byte 0 = R, byte 3
    // = A). Swap R and B so the packing matches.
    const uint32_t rgba =
        ((uint32_t(col) >> 16) & 0xFFu)        |  // R → byte 0
        ((uint32_t(col) >>  8) & 0xFFu) <<  8  |  // G → byte 1
        ((uint32_t(col)      ) & 0xFFu) << 16  |  // B → byte 2
        ((uint32_t(col) >> 24) & 0xFFu) << 24;    // A → byte 3
    plce::ui::draw_fill(x0, y0, x1, y1, rgba);
}

namespace {
// Swap R/B from the legacy 0xAARRGGBB int to the UiDraw 0xAABBGGRR
// uint32 layout.
uint32_t legacy_col_to_rgba(int col) {
    return  ((uint32_t(col) >> 16) & 0xFFu)        |
           (((uint32_t(col) >>  8) & 0xFFu) <<  8) |
           (((uint32_t(col)      ) & 0xFFu) << 16) |
           (((uint32_t(col) >> 24) & 0xFFu) << 24);
}
}

void GuiComponent::fillGradient(int x0, int y0, int x1, int y1, int col1,
                                int col2) {
    plce::ui::draw_fill_gradient(x0, y0, x1, y1,
                                 legacy_col_to_rgba(col1),
                                 legacy_col_to_rgba(col2),
                                 float(blitOffset));
}

GuiComponent::GuiComponent() { blitOffset = 0; }

void GuiComponent::drawCenteredString(Font* font, const std::string& str, int x,
                                      int y, int color) {
    font->drawShadow(str, x - (font->width(str)) / 2, y, color);
}

void GuiComponent::drawString(Font* font, const std::string& str, int x, int y,
                              int color) {
    font->drawShadow(str, x, y, color);
}

void GuiComponent::blit(int x, int y, int sx, int sy, int w, int h) {
    float us = 1 / 256.0f;
    float vs = 1 / 256.0f;

    const float extraShift = 0.75f;
    float dx = (extraShift * (float)Minecraft::GetInstance()->width) /
               (float)Minecraft::GetInstance()->width_phys;
    dx /= Gui::currentGuiScaleFactor;
    float dy = extraShift / Gui::currentGuiScaleFactor;
    float fx = (floorf((float)x * Gui::currentGuiScaleFactor)) /
               Gui::currentGuiScaleFactor;
    float fy = (floorf((float)y * Gui::currentGuiScaleFactor)) /
               Gui::currentGuiScaleFactor;
    float fw = (floorf((float)w * Gui::currentGuiScaleFactor)) /
               Gui::currentGuiScaleFactor;
    float fh = (floorf((float)h * Gui::currentGuiScaleFactor)) /
               Gui::currentGuiScaleFactor;

    float u0 = (sx + 0) * us;
    float u1 = (sx + w) * us;
    float v0 = (sy + 0) * vs;
    float v1 = (sy + h) * vs;

    auto [tvb, span] = RenderPath.alloc_transient_vertices(
        4, rp::VertexLayout::world_standard, rp::PrimitiveType::triangle_fan);
    if (span.empty()) return;
    auto* v = reinterpret_cast<rp::WorldStandardVertex*>(span.data());
    v[0] = {{fx + 0  - dx, fy + fh - dy, blitOffset}, {u0, v1}, 0, 0, 0xfe00fe00};
    v[1] = {{fx + fw - dx, fy + fh - dy, blitOffset}, {u1, v1}, 0, 0, 0xfe00fe00};
    v[2] = {{fx + fw - dx, fy + 0  - dy, blitOffset}, {u1, v0}, 0, 0, 0xfe00fe00};
    v[3] = {{fx + 0  - dx, fy + 0  - dy, blitOffset}, {u0, v0}, 0, 0, 0xfe00fe00};

    rp::DrawCall dc{};
    dc.source = rp::VertexSource::transient;
    dc.transient = tvb;
    RenderPath.submit_immediate(dc);
}

