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
    const float us = 1.0f / 256.0f;
    const float vs = 1.0f / 256.0f;

    // Pixel-snap + sub-pixel shift to keep 1:1 texel mapping on
    // arbitrary-DPI backbuffers. The shift is legacy behaviour from
    // the classic GUI renderer — kept verbatim to avoid visual
    // regressions.
    const float extraShift = 0.75f;
    const float scale  = Gui::currentGuiScaleFactor;
    const float dx     = (extraShift * (float)Minecraft::GetInstance()->width /
                          (float)Minecraft::GetInstance()->width_phys) / scale;
    const float dy     = extraShift / scale;
    const float fx     = floorf((float)x * scale) / scale;
    const float fy     = floorf((float)y * scale) / scale;
    const float fw     = floorf((float)w * scale) / scale;
    const float fh     = floorf((float)h * scale) / scale;

    const float u0 = (sx      ) * us;
    const float u1 = (sx + w  ) * us;
    const float v0 = (sy      ) * vs;
    const float v1 = (sy + h  ) * vs;

    // The caller pre-bound the texture via textures->bindTexture(&LOC);
    // currentBoundId() hands back that id so the DrawCall can capture
    // it and render_frame resolves the right atlas later, no matter
    // what else is bound by then.
    const int tex_id = Minecraft::GetInstance()->textures->currentBoundId();
    if (tex_id < 0) return;

    plce::ui::draw_textured_quad(
        fx - dx, fy - dy, fx + fw - dx, fy + fh - dy,
        (float)blitOffset,
        u0, v0, u1, v1,
        tex_id);
}

