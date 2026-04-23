#include "ExperienceOrbRenderer.h"

#include <math.h>

#include <memory>
#include <numbers>

#include "EntityRenderDispatcher.h"
#include "minecraft/SharedConstants.h"
#include "minecraft/client/renderer/Textures.h"
#include "minecraft/client/resources/ResourceLocation.h"
#include "minecraft/world/entity/Entity.h"
#include "minecraft/world/entity/ExperienceOrb.h"
#include "platform/renderer/renderer.h"
#include "platform/renderer/world/WorldDraw.h"
#include "platform/stubs.h"

ResourceLocation ExperienceOrbRenderer::XP_ORB_LOCATION =
    ResourceLocation(TN_ITEM_EXPERIENCE_ORB);

ExperienceOrbRenderer::ExperienceOrbRenderer() {
    shadowRadius = 0.15f;
    shadowStrength = 0.75f;
}

void ExperienceOrbRenderer::render(std::shared_ptr<Entity> _orb, double x,
                                   double y, double z, float rot, float a) {
    std::shared_ptr<ExperienceOrb> orb =
        std::dynamic_pointer_cast<ExperienceOrb>(_orb);
    RenderPath.MatrixPush();
    RenderPath.MatrixTranslate((float)x, (float)y, (float)z);

    int icon = orb->getIcon();
    bindTexture(orb);  // 4J was "/item/xporb.png"

    float u0 = ((icon % 4) * 16 + 0) / 64.0f;
    float u1 = ((icon % 4) * 16 + 16) / 64.0f;
    float v0 = ((icon / 4) * 16 + 0) / 64.0f;
    float v1 = ((icon / 4) * 16 + 16) / 64.0f;

    float r = 1.0f;
    float xo = 0.5f;
    float yo = 0.25f;

    if (SharedConstants::TEXTURE_LIGHTING) {
        int col = orb->getLightColor(a);
        int u = col % 65536;
        int v = col / 65536;
        RenderPath.StateSetVertexTextureUV(u / 1.0f, v / 1.0f);
        RenderPath.StateSetColour(1, 1, 1, 1);
    } else {
        float br = orb->getBrightness(a);
        RenderPath.StateSetColour(br, br, br, 1);
    }
    float br = 255.0f;
    float rr = (orb->tickCount + a) / 2;
    int rc = (int)((sinf(rr + 0 * std::numbers::pi * 2 / 3) + 1) * 0.5f * br);
    int gc = (int)(br);
    int bc = (int)((sinf(rr + 2 * std::numbers::pi * 2 / 3) + 1) * 0.1f * br);
    int col = rc << 16 | gc << 8 | bc;
    RenderPath.MatrixRotate((180 - entityRenderDispatcher->playerRotY)*(std::numbers::pi_v<float>/180.f), 0, 1, 0);
    RenderPath.MatrixRotate((-entityRenderDispatcher->playerRotX)*(std::numbers::pi_v<float>/180.f), 1, 0, 0);
    float s = 0.3f;
    RenderPath.MatrixScale(s, s, s);

    plce::world::MeshBuilder mb(plce::world::MaterialKind::alpha_test, 0);
    // Legacy tesselator->color(int c, int alpha) packed 0x00RRGGBB +
    // separate alpha; MeshBuilder takes uint8_t channels, so unpack
    // inline to keep the per-orb colour pulse working.
    mb.color(uint8_t((col >> 16) & 0xff),
             uint8_t((col >>  8) & 0xff),
             uint8_t((col >>  0) & 0xff),
             uint8_t(128));
    mb.normal(0, 1, 0);
    mb.vertexUV(0 - xo, 0 - yo, 0, u0, v1);
    mb.vertexUV(r - xo, 0 - yo, 0, u1, v1);
    mb.vertexUV(r - xo, 1 - yo, 0, u1, v0);
    mb.vertexUV(0 - xo, 1 - yo, 0, u0, v0);
    mb.flush();

    RenderPath.StateSetBlendEnable(false);
    RenderPath.MatrixPop();
}

ResourceLocation* ExperienceOrbRenderer::getTextureLocation(
    std::shared_ptr<Entity> mob) {
    return &XP_ORB_LOCATION;
}