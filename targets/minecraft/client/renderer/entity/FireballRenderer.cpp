#include "FireballRenderer.h"

#include <memory>
#include <numbers>

#include "EntityRenderDispatcher.h"
#include "java/Class.h"
#include "minecraft/client/renderer/texture/TextureAtlas.h"
#include "minecraft/world/Icon.h"
#include "minecraft/world/entity/Entity.h"
#include "minecraft/world/entity/projectile/Fireball.h"
#include "minecraft/world/item/Item.h"
#include "minecraft/world/level/tile/FireTile.h"
#include "minecraft/world/level/tile/Tile.h"
#include "minecraft/world/phys/AABB.h"
#include "platform/renderer/renderer.h"
#include "platform/renderer/world/WorldDraw.h"
#include "platform/stubs.h"

FireballRenderer::FireballRenderer(float scale) { this->scale = scale; }

void FireballRenderer::render(std::shared_ptr<Entity> _fireball, double x,
                              double y, double z, float rot, float a) {
    // 4J - dynamic cast required because we aren't using templates/generics in
    // our version
    std::shared_ptr<Fireball> fireball =
        std::dynamic_pointer_cast<Fireball>(_fireball);

    RenderPath.MatrixPush();

    RenderPath.MatrixTranslate((float)x, (float)y, (float)z);
    (void)0;
    float s = scale;
    RenderPath.MatrixScale(s / 1.0f, s / 1.0f, s / 1.0f);
    Icon* icon = Item::fireball->getIcon(
        fireball->GetType() == eTYPE_DRAGON_FIREBALL ? 1 : 0);  // 14 + 2 * 16;
    bindTexture(fireball);

    float u0 = icon->getU0();
    float u1 = icon->getU1();
    float v0 = icon->getV0();
    float v1 = icon->getV1();

    float r = 1.0f;
    float xo = 0.5f;
    float yo = 0.25f;

    RenderPath.MatrixRotate((180 - entityRenderDispatcher->playerRotY)*(std::numbers::pi_v<float>/180.f), 0, 1, 0);
    RenderPath.MatrixRotate((-entityRenderDispatcher->playerRotX)*(std::numbers::pi_v<float>/180.f), 1, 0, 0);
    plce::world::MeshBuilder mb(plce::world::MaterialKind::alpha_test, 0);
    mb.normal(0, 1, 0);
    mb.vertexUV(0 - xo, 0 - yo, 0, u0, v1);
    mb.vertexUV(r - xo, 0 - yo, 0, u1, v1);
    mb.vertexUV(r - xo, 1 - yo, 0, u1, v0);
    mb.vertexUV(0 - xo, 1 - yo, 0, u0, v0);
    mb.flush();

    RenderPath.MatrixPop();
}

// 4J Added override. Based on EntityRenderer::renderFlame
void FireballRenderer::renderFlame(std::shared_ptr<Entity> e, double x,
                                   double y, double z, float a) {
    RenderPath.StateSetLightingEnable(false);
    Icon* tex = Tile::fire->getTextureLayer(0);

    RenderPath.MatrixPush();
    RenderPath.MatrixTranslate((float)x, (float)y, (float)z);

    float s = e->bbWidth * 1.4f;
    RenderPath.MatrixScale(s, s, s);
    bindTexture(&TextureAtlas::LOCATION_BLOCKS);

    float r = 1.0f;
    float xo = 0.5f;

    float yo = (float)(e->y - e->bb.y0);

    RenderPath.MatrixRotate((180 - entityRenderDispatcher->playerRotY)*(std::numbers::pi_v<float>/180.f), 0, 1, 0);
    RenderPath.MatrixRotate((-entityRenderDispatcher->playerRotX)*(std::numbers::pi_v<float>/180.f), 1, 0, 0);
    RenderPath.MatrixTranslate(0, 0, 0.1f);
    RenderPath.StateSetColour(1, 1, 1, 1);

    float u0 = tex->getU0();
    float v0 = tex->getV0();
    float u1 = tex->getU1();
    float v1 = tex->getV1();

    float tmp = u1;
    u1 = u0;
    u0 = tmp;

    plce::world::MeshBuilder mb(plce::world::MaterialKind::alpha_test, 0);
    mb.normal(0, 1, 0);
    mb.vertexUV(0 - xo, 0 - yo, 0, u1, v1);
    mb.vertexUV(r - xo, 0 - yo, 0, u0, v1);
    mb.vertexUV(r - xo, 1.4f - yo, 0, u0, v0);
    mb.vertexUV(0 - xo, 1.4f - yo, 0, u1, v0);
    mb.flush();

    RenderPath.MatrixPop();
    RenderPath.StateSetLightingEnable(true);
}

ResourceLocation* FireballRenderer::getTextureLocation(
    std::shared_ptr<Entity> mob) {
    return &TextureAtlas::LOCATION_ITEMS;
}
