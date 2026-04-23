#include "EntityRenderer.h"

#include <cmath>
#include <numbers>

#include "EntityRenderDispatcher.h"
#include "java/Class.h"
#include "minecraft/client/Options.h"
#include "minecraft/client/renderer/Textures.h"
#include "minecraft/client/renderer/TileRenderer.h"
#include "minecraft/client/renderer/texture/TextureAtlas.h"
#include "minecraft/client/resources/ResourceLocation.h"
#include "minecraft/world/Icon.h"
#include "minecraft/world/entity/Entity.h"
#include "minecraft/world/entity/Mob.h"
#include "minecraft/world/entity/animal/Animal.h"
#include "minecraft/world/level/Level.h"
#include "minecraft/world/level/tile/FireTile.h"
#include "minecraft/world/level/tile/Tile.h"
#include "minecraft/world/phys/AABB.h"
#include "platform/renderer/renderer.h"
#include "platform/renderer/world/WorldDraw.h"
#include "platform/stubs.h"

ResourceLocation EntityRenderer::SHADOW_LOCATION =
    ResourceLocation(TN__CLAMP__MISC_SHADOW);

// 4J - added
EntityRenderer::EntityRenderer() {
    model = nullptr;
    tileRenderer = new TileRenderer();
    shadowRadius = 0;
    shadowStrength = 1.0f;
}

EntityRenderer::~EntityRenderer() { delete tileRenderer; }

void EntityRenderer::bindTexture(std::shared_ptr<Entity> entity) {
    bindTexture(getTextureLocation(entity));
}

void EntityRenderer::bindTexture(ResourceLocation* location) {
    entityRenderDispatcher->textures->bindTexture(location);
}

bool EntityRenderer::bindTexture(const std::string& urlTexture,
                                 int backupTexture) {
    Textures* t = entityRenderDispatcher->textures;

    // 4J-PB - no http textures on the xbox, mem textures instead

    // int id = t->loadHttpTexture(urlTexture, backupTexture);
    int id = t->loadMemTexture(urlTexture, backupTexture);

    if (id >= 0) {
        RenderPath.TextureBind(id);
        t->clearLastBoundId();
        return true;
    } else {
        return false;
    }
}

bool EntityRenderer::bindTexture(const std::string& urlTexture,
                                 const std::string& backupTexture) {
    Textures* t = entityRenderDispatcher->textures;

    // 4J-PB - no http textures on the xbox, mem textures instead

    // int id = t->loadHttpTexture(urlTexture, backupTexture);
    int id = t->loadMemTexture(urlTexture, backupTexture);

    if (id >= 0) {
        RenderPath.TextureBind(id);
        t->clearLastBoundId();
        return true;
    } else {
        return false;
    }
}

void EntityRenderer::renderFlame(std::shared_ptr<Entity> e, double x, double y,
                                 double z, float a) {
    RenderPath.StateSetLightingEnable(false);

    Icon* fire1 = Tile::fire->getTextureLayer(0);
    Icon* fire2 = Tile::fire->getTextureLayer(1);

    RenderPath.MatrixPush();
    RenderPath.MatrixTranslate((float)x, (float)y, (float)z);

    float s = e->bbWidth * 1.4f;
    RenderPath.MatrixScale(s, s, s);
    bindTexture(&TextureAtlas::LOCATION_BLOCKS);

    float r = 0.5f;
    float xo = 0.0f;

    float h = e->bbHeight / s;
    float yo = (float)(e->y - e->bb.y0);

    RenderPath.MatrixRotate((-entityRenderDispatcher->playerRotY)*(std::numbers::pi_v<float>/180.f), 0, 1, 0);

    RenderPath.MatrixTranslate(0, 0, -0.3f + ((int)h) * 0.02f);
    RenderPath.StateSetColour(1, 1, 1, 1);
    float zo = 0;
    int ss = 0;
    plce::world::MeshBuilder mb(plce::world::MaterialKind::alpha_test, 0);
    while (h > 0) {
        Icon* tex = nullptr;
        if (ss % 2 == 0) {
            tex = fire1;
        } else {
            tex = fire2;
        }

        float u0 = tex->getU0();
        float v0 = tex->getV0();
        float u1 = tex->getU1();
        float v1 = tex->getV1();

        if (ss / 2 % 2 == 0) {
            float tmp = u1;
            u1 = u0;
            u0 = tmp;
        }
        mb.vertexUV(r - xo,  0 - yo,    zo, u1, v1);
        mb.vertexUV(-r - xo, 0 - yo,    zo, u0, v1);
        mb.vertexUV(-r - xo, 1.4f - yo, zo, u0, v0);
        mb.vertexUV(r - xo,  1.4f - yo, zo, u1, v0);
        h -= 0.45f;
        yo -= 0.45f;
        r *= 0.9f;
        zo += 0.03f;
        ss++;
    }
    mb.flush();
    RenderPath.MatrixPop();
    RenderPath.StateSetLightingEnable(true);
}
void EntityRenderer::renderShadow(std::shared_ptr<Entity> e, double x, double y,
                                  double z, float pow, float a) {
    RenderPath.StateSetLightingEnable(false);
    RenderPath.StateSetBlendEnable(true);
    RenderPath.StateSetBlendFunc(rp::BlendFactor::src_alpha, rp::BlendFactor::one_minus_src_alpha);

    entityRenderDispatcher->textures->bindTexture(&SHADOW_LOCATION);

    Level* level = getLevel();

    RenderPath.StateSetDepthMask(false);
    float r = shadowRadius;
    float fYLocalPlayerShadowOffset = 0.0f;

    if (e->instanceof(eTYPE_MOB)) {
        std::shared_ptr<Mob> mob = std::dynamic_pointer_cast<Mob>(e);
        r *= mob->getSizeScale();

        if (mob->instanceof(eTYPE_ANIMAL)) {
            if (std::dynamic_pointer_cast<Animal>(mob)->isBaby()) {
                r *= 0.5f;
            }
        }
    }

    double ex = e->xOld + (e->x - e->xOld) * a;
    double ey = e->yOld + (e->y - e->yOld) * a + e->getShadowHeightOffs();

    // 4J-PB - local players seem to have a position at their head, and remote
    // players have a foot position. get the shadow to render by changing the
    // check here depending on the player type
    if (e->instanceof(eTYPE_LOCALPLAYER)) {
        ey -= 1.62;
        fYLocalPlayerShadowOffset = -1.62f;
    }
    double ez = e->zOld + (e->z - e->zOld) * a;

    int x0 = std::floor(ex - r);
    int x1 = std::floor(ex + r);
    int y0 = std::floor(ey - r);
    int y1 = std::floor(ey);
    int z0 = std::floor(ez - r);
    int z1 = std::floor(ez + r);

    double xo = x - ex;
    double yo = y - ey;
    double zo = z - ez;

    plce::world::MeshBuilder mb(plce::world::MaterialKind::transparent, 0);
    for (int xt = x0; xt <= x1; xt++)
        for (int yt = y0; yt <= y1; yt++)
            for (int zt = z0; zt <= z1; zt++) {
                int t = level->getTile(xt, yt - 1, zt);
                if (t > 0 && level->getRawBrightness(xt, yt, zt) > 3) {
                    renderTileShadow(mb, Tile::tiles[t], x,
                                     y + e->getShadowHeightOffs() +
                                         fYLocalPlayerShadowOffset,
                                     z, xt, yt, zt, pow, r, xo,
                                     yo + e->getShadowHeightOffs() +
                                         fYLocalPlayerShadowOffset,
                                     zo);
                }
            }
    mb.flush();

    RenderPath.StateSetColour(1, 1, 1, 1);
    RenderPath.StateSetBlendEnable(false);
    RenderPath.StateSetDepthMask(true);
    RenderPath.StateSetLightingEnable(true);
}

Level* EntityRenderer::getLevel() { return entityRenderDispatcher->level; }

void EntityRenderer::renderTileShadow(plce::world::MeshBuilder& mb,
                                      Tile* tt, double x, double y, double z,
                                      int xt, int yt, int zt, float pow,
                                      float r, double xo, double yo,
                                      double zo) {
    if (!tt->isCubeShaped()) return;

    double a = ((pow - (y - (yt + yo)) / 2) * 0.5f) *
               getLevel()->getBrightness(xt, yt, zt);
    if (a < 0) return;
    if (a > 1) a = 1;

    mb.color(1.0f, 1.0f, 1.0f, (float)a);

    double x0 = xt + tt->getShapeX0() + xo;
    double x1 = xt + tt->getShapeX1() + xo;
    double y0 = yt + tt->getShapeY0() + yo + 1.0 / 64.0f;
    double z0 = zt + tt->getShapeZ0() + zo;
    double z1 = zt + tt->getShapeZ1() + zo;

    float u0 = (float)((x - (x0)) / 2 / r + 0.5f);
    float u1 = (float)((x - (x1)) / 2 / r + 0.5f);
    float v0 = (float)((z - (z0)) / 2 / r + 0.5f);
    float v1 = (float)((z - (z1)) / 2 / r + 0.5f);

    mb.vertexUV((float)x0, (float)y0, (float)z0, u0, v0);
    mb.vertexUV((float)x0, (float)y0, (float)z1, u0, v1);
    mb.vertexUV((float)x1, (float)y0, (float)z1, u1, v1);
    mb.vertexUV((float)x1, (float)y0, (float)z0, u1, v0);
}

void EntityRenderer::render(AABB* bb, double xo, double yo, double zo) {
    RenderPath.StateSetTextureEnable(false);
    RenderPath.StateSetColour(1, 1, 1, 1);
    plce::world::MeshBuilder mb(plce::world::MaterialKind::opaque, 0);
    mb.offset((float)xo, (float)yo, (float)zo);
    mb.normal(0, 0, -1);
    mb.vertex((float)bb->x0, (float)bb->y1, (float)bb->z0);
    mb.vertex((float)bb->x1, (float)bb->y1, (float)bb->z0);
    mb.vertex((float)bb->x1, (float)bb->y0, (float)bb->z0);
    mb.vertex((float)bb->x0, (float)bb->y0, (float)bb->z0);

    mb.normal(0, 0, 1);
    mb.vertex((float)bb->x0, (float)bb->y0, (float)bb->z1);
    mb.vertex((float)bb->x1, (float)bb->y0, (float)bb->z1);
    mb.vertex((float)bb->x1, (float)bb->y1, (float)bb->z1);
    mb.vertex((float)bb->x0, (float)bb->y1, (float)bb->z1);

    mb.normal(0, -1, 0);
    mb.vertex((float)bb->x0, (float)bb->y0, (float)bb->z0);
    mb.vertex((float)bb->x1, (float)bb->y0, (float)bb->z0);
    mb.vertex((float)bb->x1, (float)bb->y0, (float)bb->z1);
    mb.vertex((float)bb->x0, (float)bb->y0, (float)bb->z1);

    mb.normal(0, 1, 0);
    mb.vertex((float)bb->x0, (float)bb->y1, (float)bb->z1);
    mb.vertex((float)bb->x1, (float)bb->y1, (float)bb->z1);
    mb.vertex((float)bb->x1, (float)bb->y1, (float)bb->z0);
    mb.vertex((float)bb->x0, (float)bb->y1, (float)bb->z0);

    mb.normal(-1, 0, 0);
    mb.vertex((float)bb->x0, (float)bb->y0, (float)bb->z1);
    mb.vertex((float)bb->x0, (float)bb->y1, (float)bb->z1);
    mb.vertex((float)bb->x0, (float)bb->y1, (float)bb->z0);
    mb.vertex((float)bb->x0, (float)bb->y0, (float)bb->z0);

    mb.normal(1, 0, 0);
    mb.vertex((float)bb->x1, (float)bb->y0, (float)bb->z0);
    mb.vertex((float)bb->x1, (float)bb->y1, (float)bb->z0);
    mb.vertex((float)bb->x1, (float)bb->y1, (float)bb->z1);
    mb.vertex((float)bb->x1, (float)bb->y0, (float)bb->z1);
    mb.flush();
    RenderPath.StateSetTextureEnable(true);
}

void EntityRenderer::renderFlat(AABB* bb) {
    plce::world::MeshBuilder mb(plce::world::MaterialKind::opaque, 0);
    mb.vertex((float)bb->x0, (float)bb->y1, (float)bb->z0);
    mb.vertex((float)bb->x1, (float)bb->y1, (float)bb->z0);
    mb.vertex((float)bb->x1, (float)bb->y0, (float)bb->z0);
    mb.vertex((float)bb->x0, (float)bb->y0, (float)bb->z0);
    mb.vertex((float)bb->x0, (float)bb->y0, (float)bb->z1);
    mb.vertex((float)bb->x1, (float)bb->y0, (float)bb->z1);
    mb.vertex((float)bb->x1, (float)bb->y1, (float)bb->z1);
    mb.vertex((float)bb->x0, (float)bb->y1, (float)bb->z1);
    mb.vertex((float)bb->x0, (float)bb->y0, (float)bb->z0);
    mb.vertex((float)bb->x1, (float)bb->y0, (float)bb->z0);
    mb.vertex((float)bb->x1, (float)bb->y0, (float)bb->z1);
    mb.vertex((float)bb->x0, (float)bb->y0, (float)bb->z1);
    mb.vertex((float)bb->x0, (float)bb->y1, (float)bb->z1);
    mb.vertex((float)bb->x1, (float)bb->y1, (float)bb->z1);
    mb.vertex((float)bb->x1, (float)bb->y1, (float)bb->z0);
    mb.vertex((float)bb->x0, (float)bb->y1, (float)bb->z0);
    mb.vertex((float)bb->x0, (float)bb->y0, (float)bb->z1);
    mb.vertex((float)bb->x0, (float)bb->y1, (float)bb->z1);
    mb.vertex((float)bb->x0, (float)bb->y1, (float)bb->z0);
    mb.vertex((float)bb->x0, (float)bb->y0, (float)bb->z0);
    mb.vertex((float)bb->x1, (float)bb->y0, (float)bb->z0);
    mb.vertex((float)bb->x1, (float)bb->y1, (float)bb->z0);
    mb.vertex((float)bb->x1, (float)bb->y1, (float)bb->z1);
    mb.vertex((float)bb->x1, (float)bb->y0, (float)bb->z1);
    mb.flush();
}

void EntityRenderer::renderFlat(float x0, float y0, float z0, float x1,
                                float y1, float z1) {
    plce::world::MeshBuilder mb(plce::world::MaterialKind::opaque, 0);
    mb.vertex(x0, y1, z0);
    mb.vertex(x1, y1, z0);
    mb.vertex(x1, y0, z0);
    mb.vertex(x0, y0, z0);
    mb.vertex(x0, y0, z1);
    mb.vertex(x1, y0, z1);
    mb.vertex(x1, y1, z1);
    mb.vertex(x0, y1, z1);
    mb.vertex(x0, y0, z0);
    mb.vertex(x1, y0, z0);
    mb.vertex(x1, y0, z1);
    mb.vertex(x0, y0, z1);
    mb.vertex(x0, y1, z1);
    mb.vertex(x1, y1, z1);
    mb.vertex(x1, y1, z0);
    mb.vertex(x0, y1, z0);
    mb.vertex(x0, y0, z1);
    mb.vertex(x0, y1, z1);
    mb.vertex(x0, y1, z0);
    mb.vertex(x0, y0, z0);
    mb.vertex(x1, y0, z0);
    mb.vertex(x1, y1, z0);
    mb.vertex(x1, y1, z1);
    mb.vertex(x1, y0, z1);
    mb.flush();
}

void EntityRenderer::init(EntityRenderDispatcher* entityRenderDispatcher) {
    this->entityRenderDispatcher = entityRenderDispatcher;
}

void EntityRenderer::postRender(std::shared_ptr<Entity> entity, double x,
                                double y, double z, float rot, float a,
                                bool bRenderPlayerShadow) {
    if (!entityRenderDispatcher
             ->isGuiRender)  // 4J - added, don't render shadow in gui as it
                             // uses its own blending, and we have globally
                             // enabled blending for interface opacity
    {
        if (bRenderPlayerShadow &&
            entityRenderDispatcher->options->fancyGraphics &&
            shadowRadius > 0 && !entity->isInvisible()) {
            double dist = entityRenderDispatcher->distanceToSqr(
                entity->x, entity->y, entity->z);
            float pow = (float)((1 - dist / (16.0f * 16.0f)) * shadowStrength);
            if (pow > 0) {
                renderShadow(entity, x, y, z, pow, a);
            }
        }
    }
    if (entity->isOnFire()) renderFlame(entity, x, y, z, a);
}

Font* EntityRenderer::getFont() { return entityRenderDispatcher->getFont(); }

void EntityRenderer::registerTerrainTextures(IconRegister* iconRegister) {}

ResourceLocation* EntityRenderer::getTextureLocation(
    std::shared_ptr<Entity> mob) {
    return nullptr;
}
