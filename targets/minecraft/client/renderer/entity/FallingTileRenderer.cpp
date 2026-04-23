#include "FallingTileRenderer.h"

#include <cmath>
#include <memory>

#include "minecraft/client/renderer/TileRenderer.h"
#include "platform/renderer/world/WorldDraw.h"
#include "minecraft/client/renderer/entity/EntityRenderer.h"
#include "minecraft/client/renderer/texture/TextureAtlas.h"
#include "minecraft/world/entity/Entity.h"
#include "minecraft/world/entity/item/FallingTile.h"
#include "minecraft/world/level/Level.h"
#include "minecraft/world/level/tile/AnvilTile.h"
#include "minecraft/world/level/tile/Tile.h"
#include "platform/renderer/renderer.h"
#include "platform/stubs.h"

FallingTileRenderer::FallingTileRenderer() : EntityRenderer() {
    tileRenderer = new TileRenderer();
    this->shadowRadius = 0.5f;
}

void FallingTileRenderer::render(std::shared_ptr<Entity> _tile, double x,
                                 double y, double z, float rot, float a) {
    // 4J - dynamic cast required because we aren't using templates/generics in
    // our version
    std::shared_ptr<FallingTile> tile =
        std::dynamic_pointer_cast<FallingTile>(_tile);
    Level* level = tile->getLevel();

    if (level->getTile(floor(tile->x), floor(tile->y), floor(tile->z)) !=
        tile->tile) {
        RenderPath.MatrixPush();
        RenderPath.MatrixTranslate((float)x, (float)y, (float)z);

        bindTexture(tile);  // 4J was "/terrain.png"
        Tile* tt = Tile::tiles[tile->tile];

        Level* level = tile->getLevel();

        RenderPath.StateSetLightingEnable(false);
        RenderPath.StateSetColour(1, 1, 1,
                  1);  // 4J added - this wouldn't be needed in real opengl as
                       // the block render has vertex colours and so this isn't
                       // use, but our pretend gl always modulates with this
        if (tt == Tile::anvil && tt->getRenderShape() == Tile::SHAPE_ANVIL) {
            tileRenderer->level = level;
            plce::world::MeshBuilder mb(
                plce::world::MaterialKind::alpha_test, 0);
            mb.offset(-std::floor(tile->x) - 0.5f,
                      -std::floor(tile->y) - 0.5f,
                      -std::floor(tile->z) - 0.5f);
            tileRenderer->set_builder(&mb);
            tileRenderer->tesselateAnvilInWorld(
                (AnvilTile*)tt, std::floor(tile->x), std::floor(tile->y),
                std::floor(tile->z), tile->data);
            tileRenderer->set_builder(nullptr);
            mb.offset(0, 0, 0);
            mb.flush();
        } else if (tt == Tile::dragonEgg) {
            tileRenderer->level = level;
            plce::world::MeshBuilder mb(
                plce::world::MaterialKind::alpha_test, 0);
            mb.offset(-std::floor(tile->x) - 0.5f,
                      -std::floor(tile->y) - 0.5f,
                      -std::floor(tile->z) - 0.5f);
            tileRenderer->set_builder(&mb);
            tileRenderer->tesselateInWorld(tt, std::floor(tile->x),
                                           std::floor(tile->y),
                                           std::floor(tile->z));
            tileRenderer->set_builder(nullptr);
            mb.offset(0, 0, 0);
            mb.flush();
        } else if (tt != nullptr) {
            plce::world::MeshBuilder mb(
                plce::world::MaterialKind::alpha_test, 0);
            tileRenderer->set_builder(&mb);
            tileRenderer->setShape(tt);
            tileRenderer->renderBlock(tt, level, std::floor(tile->x),
                                      std::floor(tile->y), std::floor(tile->z),
                                      tile->data);
            tileRenderer->set_builder(nullptr);
            mb.flush();
        }
        RenderPath.StateSetLightingEnable(true);
        RenderPath.MatrixPop();
    }
}

ResourceLocation* FallingTileRenderer::getTextureLocation(
    std::shared_ptr<Entity> mob) {
    return &TextureAtlas::LOCATION_BLOCKS;
}