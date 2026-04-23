#include "PistonPieceRenderer.h"

#include <memory>

#include "minecraft/client/Lighting.h"
#include "minecraft/client/renderer/Textures.h"
#include "platform/renderer/world/WorldDraw.h"
#include "minecraft/client/renderer/TileRenderer.h"
#include "minecraft/client/renderer/texture/TextureAtlas.h"
#include "minecraft/client/resources/ResourceLocation.h"
#include "minecraft/world/level/Level.h"
#include "minecraft/world/level/tile/Tile.h"
#include "minecraft/world/level/tile/entity/PistonPieceTileEntity.h"
#include "minecraft/world/level/tile/entity/TileEntity.h"
#include "minecraft/world/level/tile/piston/PistonBaseTile.h"
#include "minecraft/world/level/tile/piston/PistonExtensionTile.h"
#include "platform/renderer/renderer.h"
#include "platform/stubs.h"

ResourceLocation PistonPieceRenderer::SIGN_LOCATION =
    ResourceLocation(TN_ITEM_SIGN);

PistonPieceRenderer::PistonPieceRenderer() { tileRenderer = nullptr; }

void PistonPieceRenderer::render(std::shared_ptr<TileEntity> _entity, double x,
                                 double y, double z, float a, bool setColor,
                                 float alpha, bool useCompiled) {
    // 4J - dynamic cast required because we aren't using templates/generics in
    // our version
    std::shared_ptr<PistonPieceEntity> entity =
        std::dynamic_pointer_cast<PistonPieceEntity>(_entity);

    Tile* tile = Tile::tiles[entity->getId()];
    if (tile != nullptr &&
        entity->getProgress(a) <=
            1)  // 4J - changed condition from < to <= as our chunk update is
                // async to main thread and so we can have to render these with
                // progress of 1
    {
        bindTexture(&TextureAtlas::LOCATION_BLOCKS);

        Lighting::turnOff();
        RenderPath.StateSetColour(1, 1, 1, 1);
        RenderPath.StateSetBlendFunc(rp::BlendFactor::src_alpha, rp::BlendFactor::one_minus_src_alpha);
        RenderPath.StateSetBlendEnable(true);
        RenderPath.StateSetFaceCull(false);

        plce::world::MeshBuilder mb(plce::world::MaterialKind::alpha_test, 0);
        mb.offset((float)x - entity->x + entity->getXOff(a),
                  (float)y - entity->y + entity->getYOff(a),
                  (float)z - entity->z + entity->getZOff(a));
        mb.color(uint8_t(255), uint8_t(255), uint8_t(255));
        tileRenderer->set_builder(&mb);
        if (tile == Tile::pistonExtension && entity->getProgress(a) < 0.5f) {
            tileRenderer->tesselatePistonArmNoCulling(tile, entity->x,
                                                      entity->y, entity->z,
                                                      false, entity->getData());
        } else if (entity->isSourcePiston() && !entity->isExtending()) {
            Tile::pistonExtension->setOverrideTopTexture(
                ((PistonBaseTile*)tile)->getPlatformTexture());
            tileRenderer->tesselatePistonArmNoCulling(
                Tile::pistonExtension, entity->x, entity->y, entity->z,
                entity->getProgress(a) < 0.5f, entity->getData());
            Tile::pistonExtension->clearOverrideTopTexture();

            mb.offset((float)x - entity->x, (float)y - entity->y,
                      (float)z - entity->z);
            tileRenderer->tesselatePistonBaseForceExtended(
                tile, entity->x, entity->y, entity->z, entity->getData());
        } else {
            tileRenderer->tesselateInWorldNoCulling(tile, entity->x, entity->y,
                                                    entity->z,
                                                    entity->getData(), entity);
        }
        tileRenderer->set_builder(nullptr);
        mb.offset(0, 0, 0);
        mb.flush();

        Lighting::turnOn();
    }
}

void PistonPieceRenderer::onNewLevel(Level* level) {
    delete tileRenderer;
    tileRenderer = new TileRenderer(level);
}
