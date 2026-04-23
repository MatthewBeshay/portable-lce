#pragma once

#include <memory>

#include "platform/renderer/IRenderPath.h"

class Minecraft;
class ItemInstance;
class Minimap;
class LivingEntity;
class TileRenderer;
class Tesselator;
class Icon;
class ResourceLocation;

class ItemInHandRenderer {
public:
    // 4J - made these public
    static ResourceLocation ENCHANT_GLINT_LOCATION;
    static ResourceLocation MAP_BACKGROUND_LOCATION;
    static ResourceLocation UNDERWATER_LOCATION;

private:
    Minecraft* minecraft;
    std::shared_ptr<ItemInstance> selectedItem;
    float height;
    float oHeight;
    TileRenderer* tileRenderer;
    // listItem / listTerrain are 3D cube meshes precompiled once for
    // item / terrain icon rendering in first- and third-person view;
    // listGlint is the enchant-foil overlay. All three were CBuff ids
    // registered via RenderPath.CBuffStart in the legacy renderer and
    // are now persistent MeshHandles registered via Renderer::create_mesh.
    static rp::MeshHandle listItem, listGlint, listTerrain;

public:
    // 4J Stu - Made public so we can use it from ItemFramRenderer
    Minimap* minimap;

public:
    ItemInHandRenderer(
        Minecraft* mc,
        bool optimisedMinimap = true);  // 4J Added optimisedMinimap param
    void renderItem(std::shared_ptr<LivingEntity> mob,
                    std::shared_ptr<ItemInstance> item, int layer,
                    bool setColor = true);  // 4J added setColor parameter
    static void renderItem3D(
        Tesselator* t, float u0, float v0, float u1, float v1, int width,
        int height, float depth, bool isGlint,
        bool isTerrain);  // 4J added isGlint and isTerrain parameter
public:
    void render(float a);
    void renderScreenEffect(float a);

private:
    void renderTex(float a, Icon* slot);
    void renderWater(float a);
    void renderFire(float a);
    int lastSlot;

public:
    void tick();
    void itemPlaced();
    void itemUsed();
};
