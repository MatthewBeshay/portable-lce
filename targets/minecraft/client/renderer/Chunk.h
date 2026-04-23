#pragma once
#include <stdint.h>

#include <format>
#include <memory>
#include <mutex>
#include <vector>

#include "LevelRenderer.h"
#include "Tesselator.h"
#include "minecraft/client/renderer/culling/AllowAllCuller.h"

namespace plce::world { class MeshBuilder; }
#include "minecraft/world/phys/AABB.h"

class Level;
class TileEntity;
class Entity;
class Chunk;
class Culler;

class ClipChunk {
public:
    Chunk* chunk;
    int globalIdx;
    bool visible;
    float aabb[6];
    int xm, ym, zm;
};

class Chunk {
private:
    static const int XZSIZE = LevelRenderer::CHUNK_XZSIZE;
    static const int SIZE = LevelRenderer::CHUNK_SIZE;

public:
    Level* level;
    static LevelRenderer* levelRenderer;

private:
#if !defined(_LARGE_WORLDS)
    static Tesselator* t;
#else
    static thread_local uint8_t* m_tlsTileIds;

public:
    static void CreateNewThreadStorage();
    static void ReleaseThreadStorage();
    static uint8_t* GetTileIdsStorage();
#endif

private:
    // P4.4 transitional: TileRenderer now emits into a MeshBuilder via
    // set_builder(). rebuild() creates a scratch one per layer so the
    // tesselate* calls have a valid sink; vertices are dropped when the
    // layer closes (chunks invisible until P5 wires the worker-to-main
    // upload queue).
    std::unique_ptr<plce::world::MeshBuilder> chunk_builder_;

public:
    static int updates;

    int x, y, z;
    int xRender, yRender, zRender;
    int xRenderOffs, yRenderOffs, zRenderOffs;

    int xm, ym, zm;
    AABB bb;
    ClipChunk* clipChunk;
#ifdef OCCLUSION_MODE_BFS
    uint64_t computeConnectivity(const uint8_t* tileIds);
#endif
    int id;
    // public:
    //	std::vector<std::shared_ptr<TileEntity> > renderableTileEntities;
    //// 4J - removed

private:
    LevelRenderer::rteMap* globalRenderableTileEntities;
    std::mutex* globalRenderableTileEntities_cs;
    bool assigned;

public:
    Chunk(Level* level, LevelRenderer::rteMap& globalRenderableTileEntities,
          std::mutex& globalRenderableTileEntities_cs, int x, int y, int z,
          ClipChunk* clipChunk);
    Chunk();
    // Out-of-line so the chunk_builder_ unique_ptr can hold an
    // incomplete plce::world::MeshBuilder forward decl without
    // forcing every Chunk.h consumer to include WorldDraw.h.
    ~Chunk();

    void setPos(int x, int y, int z);

private:
    void translateToPos();
    void reconcileRenderableTileEntities(
        const std::vector<std::shared_ptr<TileEntity> >&
            renderableTileEntities);

public:
    void makeCopyForRebuild(Chunk* source);
    void rebuild();
    float distanceToSqr(std::shared_ptr<Entity> player) const;
    float squishedDistanceToSqr(std::shared_ptr<Entity> player);
    void reset();
    void _delete();

    int getList(int layer);
    void cull(Culler* culler);
    void renderBB();
    bool isEmpty();
    void setDirty();
    void clearDirty();  // 4J added
    bool emptyFlagSet(int layer);
};
