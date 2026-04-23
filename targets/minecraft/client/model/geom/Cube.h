#pragma once
#include <array>
#include <cstdint>
#include <string>

#include "minecraft/client/model/Polygon.h"
#include "minecraft/client/model/Vertex.h"

class Model;
class ModelPart;
class Vertex;
class _Polygon;

namespace plce::world { class MeshBuilder; }

class Cube {
private:
    std::array<Vertex, 8> vertices;
    std::array<_Polygon, 6> polygons;
    uint8_t faceCount;

public:
    const float x0, y0, z0, x1, y1, z1;
    std::string id;

public:
    Cube(ModelPart* modelPart, int xTexOffs, int yTexOffs, float x0, float y0,
         float z0, int w, int h, int d, float g, int faceMask = 63,
         bool bFlipPoly3UVs = false);  // 4J - added faceMask

    void render(plce::world::MeshBuilder& mb, float scale);
    Cube* setId(const std::string& id);
};
