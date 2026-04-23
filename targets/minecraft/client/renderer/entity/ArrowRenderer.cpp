#include "ArrowRenderer.h"

#include <math.h>

#include <memory>
#include <numbers>

#include "minecraft/client/renderer/Textures.h"
#include "minecraft/client/resources/ResourceLocation.h"
#include "minecraft/world/entity/Entity.h"
#include "minecraft/world/entity/projectile/Arrow.h"
#include "platform/renderer/renderer.h"
#include "platform/renderer/world/WorldDraw.h"
#include "platform/stubs.h"

ResourceLocation ArrowRenderer::ARROW_LOCATION =
    ResourceLocation(TN_ITEM_ARROWS);

void ArrowRenderer::render(std::shared_ptr<Entity> _arrow, double x, double y,
                           double z, float rot, float a) {
    // 4J - original version used generics and thus had an input parameter of
    // type Arrow rather than shared_ptr<Entity>  we have here - do some casting
    // around instead
    std::shared_ptr<Arrow> arrow = std::dynamic_pointer_cast<Arrow>(_arrow);
    bindTexture(_arrow);  // 4J - was "/item/arrows.png"

    RenderPath.MatrixPush();

    float yRot = arrow->yRot;
    float xRot = arrow->xRot;
    float yRotO = arrow->yRotO;
    float xRotO = arrow->xRotO;
    if ((yRot - yRotO) > 180.0f)
        yRot -= 360.0f;
    else if ((yRot - yRotO) < -180.0f)
        yRot += 360.0f;
    if ((xRot - xRotO) > 180.0f)
        xRot -= 360.0f;
    else if ((xRot - xRotO) < -180.0f)
        xRot += 360.0f;

    RenderPath.MatrixTranslate((float)x, (float)y, (float)z);
    RenderPath.MatrixRotate((yRotO + (yRot - yRotO) * a - 90)*(std::numbers::pi_v<float>/180.f), 0, 1, 0);
    RenderPath.MatrixRotate((xRotO + (xRot - xRotO) * a)*(std::numbers::pi_v<float>/180.f), 0, 0, 1);

    plce::world::MeshBuilder mb(plce::world::MaterialKind::alpha_test, 0);
    int type = 0;

    float u0 = 0 / 32.0f;
    float u1 = 16 / 32.0f;
    float v0 = (0 + type * 10) / 32.0f;
    float v1 = (5 + type * 10) / 32.0f;

    float u02 = 0 / 32.0f;
    float u12 = 5 / 32.0f;
    float v02 = (5 + type * 10) / 32.0f;
    float v12 = (10 + type * 10) / 32.0f;
    float ss = 0.9f / 16.0f;
    float shake = arrow->shakeTime - a;
    if (shake > 0) {
        float pow = -sinf(shake * 3) * shake;
        RenderPath.MatrixRotate((pow)*(std::numbers::pi_v<float>/180.f), 0, 0, 1);
    }
    RenderPath.MatrixRotate((45)*(std::numbers::pi_v<float>/180.f), 1, 0, 0);
    RenderPath.MatrixScale(ss, ss, ss);

    RenderPath.MatrixTranslate(-4, 0, 0);

    // Each quad was its own Tesselator begin/end — a separate draw
    // under whatever matrix state was live at that point. Preserve
    // that per-quad flush so the rotations interleaved with the
    // draws below each land on the right draw.
    mb.normal(1, 0, 0);
    mb.vertexUV(-7, -2, -2, u02, v02);
    mb.vertexUV(-7, -2, +2, u12, v02);
    mb.vertexUV(-7, +2, +2, u12, v12);
    mb.vertexUV(-7, +2, -2, u02, v12);
    mb.flush();

    mb.normal(-1, 0, 0);
    mb.vertexUV(-7, +2, -2, u02, v02);
    mb.vertexUV(-7, +2, +2, u12, v02);
    mb.vertexUV(-7, -2, +2, u12, v12);
    mb.vertexUV(-7, -2, -2, u02, v12);
    mb.flush();

    for (int i = 0; i < 4; i++) {
        RenderPath.MatrixRotate((90)*(std::numbers::pi_v<float>/180.f), 1, 0, 0);
        mb.normal(0, 0, 1);
        mb.vertexUV(-8, -2, 0, u0, v0);
        mb.vertexUV(+8, -2, 0, u1, v0);
        mb.vertexUV(+8, +2, 0, u1, v1);
        mb.vertexUV(-8, +2, 0, u0, v1);
        mb.flush();
    }
    RenderPath.MatrixPop();
}

ResourceLocation* ArrowRenderer::getTextureLocation(
    std::shared_ptr<Entity> mob) {
    return &ARROW_LOCATION;
}