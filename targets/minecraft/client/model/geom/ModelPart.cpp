#include "ModelPart.h"

#include <numbers>

#include "Cube.h"
#include "TexOffs.h"
#include "minecraft/client/model/geom/Model.h"
#include "platform/renderer/renderer.h"
#include "platform/renderer/world/WorldDraw.h"
#include "platform/stubs.h"

const float ModelPart::RAD = (180.0f / std::numbers::pi);

void ModelPart::_init() {
    xTexSize = 64.0f;
    yTexSize = 32.0f;
    list = 0;
    compiled = false;
    bMirror = false;
    visible = true;
    neverRender = false;
    x = y = z = 0.0f;
    xRot = yRot = zRot = 0.0f;
    translateX = translateY = translateZ = 0.0f;
}

ModelPart::ModelPart() { _init(); }

ModelPart::ModelPart(Model* model, const std::string& id) {
    construct(model, id);
}

ModelPart::ModelPart(Model* model) { construct(model); }

ModelPart::ModelPart(Model* model, int xTexOffs, int yTexOffs) {
    construct(model, xTexOffs, yTexOffs);
}

void ModelPart::construct(Model* model, const std::string& id) {
    _init();
    this->model = model;
    model->cubes.push_back(this);
    this->id = id;
    setTexSize(model->texWidth, model->texHeight);
}

void ModelPart::construct(Model* model) {
    _init();
    construct(model, "");
}

void ModelPart::construct(Model* model, int xTexOffs, int yTexOffs) {
    _init();
    construct(model);
    texOffs(xTexOffs, yTexOffs);
}

void ModelPart::addChild(ModelPart* child) {
    // if (children == nullptr) children = new std::vector<ModelPart*>;
    children.push_back(child);
}

ModelPart* ModelPart::retrieveChild(SKIN_BOX* pBox) {
    for (auto it = children.begin(); it != children.end(); ++it) {
        ModelPart* child = *it;

        for (auto itcube = child->cubes.begin(); itcube != child->cubes.end();
             ++itcube) {
            Cube* pCube = *itcube;

            if ((pCube->x0 == pBox->fX) && (pCube->y0 == pBox->fY) &&
                (pCube->z0 == pBox->fZ) &&
                (pCube->x1 == (pBox->fX + pBox->fW)) &&
                (pCube->y1 == (pBox->fY + pBox->fH)) &&
                (pCube->z1 == (pBox->fZ + pBox->fD))) {
                return child;
                break;
            }
        }
    }

    return nullptr;
}

ModelPart* ModelPart::mirror() {
    bMirror = !bMirror;
    return this;
}

ModelPart* ModelPart::texOffs(int xTexOffs, int yTexOffs) {
    this->xTexOffs = xTexOffs;
    this->yTexOffs = yTexOffs;
    return this;
}

ModelPart* ModelPart::addBox(std::string id, float x0, float y0, float z0,
                             int w, int h, int d) {
    id = this->id + "." + id;
    TexOffs* offs = model->getMapTex(id);
    texOffs(offs->x, offs->y);
    cubes.push_back((new Cube(this, xTexOffs, yTexOffs, x0, y0, z0, w, h, d, 0))
                        ->setId(id));
    return this;
}

ModelPart* ModelPart::addBox(float x0, float y0, float z0, int w, int h,
                             int d) {
    cubes.push_back(new Cube(this, xTexOffs, yTexOffs, x0, y0, z0, w, h, d, 0));
    return this;
}

void ModelPart::addHumanoidBox(float x0, float y0, float z0, int w, int h,
                               int d, float g) {
    cubes.push_back(
        new Cube(this, xTexOffs, yTexOffs, x0, y0, z0, w, h, d, g, 63, true));
}

ModelPart* ModelPart::addBoxWithMask(float x0, float y0, float z0, int w, int h,
                                     int d, int faceMask) {
    cubes.push_back(
        new Cube(this, xTexOffs, yTexOffs, x0, y0, z0, w, h, d, 0, faceMask));
    return this;
}

void ModelPart::addBox(float x0, float y0, float z0, int w, int h, int d,
                       float g) {
    cubes.push_back(new Cube(this, xTexOffs, yTexOffs, x0, y0, z0, w, h, d, g));
}

void ModelPart::addTexBox(float x0, float y0, float z0, int w, int h, int d,
                          int tex) {
    cubes.push_back(
        new Cube(this, xTexOffs, yTexOffs, x0, y0, z0, w, h, d, (float)tex));
}

void ModelPart::setPos(float x, float y, float z) {
    this->x = x;
    this->y = y;
    this->z = z;
}

namespace {
// Helper: push every cube in `cubes` into one MeshBuilder and flush,
// so the whole part emits a single DrawCall per matrix-stack snapshot.
// The material defaults to alpha_test (entity atlases have transparent
// pixels around the body parts); entity renderers that want an opaque
// or blended material set it explicitly before calling Model::render.
void flush_cubes(const std::vector<Cube*>& cubes, float scale) {
    if (cubes.empty()) return;
    plce::world::MeshBuilder mb(plce::world::MaterialKind::alpha_test, 0);
    for (Cube* c : cubes) c->render(mb, scale);
    mb.flush();
}
}  // namespace

void ModelPart::render(float scale, bool /*usecompiled*/,
                       bool bHideParentBodyPart) {
    if (neverRender) return;
    if (!visible) return;

    RenderPath.MatrixTranslate(translateX, translateY, translateZ);

    if (xRot != 0 || yRot != 0 || zRot != 0) {
        RenderPath.MatrixPush();
        RenderPath.MatrixTranslate(x * scale, y * scale, z * scale);
        if (zRot != 0) RenderPath.MatrixRotate((zRot * RAD)*(std::numbers::pi_v<float>/180.f), 0, 0, 1);
        if (yRot != 0) RenderPath.MatrixRotate((yRot * RAD)*(std::numbers::pi_v<float>/180.f), 0, 1, 0);
        if (xRot != 0) RenderPath.MatrixRotate((xRot * RAD)*(std::numbers::pi_v<float>/180.f), 1, 0, 0);

        if (!bHideParentBodyPart) flush_cubes(cubes, scale);
        for (unsigned int i = 0; i < children.size(); i++) {
            children.at(i)->render(scale, false);
        }

        RenderPath.MatrixPop();
    } else if (x != 0 || y != 0 || z != 0) {
        RenderPath.MatrixTranslate(x * scale, y * scale, z * scale);
        if (!bHideParentBodyPart) flush_cubes(cubes, scale);
        for (unsigned int i = 0; i < children.size(); i++) {
            children.at(i)->render(scale, false);
        }
        RenderPath.MatrixTranslate(-x * scale, -y * scale, -z * scale);
    } else {
        if (!bHideParentBodyPart) flush_cubes(cubes, scale);
        for (unsigned int i = 0; i < children.size(); i++) {
            children.at(i)->render(scale, false);
        }
    }

    RenderPath.MatrixTranslate(-translateX, -translateY, -translateZ);
}

void ModelPart::renderRollable(float scale, bool /*usecompiled*/) {
    if (neverRender) return;
    if (!visible) return;

    RenderPath.MatrixPush();
    RenderPath.MatrixTranslate(x * scale, y * scale, z * scale);
    if (yRot != 0) RenderPath.MatrixRotate((yRot * RAD)*(std::numbers::pi_v<float>/180.f), 0, 1, 0);
    if (xRot != 0) RenderPath.MatrixRotate((xRot * RAD)*(std::numbers::pi_v<float>/180.f), 1, 0, 0);
    if (zRot != 0) RenderPath.MatrixRotate((zRot * RAD)*(std::numbers::pi_v<float>/180.f), 0, 0, 1);
    flush_cubes(cubes, scale);
    RenderPath.MatrixPop();
}

void ModelPart::translateTo(float scale) {
    if (neverRender) return;
    if (!visible) return;
    if (!compiled) compile(scale);

    if (xRot != 0 || yRot != 0 || zRot != 0) {
        RenderPath.MatrixTranslate(x * scale, y * scale, z * scale);
        if (zRot != 0) RenderPath.MatrixRotate((zRot * RAD)*(std::numbers::pi_v<float>/180.f), 0, 0, 1);
        if (yRot != 0) RenderPath.MatrixRotate((yRot * RAD)*(std::numbers::pi_v<float>/180.f), 0, 1, 0);
        if (xRot != 0) RenderPath.MatrixRotate((xRot * RAD)*(std::numbers::pi_v<float>/180.f), 1, 0, 0);
    } else if (x != 0 || y != 0 || z != 0) {
        RenderPath.MatrixTranslate(x * scale, y * scale, z * scale);
    } else {
    }
}

void ModelPart::compile(float /*scale*/) {
    // No-op under the MeshBuilder path — render() now emits one DrawCall
    // per frame straight from Cube::render / Polygon::render, so there's
    // no precompiled CBuff to prepare. Signature stays so Model
    // subclasses can keep calling part->compile(...) without touching
    // each one.
    compiled = true;
}

ModelPart* ModelPart::setTexSize(int xs, int ys) {
    this->xTexSize = (float)xs;
    this->yTexSize = (float)ys;
    return this;
}

void ModelPart::mimic(ModelPart* o) {
    x = o->x;
    y = o->y;
    z = o->z;
    xRot = o->xRot;
    yRot = o->yRot;
    zRot = o->zRot;
}
