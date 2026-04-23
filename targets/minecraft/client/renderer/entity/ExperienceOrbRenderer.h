#pragma once
#include "EntityRenderer.h"

class ResourceLocation;

class ExperienceOrbRenderer : public EntityRenderer {
private:
    static ResourceLocation XP_ORB_LOCATION;

public:
    ExperienceOrbRenderer();

    virtual void render(std::shared_ptr<Entity> _orb, double x, double y,
                        double z, float rot, float a);

    virtual ResourceLocation* getTextureLocation(std::shared_ptr<Entity> mob);
};