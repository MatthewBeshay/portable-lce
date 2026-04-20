#pragma once

#include <vulkan/vulkan.h>

#include <cstdio>
#include <stdexcept>

namespace plce::vk {

/// Check a VkResult and throw on failure.
inline void check(VkResult r, const char* msg) {
    if (r != VK_SUCCESS) {
        char buf[128];
        std::snprintf(buf, sizeof buf, "%s: VkResult=%d", msg, int(r));
        throw std::runtime_error(buf);
    }
}

}  // namespace plce::vk
