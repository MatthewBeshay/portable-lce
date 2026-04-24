#include "MemoryTracker.h"

#include <atomic>
#include <utility>
#include <vector>

#include "java/ByteBuffer.h"
#include "platform/renderer/renderer.h"
#include "platform/stubs.h"

std::unordered_map<int, int> MemoryTracker::GL_LIST_IDS;
std::vector<int> MemoryTracker::TEXTURE_IDS;

#ifdef PLCE_RENDERER_VULKAN
// Raw-vk has no CBuff backing store — ids are only used to keep the
// bgfx backend's id space consistent. Hand out a monotonic counter
// locally so the allocator doesn't drag the deprecated CBuffCreate
// hook into the build.
static std::atomic<int> s_next_list_id{1};
#endif

int MemoryTracker::genLists(int count) {
#ifdef PLCE_RENDERER_VULKAN
    int id = s_next_list_id.fetch_add(count > 0 ? count : 1,
                                      std::memory_order_relaxed);
#else
    int id = RenderPath.CBuffCreate(count);
#endif
    GL_LIST_IDS.insert(std::pair<int, int>(id, count));
    return id;
}

int MemoryTracker::genTextures() {
    int id = glGenTextures();
    TEXTURE_IDS.push_back(id);
    return id;
}

void MemoryTracker::releaseLists(int id) {
    auto it = GL_LIST_IDS.find(id);
    if (it != GL_LIST_IDS.end()) {
#ifndef PLCE_RENDERER_VULKAN
        RenderPath.CBuffDelete(id, it->second);
#endif
        GL_LIST_IDS.erase(it);
    }
}

void MemoryTracker::releaseTextures() {
    for (int i = 0; i < TEXTURE_IDS.size(); i++) {
        glDeleteTextures(TEXTURE_IDS.at(i));
    }
    TEXTURE_IDS.clear();
}

void MemoryTracker::release() {
#ifndef PLCE_RENDERER_VULKAN
    for (auto it = GL_LIST_IDS.begin(); it != GL_LIST_IDS.end(); ++it) {
        RenderPath.CBuffDelete(it->first, it->second);
    }
#endif
    GL_LIST_IDS.clear();

    releaseTextures();
}

ByteBuffer* MemoryTracker::createByteBuffer(int size) {
    // 4J - was ByteBuffer.allocateDirect(size).order(std::endian.nativeOrder())
    ByteBuffer* bb = ByteBuffer::allocate(size);
    return bb;
}

IntBuffer* MemoryTracker::createIntBuffer(int size) {
    return createByteBuffer(size << 2)->asIntBuffer();
}

FloatBuffer* MemoryTracker::createFloatBuffer(int size) {
    return createByteBuffer(size << 2)->asFloatBuffer();
}