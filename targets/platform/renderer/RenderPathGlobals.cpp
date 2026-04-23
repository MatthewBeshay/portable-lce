#include <cassert>
#include <thread>
#include <vector>

#include "IRenderPath.h"

namespace rp::render_path_internal {

static IRenderPath* s_active = nullptr;

void set_active(IRenderPath* path) { s_active = path; }

IRenderPath& get_active() {
    assert(s_active && "RenderPath accessed before set_active()");
    return *s_active;
}

}  // namespace rp::render_path_internal

namespace rp::ui_overlay {

// Frame-scoped DrawCall buffer. Main-thread only — no synchronisation.
// Cleared at the top of each frame by the host loop, filled by migrated
// subsystems during their normal draw flow, then handed to the renderer
// via FrameDesc::ui_overlay before render_frame().
//
// s_main_thread is captured the first time push() is called and every
// subsequent push() asserts the caller is on the same thread. Without
// the guard, a future off-main-thread draw site would race the vector
// silently; an explicit assert surfaces it on the first hit.
static std::vector<DrawCall> s_overlay_draws;
static std::thread::id       s_main_thread{};

void push(const DrawCall& dc) {
    const auto tid = std::this_thread::get_id();
    if (s_main_thread == std::thread::id{}) s_main_thread = tid;
    assert(tid == s_main_thread &&
           "rp::ui_overlay::push called from a non-main thread");
    s_overlay_draws.push_back(dc);
}
void clear()                  { s_overlay_draws.clear(); }
std::span<const DrawCall> get() {
    return {s_overlay_draws.data(), s_overlay_draws.size()};
}

}  // namespace rp::ui_overlay

namespace rp::world_draws {

// Frame-scoped DrawCall / ChunkDrawCall buffers feeding ViewDesc's
// six world_/chunk_ spans. Main-thread only — same discipline as
// ui_overlay. The thread-id capture is shared with ui_overlay (first
// push() on either wins); world draws and UI draws always run on the
// same host thread, so piggybacking a single guard is enough.
static std::vector<DrawCall>      s_world_opaque;
static std::vector<DrawCall>      s_world_alpha_test;
static std::vector<DrawCall>      s_world_transparent;
static std::vector<ChunkDrawCall> s_chunk_opaque;
static std::vector<ChunkDrawCall> s_chunk_alpha_test;
static std::vector<ChunkDrawCall> s_chunk_transparent;
static std::thread::id            s_world_thread{};

namespace {
inline void check_thread() {
    const auto tid = std::this_thread::get_id();
    if (s_world_thread == std::thread::id{}) s_world_thread = tid;
    assert(tid == s_world_thread &&
           "rp::world_draws push called from a non-main thread");
}
}  // namespace

void push_opaque     (const DrawCall& dc) { check_thread(); s_world_opaque.push_back(dc); }
void push_alpha_test (const DrawCall& dc) { check_thread(); s_world_alpha_test.push_back(dc); }
void push_transparent(const DrawCall& dc) { check_thread(); s_world_transparent.push_back(dc); }
void push_chunk_opaque     (const ChunkDrawCall& c) { check_thread(); s_chunk_opaque.push_back(c); }
void push_chunk_alpha_test (const ChunkDrawCall& c) { check_thread(); s_chunk_alpha_test.push_back(c); }
void push_chunk_transparent(const ChunkDrawCall& c) { check_thread(); s_chunk_transparent.push_back(c); }

void clear() {
    s_world_opaque.clear();
    s_world_alpha_test.clear();
    s_world_transparent.clear();
    s_chunk_opaque.clear();
    s_chunk_alpha_test.clear();
    s_chunk_transparent.clear();
}

std::span<const DrawCall>      opaque()            { return {s_world_opaque.data(),      s_world_opaque.size()}; }
std::span<const DrawCall>      alpha_test()        { return {s_world_alpha_test.data(),  s_world_alpha_test.size()}; }
std::span<const DrawCall>      transparent()       { return {s_world_transparent.data(), s_world_transparent.size()}; }
std::span<const ChunkDrawCall> chunk_opaque()      { return {s_chunk_opaque.data(),      s_chunk_opaque.size()}; }
std::span<const ChunkDrawCall> chunk_alpha_test()  { return {s_chunk_alpha_test.data(),  s_chunk_alpha_test.size()}; }
std::span<const ChunkDrawCall> chunk_transparent() { return {s_chunk_transparent.data(), s_chunk_transparent.size()}; }

}  // namespace rp::world_draws
