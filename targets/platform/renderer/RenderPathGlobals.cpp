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
