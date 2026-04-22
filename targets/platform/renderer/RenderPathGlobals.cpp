#include <cassert>
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
static std::vector<DrawCall> s_overlay_draws;

void push(const DrawCall& dc) { s_overlay_draws.push_back(dc); }
void clear()                  { s_overlay_draws.clear(); }
std::span<const DrawCall> get() {
    return {s_overlay_draws.data(), s_overlay_draws.size()};
}

}  // namespace rp::ui_overlay
