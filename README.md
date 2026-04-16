# Portable LCE

<div align="center">

![](.github-assets/transrights.png) ![](.github-assets/progress.png) ![](.github-assets/freepalestine.gif) ![](.github-assets/internetarchive.gif) ![](.github-assets//ieget-an.gif) ![](.github-assets/minecraft.gif) ![](.github-assets/powered-llvm.gif)
![](.github-assets/opengl.gif) ![](.github-assets/sgi.gif) ![](.github-assets/not-binary.png) ![](.github-assets/adobe_getflash2.gif) ![](.github-assets/flash_get_20010813.gif) ![](.github-assets/SiliconValley_7479_English_imagens_get_flashplayer.gif) ![](.github-assets/problematic-media.gif) ![](.github-assets/seal.gif) ![](.github-assets/notepad-logo3.webp) ![](.github-assets/hrt-e2.gif) ![](.github-assets/4j.png)

</div>

---

This project is a heavily modified version of the Minecraft Console Legacy Edition codebase, aimed at porting old Minecraft (TU19/1.6.1) to different platforms and refactoring the codebase to improve organization and use modern C++ features.

## Status

|  | **Linux** | **Windows** | **macOS[^1]** |
| - | - | - | - |
| **app** | `desktop` | `desktop` | `desktop` |
| **ui** | `java`, `shiggy`[^2] | `java`, `shiggy`[^2] | `java` |
| **fs** | `std` | `std` | `std` |
| **renderer** | `gl` | `gl` | `gl` |
| **sound** | `miniaudio` | `miniaudio` | `miniaudio` |
| **input** | `sdl2` | `sdl2` | `sdl2` |
| **thread** | `std` | `std` | `std` |
| **game** | `stub` | `stub` | `stub` |
| **network** | `stub` | `stub` | `stub` |
| **storage** | `stub` | `stub` | `stub` |
| **profile** | `stub` | `stub` | `stub` |
| **leaderboard** | `stub` | `stub` | `stub` |

[^1]: `platform_renderer_gl` is unstable on this platform and known to segfault. Development is WIP.
[^2]: `-Dui_backend=shiggy` supports x86-64 architectures only.

> [!TIP]
>
> This table describes the current backend used for each game component on each platform. If a backend is `stub`, that means that the game uses a [stubbed implementation](https://en.wikipedia.org/wiki/Method_stub) and the feature is unsupported at the moment. In some cases (e.g. leaderboards and profile) it makes sense to use a stubbed implementation, since we don't have access to console services like Xbox live on desktop operating systems. In other cases, it is used temporarily while work is done to properly implement the feature (such as storage for world/DLC saving and loading).

These platforms are currently work-in-progress:
- **Android**: Game runs, but the port predates many refactors and therefore can't be easily upstreamed at the moment. `ui-backend=java` only.
- **Emscripten**: Works except for audio. Predates a major refactor, and requires a rebase. `ui-backend=java` only.

---

## Join our community:
* **Discord:** https://discord.gg/SC6WCZezry

## Building

The project builds with **CMake 3.28+** and uses **vcpkg** in manifest mode
for third-party dependencies.

### Prerequisites

- **CMake ≥ 3.28**
- **vcpkg** - clone the repo and set `VCPKG_ROOT` to its path
  (`$env:VCPKG_ROOT` on Windows, `export VCPKG_ROOT=…` on Linux/macOS)
- **Python 3** (used by asset pipeline scripts)
- A **C++23** toolchain:
  - Windows: Visual Studio 2022 17.10+ or VS 2026 (MSVC 19.40+)
  - Linux: GCC 15+ or Clang 18+ with libc++
  - macOS: Apple Clang from Xcode 16+

#### Platform-specific system libraries

Linux (Debian/Ubuntu):
```bash
sudo apt-get install -y build-essential ninja-build libsdl2-dev libgl-dev \
    libglu1-mesa-dev libpthread-stubs0-dev python3
```

Arch/Manjaro:
```bash
sudo pacman -S base-devel cmake ninja pkgconf sdl2-compat mesa glu python
```

Fedora/RHEL:
```bash
sudo dnf install gcc gcc-c++ make cmake ninja-build SDL2-devel \
    mesa-libGL-devel mesa-libGLU-devel openssl-devel python3
```

### Configure & build

`CMakePresets.json` provides presets for the common toolchains.

Windows (Visual Studio 2026 multi-config):
```pwsh
cmake --preset windows-vs
cmake --build --preset windows-vs-debug
# → build\windows-vs\targets\app\Debug\Minecraft.Client.exe
```

Windows (Ninja - launch from a VS Developer prompt so `cl.exe` is on PATH):
```pwsh
cmake --preset windows-ninja
cmake --build --preset windows-ninja-debug
```

Linux (GCC):
```bash
cmake --preset linux-gcc
cmake --build --preset linux-gcc-debug
# → build/linux-gcc/targets/app/Debug/Minecraft.Client
```

Linux (Clang + libc++):
```bash
cmake --preset linux-clang
cmake --build --preset linux-clang-debug
```

macOS:
```bash
cmake --preset macos-clang
cmake --build --preset macos-clang-debug
```

### Project options

Pass `-DPLCE_<OPTION>=<VALUE>` at configure time:

| Option | Values | Default | Notes |
|---|---|---|---|
| `PLCE_RENDERER` | `bgfx` \| `vulkan` | `bgfx` | `vulkan` enables the raw Vulkan 1.3 backend (WIP) |
| `PLCE_UI_BACKEND` | `shiggy` \| `java` | `shiggy` | `shiggy` is x86-64 only |
| `PLCE_ENABLE_VSYNC` | `ON` \| `OFF` | `ON` | |
| `PLCE_OCCLUSION_CULLING` | `off` \| `frustum` \| `bfs` \| `hardware` | `frustum` | |
| `PLCE_ENABLE_FRAME_PROFILER` | `ON` \| `OFF` | `OFF` | |
| `PLCE_ENABLE_MIMALLOC` | `ON` \| `OFF` | `OFF` | Requires the `mimalloc` vcpkg feature |

vcpkg features (add to `--x-feature=<name>` when configuring):

- `vulkan` - adds `volk`, `vulkan-memory-allocator`, `glslang`, `spirv-reflect`
- `tracy` - CPU+GPU profiler integration
- `mimalloc` - replaces malloc

### Clean rebuild

Delete the preset's build directory and reconfigure:

```bash
rm -rf build/<preset-name>
cmake --preset <preset-name>
```

---

## Running

Game assets are automatically copied next to the executable during the
build. Launch from that directory:

```pwsh
cd build\windows-vs\targets\app\Debug
./Minecraft.Client.exe
```

Set `BGFX_RENDERER` to pick a backend at runtime when `PLCE_RENDERER=bgfx`
(values: `gl`, `d3d11`, `d3d12`, `vulkan`).

<!-- ### View the online documentation [here](https://portable-lce.github.io/portable-lce). -->

## Generative AI Policy

Submitting code to this repository authored by generative AI tools (LLMs, agentic coding tools, etc...) is strictly forbidden (see [CONTRIBUTING.md](./CONTRIBUTING.md)). Pull requests that are clearly vibe-coded or written by an LLM will be closed. Contributors are expected to both fully understand the code that they write **and** have the necessary skills to *maintain it*.
