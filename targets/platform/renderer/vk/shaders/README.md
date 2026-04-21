# vk shaders

Source files:

- `basic.vert` / `basic.frag` — world/entity draw pair (32-byte vertex).
- `basic_compact.vert` — chunk draw vertex shader (16-byte compact vertex,
  decoded in-shader; pairs with `basic.frag`).

Each `.vert` / `.frag` is compiled to SPIR-V at build time by the
`plce_compile_shader` CMake function (see
`cmake/PlceHelpers.cmake`). The function shells out to
`glslangValidator` and emits a C header containing the binary as a
`static const uint32_t <name>[]`. The header lives under the build
tree, next to the source, e.g.
`build/.../vk/shaders/basic.frag.spv.h`.

Shader sources are embedded via the variable names in `Renderer.cpp`:

| GLSL source            | Variable name              |
| ---------------------- | -------------------------- |
| `basic.vert`           | `kBasicVertSpv`            |
| `basic.frag`           | `kBasicFragSpv`            |
| `basic_compact.vert`   | `kBasicCompactVertSpv`     |

## Regenerating manually

The CMake build handles compilation automatically whenever a source
file changes. To invoke `glslangValidator` directly (e.g. for SPIR-V
inspection outside the build), use:

```
glslangValidator -V --vn kBasicFragSpv \
    -o basic.frag.spv.h  \
    targets/platform/renderer/vk/shaders/basic.frag
```

`glslangValidator` ships with the Vulkan SDK. Set `VULKAN_SDK` in your
environment so CMake can find it.

## Bit-packed push-constant `flags`

`PushConstants::flags` is a 32-bit bitfield shared between the vertex
and fragment stages. The canonical layout lives in `basic.frag` as a
block of `#define` macros (`FLAG_TEXTURED`, `FLAG_ALPHA_TEST`,
`FLAG_LIGHTMAP`, `FLAG_FORCE_LOD`, `TEX_ID_SHIFT`, `LM_TEX_ID_SHIFT`,
`FORCE_LOD_SHIFT`). When extending that layout, keep the macros in
sync with `PushConstants.h` and `Renderer::fill_push_constants` — the
C++ side is the source of truth for the bit numbers.
