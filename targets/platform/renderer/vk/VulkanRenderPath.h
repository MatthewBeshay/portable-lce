#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "platform/renderer/IRenderPath.h"

struct SDL_Window;

class VulkanRenderPath final : public rp::IRenderPath {
public:
    explicit VulkanRenderPath(SDL_Window* window);
    ~VulkanRenderPath() override;

    VulkanRenderPath(const VulkanRenderPath&) = delete;
    VulkanRenderPath& operator=(const VulkanRenderPath&) = delete;

    // -- Lifecycle --
    void StartFrame() override;
    void Present() override;
    void Clear(int flags) override;
    void SetClearColour(const float rgba[4]) override;
    void render_frame(const rp::FrameDesc& frame) override;
    void resize(uint32_t w, uint32_t h) override;
    void GetFramebufferSize(int& w, int& h) override;
    void SetWindowSize(int w, int h) override;
    void SetFullscreen(bool fs) override;
    void Close() override;
    [[nodiscard]] bool ShouldClose() override;
    [[nodiscard]] const rp::FrameFramebuffer& framebuffer() const override;
    [[nodiscard]] bool IsWidescreen() override;
    [[nodiscard]] bool IsHiDef() override;

    // -- Resource handles (stubbed) --
    [[nodiscard]] rp::MeshHandle create_mesh(const rp::MeshDesc&) override {
        return rp::kInvalidMesh;
    }
    void update_mesh(rp::MeshHandle, const rp::MeshDesc&) override {}
    void destroy_mesh(rp::MeshHandle) override {}

    [[nodiscard]] rp::TextureHandle create_texture(
        const rp::TextureDesc&) override {
        return rp::kInvalidTexture;
    }
    void update_texture(rp::TextureHandle, const rp::TextureRegion&) override {}
    void destroy_texture(rp::TextureHandle) override {}

    [[nodiscard]] rp::MaterialHandle create_material(
        const rp::MaterialDesc&) override {
        return rp::kInvalidMaterial;
    }
    void update_material(rp::MaterialHandle, const rp::MaterialDesc&) override {
    }
    void destroy_material(rp::MaterialHandle) override {}

    [[nodiscard]] std::pair<rp::TransientVertexBuffer, std::span<std::byte>>
    alloc_transient_vertices(uint32_t, rp::VertexLayout,
                             rp::PrimitiveType) override {
        return {{}, {}};
    }

    void read_framebuffer(const rp::TextureReadback&) override {}
    [[nodiscard]] rp::ResourceFootprint query_resource_footprint()
        const override {
        return {};
    }
    void seal_static_resource_tier() override {}
    void begin_atomic_resource_batch() override {}
    void end_atomic_resource_batch() override {}
    void push_debug_event(const char*) override {}
    void pop_debug_event() override {}
    void tick() override {}

    // -- Legacy fixed-function (no-op stubs) --
    void MatrixMode(rp::MatrixStack) override {}
    void MatrixSetIdentity() override {}
    void MatrixTranslate(float, float, float) override {}
    void MatrixRotate(float, float, float, float) override {}
    void MatrixScale(float, float, float) override {}
    void MatrixPerspective(float, float, float, float) override {}
    void MatrixOrthogonal(float, float, float, float, float, float) override {}
    void MatrixPop() override {}
    void MatrixPush() override {}
    void MatrixMult(float*) override {}
    [[nodiscard]] const float* MatrixGet(rp::MatrixStack) override;

    void DrawVertices(int, int, void*, int, int) override {}

    [[nodiscard]] int CBuffCreate(int n) override {
        int first = next_cbuff_;
        next_cbuff_ += n > 0 ? n : 1;
        return first;
    }
    void CBuffDelete(int, int) override {}
    void CBuffDeleteAll() override {}
    void CBuffStart(int, bool) override {}
    void CBuffClear(int) override {}
    [[nodiscard]] int CBuffSize(int) override { return 0; }
    void CBuffEnd() override {}
    [[nodiscard]] bool CBuffCall(int, bool) override { return false; }
    void CBuffDeferredModeStart() override {}
    void CBuffDeferredModeEnd() override {}

    [[nodiscard]] int TextureCreate() override { return ++next_handle_; }
    void TextureFree(int) override {}
    void TextureBind(int) override {}
    void TextureBindVertex(int, bool) override {}
    void TextureSetTextureLevels(int) override {}
    void TextureData(int, int, void*, int, int) override {}
    void TextureDataUpdate(int, int, int, int, void*, int) override {}
    void TextureSetParam(int, int) override {}
    [[nodiscard]] int TextureGetTextureLevels() override { return 1; }

    void StateSetColour(float, float, float, float) override {}
    void StateSetDepthMask(bool) override {}
    void StateSetBlendEnable(bool) override {}
    void StateSetBlendFunc(rp::BlendFactor, rp::BlendFactor) override {}
    void StateSetBlendFactor(unsigned int) override {}
    void StateSetAlphaFunc(rp::AlphaTest, float) override {}
    void StateSetDepthFunc(rp::DepthTest) override {}
    void StateSetFaceCull(bool) override {}
    void StateSetLineWidth(float) override {}
    void StateSetWriteEnable(bool, bool, bool, bool) override {}
    void StateSetDepthTestEnable(bool) override {}
    void StateSetAlphaTestEnable(bool) override {}
    void StateSetDepthSlopeAndBias(float, float) override {}
    void StateSetFogEnable(bool) override {}
    void StateSetFogMode(rp::FogMode) override {}
    void StateSetFogNearDistance(float) override {}
    void StateSetFogFarDistance(float) override {}
    void StateSetFogDensity(float) override {}
    void StateSetFogColour(float, float, float) override {}
    void StateSetLightingEnable(bool) override {}
    void StateSetLightColour(int, float, float, float) override {}
    void StateSetLightAmbientColour(float, float, float) override {}
    void StateSetLightDirection(int, float, float, float) override {}
    void StateSetLightEnable(int, bool) override {}
    void StateSetViewport(int) override {}
    void StateSetEnableViewportClipPlanes(bool) override {}
    void StateSetStencil(int, uint8_t, uint8_t, uint8_t) override {}
    void StateSetForceLOD(int) override {}
    void StateSetTextureEnable(bool) override {}
    void StateSetActiveTexture(int) override {}
    void StateSetVertexTextureUV(float, float) override {}

    void SetChunkOffset(float, float, float) override {}

    void ReadPixels(int, int, int, int, void*) override {}
    [[nodiscard]] int LoadTextureData(const char* filename, void* srcInfo,
                                      int** dataOut) override;
    [[nodiscard]] int LoadTextureData(uint8_t* data, uint32_t bytes,
                                      void* srcInfo, int** dataOut) override;

    void Set_matrixDirty() override {}
    void CBuffLockStaticCreations() override {}
    void UpdateGamma(unsigned short) override {}
    void Suspend() override {}
    [[nodiscard]] bool Suspended() override { return false; }
    void Resume() override {}
    void BeginEvent(const char*) override {}
    void EndEvent() override {}
    void submit_immediate(const rp::DrawCall&) override {}

private:
    static constexpr uint32_t kFramesInFlight = 2;

    struct PerFrame {
        VkCommandPool       pool   = VK_NULL_HANDLE;
        VkCommandBuffer     cmd    = VK_NULL_HANDLE;
        VkSemaphore         image_acquired = VK_NULL_HANDLE;
        VkSemaphore         render_done    = VK_NULL_HANDLE;
        VkFence             in_flight      = VK_NULL_HANDLE;
    };

    void create_instance(bool enable_validation);
    void create_surface();
    void pick_physical_device();
    void create_device();
    void create_swapchain(uint32_t width, uint32_t height);
    void destroy_swapchain();
    void create_per_frame();
    void destroy_per_frame();

    SDL_Window*     window_   = nullptr;
    bool            should_close_ = false;

    VkInstance      instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    VkSurfaceKHR    surface_  = VK_NULL_HANDLE;
    VkPhysicalDevice phys_     = VK_NULL_HANDLE;
    VkDevice        device_   = VK_NULL_HANDLE;
    uint32_t        graphics_family_ = 0;
    VkQueue         graphics_queue_ = VK_NULL_HANDLE;

    VkSwapchainKHR  swapchain_ = VK_NULL_HANDLE;
    VkFormat        swapchain_format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D      swapchain_extent_ = {0, 0};
    std::vector<VkImage>     swapchain_images_;
    std::vector<VkImageView> swapchain_views_;

    std::array<PerFrame, kFramesInFlight> frames_{};
    uint32_t        frame_index_   = 0;
    uint32_t        acquired_image_ = 0;
    bool            frame_active_  = false;

    rp::FrameFramebuffer fb_{};
    std::array<float, 4> clear_color_{0.05f, 0.05f, 0.10f, 1.0f};
    std::array<float, 16> identity_matrix_{
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    int next_handle_ = 0;
    int next_cbuff_  = 1;
};
