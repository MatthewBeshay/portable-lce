#pragma once
#include <cstdint>
#include <string>
#include <vector>

class Graphics;

class BufferedImage {
private:
    // Per-mip pixel storage — 32-bit ARGB pixels packed as uint32_t, one
    // vector per mip level (level 0 = full resolution). An empty vector
    // means that mip level is not present.
    std::vector<uint32_t> data[10];
    int width;
    int height;
    void ByteFlip4(unsigned int& data);  // 4J added
public:
    static const int TYPE_INT_ARGB = 0;
    static const int TYPE_INT_RGB = 1;
    BufferedImage();  // empty image; fill with loadMipmapPng()
    BufferedImage(int width, int height, int type);
    BufferedImage(const std::string& File, bool filenameHasExtension = false,
                  bool bTitleUpdateTexture = false,
                  const std::string& drive = "");                  // 4J added
    BufferedImage(std::uint8_t* pbData, std::uint32_t dataBytes);  // 4J added
    ~BufferedImage();

    // Decode `numBytes` of PNG-encoded texture data into mipmap slot
    // `level`. The first call (level == 0) populates width/height.
    // Returns true on success.
    bool loadMipmapPng(int level, std::uint8_t* bytes, std::uint32_t numBytes);

    int getWidth();
    int getHeight();
    void getRGB(int startX, int startY, int w, int h, std::vector<int>& out,
                int offset, int scansize,
                int level = 0);  // 4J Added level param
    std::uint32_t* getData();          // 4J added — ARGB pixels, level 0
    std::uint32_t* getData(int level); // 4J added — ARGB pixels, arbitrary level
    Graphics* getGraphics();
    int getTransparency();
    BufferedImage* getSubimage(int x, int y, int w, int h);

    void preMultiplyAlpha();
};
