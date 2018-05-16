#pragma once

#include <vector>

std::vector<int> offsetToVec2(int offset, int w);
unsigned char rgbToLuma(unsigned char r, unsigned char g, unsigned char b);
unsigned char* toGrayscale(unsigned char inputBuf[], int w, int h);
float* toGrayscaleFloat(unsigned char inputBuf[], int w, int h);
float* imageDataToNormalizedBuffer(unsigned char inputBuf[], int w, int h);