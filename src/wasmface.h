#pragma once

#include <emscripten/emscripten.h>

#ifdef __cplusplus
extern "C" {
#endif

bool compareDereferencedPtrs(int* a, int* b);
std::vector<std::array<int, 3>> nonMaxSuppression(std::vector<std::array<int, 3>>& boxes, float thresh = 0.3f, int nthresh = 0);
EMSCRIPTEN_KEEPALIVE int detect(unsigned char inputBuf[], int w, int h);
EMSCRIPTEN_KEEPALIVE bool isFace(unsigned char inputBuf[]);

#ifdef __cplusplus
}
#endif