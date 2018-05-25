#pragma once

#include <emscripten/emscripten.h>

#ifdef __cplusplus
extern "C" {
#endif

class CascadeClassifier;

bool compareDereferencedPtrs(int* a, int* b);
std::vector<std::array<int, 3>> nonMaxSuppression(std::vector<std::array<int, 3>>& boxes, float thresh = 0.3f, int nthresh = 0);
EMSCRIPTEN_KEEPALIVE CascadeClassifier* create(char model[]);
EMSCRIPTEN_KEEPALIVE uint16_t* detect(unsigned char inputBuf[], int w, int h, CascadeClassifier* cco, 
							    float step, float delta, bool pp, float othresh, int nthresh);

#ifdef __cplusplus
}
#endif