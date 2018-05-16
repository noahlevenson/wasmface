#pragma once

#include <emscripten/emscripten.h>

#ifdef __cplusplus
extern "C" {
#endif

EMSCRIPTEN_KEEPALIVE bool isFace(unsigned char inputBuf[]);

#ifdef __cplusplus
}
#endif