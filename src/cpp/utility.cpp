#include <vector>
#include <cmath>

#include "utility.h"

/**
 * Convert an HTML5 ImageData offset to a 2D vector
 * @param {Int} offset The offset
 * @param {Int} w      Width of the ImageData object
 */
std::vector<int> offsetToVec2(int offset, int w) {
	std::vector<int> vec2(2);
	int pixelOffset = offset / 4;
	vec2[0] = pixelOffset % w;
	vec2[1] = pixelOffset / w;
	return vec2;
}

/**
 * Convert RGB to luma
 * @param  {Unsigned char} r Red value
 * @param  {Unsigned char} g Green value
 * @param  {Unsigned char} b Blue value
 * @return {Unsigned char}   Luma value
 */
unsigned char rgbToLuma(unsigned char r, unsigned char g, unsigned char b) {
	unsigned char luma = r * 0.2126f + g * 0.7152f + b * 0.0722f;
	return luma;
}

/**
 * Convert an HTML5 ImageData buffer in-place to pseudograyscale format (discard RGB and store luma in 4th byte)
 * @param  {Unsigned char*} inputBuf Pointer to an ImageData buffer
 * @param  {Int}            w        Width of ImageData object
 * @param  {Int}            h        Height of ImageData object
 * @return {Unsigned char*}          Pointer to the original buffer
 */
unsigned char* toGrayscale(unsigned char inputBuf[], int w, int h) {
	int size = w * h * 4;
	for (int i = 0; i < size; i += 4) {
		int luma = inputBuf[i] * 0.2126f + inputBuf[i + 1] * 0.7152f + inputBuf[i + 2] * 0.0722f;
		inputBuf[i] = 0;
		inputBuf[i + 1] = 0;
		inputBuf[i + 2] = 0;
		inputBuf[i + 3] = 255 - luma;
	}
	return inputBuf;
}

/**
 * Convert an HTML5 ImageData buffer to floating point pseudograyscale format (discard RGB and store luma in 4th float)
 * @param  {Unsigned char*} inputBuf Pointer to an ImageData buffer
 * @param  {Int}            w        Width of ImageData object
 * @param  {Int}            h        Height of ImageData object
 * @return {Unsigned char*}          Pointer to a new buffer
 */
float* toGrayscaleFloat(unsigned char inputBuf[], int w, int h) {
	int size = w * h * 4;
	float* gs = new float[size];
	for (int i = 0; i < size; i += 4) {
		float r = float(inputBuf[i]) * 0.2126f;
		float g = float(inputBuf[i + 1]) * 0.7152f;
		float b = float(inputBuf[i + 2]) * 0.0722f;
		float luma = r + g + b;
		gs[i] = 0;
		gs[i + 1] = 0;
		gs[i + 2] = 0;
		gs[i + 3] = 255.0f - luma;
	}
	return gs;
}

/**
 * Apply variance normalization to a pseudograyscale HTML5 ImageData buffer 
 * @param  {Unsigned char*} inputBuf Pointer to an ImageData buffer
 * @param  {Int}            w        Width of ImageData object
 * @param  {Int}            h        Height of ImageData object
 * @return {Float*}                  Pointer to a new buffer
 */
float* imageDataToNormalizedBuffer(unsigned char inputBuf[], int w, int h) {
	int byteSize = w * h * 4;
	int size = w * h;
	
	int sum = 0;
	for (int i = 3; i < byteSize; i += 4) sum += inputBuf[i];

	float sd = 0;
	float mean = float(sum) / float(size);
	for (int i = 3; i < byteSize; i += 4) sd += std::pow(float(inputBuf[i]) - mean, 2);
	sd /= size;
	sd = std::sqrt(sd);
	if (sd == 0) sd = 1;

	float* normalizedBuf = new float[byteSize];
	for (int i = 3; i < byteSize; i += 4) normalizedBuf[i] = (float(inputBuf[i]) - mean) / sd;
	return normalizedBuf;
}