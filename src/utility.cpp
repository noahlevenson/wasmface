#include <vector>
#include <cmath>
#include <iostream>

#include "utility.h"

// Convert a raw offset in an HTML imagedata object to a 2D vector
// TODO: figure out where the w comes from
std::vector<int> offsetToVec2(int offset, int w) {
	std::vector<int> vec2(2);
	int pixelOffset = offset / 4;
	vec2[0] = pixelOffset % w;
	vec2[1] = pixelOffset / w;
	return vec2;
}

// Convert a set of RGB values to grayscale luma
// TODO: do we even use this?
unsigned char rgbToLuma(unsigned char r, unsigned char g, unsigned char b) {
	int sum = r * 0.2126 + g * 0.7152 + b * 0.0722;
	unsigned char luma = 255 - sum; // Inversion for HTML canvas
	return luma;
}

// Convert an HTML imagedata format buffer in-place to pseudograyscale
// color space (discard/ignore RGB and store luma in 4th byte)
unsigned char* toGrayscale(unsigned char inputBuf[], int w, int h) {
	int size = w * h * 4;
	for (int i = 0; i < size; i += 4) {
		int luma = inputBuf[i] * 0.2126 + inputBuf[i + 1] * 0.7152 + inputBuf[i + 2] * 0.0722;
		// Forgoing these three memory puts saves us cycles, but leaves dangerous garbage in our buffer
		inputBuf[i] = 0;
		inputBuf[i + 1] = 0;
		inputBuf[i + 2] = 0;
		inputBuf[i + 3] = 255 - luma;
	}
	return inputBuf;
}

// TODO: I don't like that this and imageDataToNormalizedBuffer return objects on the heap
// that must be memory managed - consider turning both into classes with useful destructors?
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

// TODO: Rename this function "normalizeImageData()"
// Convert an HTML imagedata format buffer to a buffer of normalized floating point values
float* imageDataToNormalizedBuffer(unsigned char inputBuf[], int w, int h) {
	int byteSize = w * h * 4;
	int size = w * h;

	int sum = 0;
	for (int i = 3; i < byteSize; i += 4) sum += inputBuf[i];

	float mean = float(sum) / float(size);

	float sd = 0;
	for (int i = 3; i < byteSize; i += 4) sd += std::pow(float(inputBuf[i]) - mean, 2);
	sd /= size;
	sd = std::sqrt(sd);
	if (sd == 0) sd = 1;

	// TODO: Should we collapse this (and all floating point intermediate representations) to 1 dimension?
	float* normalizedBuf = new float[byteSize];
	for (int i = 3; i < byteSize; i += 4) normalizedBuf[i] = (float(inputBuf[i]) - mean) / sd;
		
	return normalizedBuf;
}