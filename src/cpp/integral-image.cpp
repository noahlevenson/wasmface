#include <vector>
#include <cmath>

#include "integral-image.h"
#include "utility.h"
#include "haar-like.h"

/**
 * Constructor
 * @param {Float*} inputBuf Pointer to a buffer of input values in floating point ImageData pseudograyscale format
 * @param {Int}    w        Width of source image
 * @param {Int}    h        Height of source image
 * @param {Int}    size     Length of input buffer
 * @param {Bool}   squared  True produces an integral image derived from squared input values
 */
IntegralImage::IntegralImage(float inputBuf[], int w, int h, int size, bool squared) {
	this->data.resize(w, std::vector<float>(h, 0));
	std::vector<float> sumTable(size);
	for (int i = 3; i < size; i += 4) {
		auto vec2 = offsetToVec2(i - 3, w);
		int x = vec2[0];
		int y = vec2[1];
		float yP = y - 1 < 0 ? 0 : sumTable[i - w * 4];
		float xP = x - 1 < 0 ? 0 : this->data[x - 1][y];
		sumTable[i] = !squared ? yP + inputBuf[i] : yP + std::pow(inputBuf[i], 2);
		this->data[x][y] = xP + sumTable[i];	
	}
}

/**
 * Compute the sum of values within a rectangular region of an integral image
 * @param  {Int} x X offset for upper left corner
 * @param  {Int} y Y offset for upper left corner
 * @param  {Int} w Width of rectangle
 * @param  {Int} h Height of rectangle
 * @return {Float} Sum
 */
float IntegralImage::getRectangleSum(int x, int y, int w, int h) {
	float sum;
	if (x != 0 && y != 0) {
		float a = this->data[x - 1][y - 1];
		float b = this->data[x + w - 1][y - 1];
		float c = this->data[x + w - 1][y + h - 1];
		float d = this->data[x - 1][y + h - 1];
		sum = c + a - (b + d);
	} else if (x == 0 && y != 0) {
		float b = this->data[x + w - 1][y - 1];
		float c = this->data[x + w - 1][y + h - 1];
		sum = c - b;
	} else if (y == 0 && x != 0) {
		float c = this->data[x + w - 1][y + h - 1];
		float d = this->data[x - 1][y + h - 1];
		sum = c - d;
	} else {
		sum = this->data[x + w - 1][y + h - 1];
	}	
	return sum;
}

/**
 * Compute the value of a Haar-like feature over an integral image
 * @param  {Haarlike} h  The Haar-like feature to compute
 * @param  {Int}      sx Integral image x offset
 * @param  {Int}      sy Integral image y offset
 * @return {Float}       Feature value
 */
float IntegralImage::computeFeature(Haarlike& h, int sx, int sy) {
	float wSum, bSum;
	if (h.type == 1) {
		wSum = this->getRectangleSum(h.x + sx, h.y + sy, h.w, h.h);
		bSum = this->getRectangleSum(h.x + sx + h.w, h.y + sy, h.w, h.h);
	} else if (h.type == 2) {
		wSum = this->getRectangleSum(h.x + sx, h.y + sy, h.w, h.h) + 
			this->getRectangleSum(h.x + sx + h.w * 2, h.y + sy, h.w, h.h);
		bSum = this->getRectangleSum(h.x + sx + h.w, h.y + sy, h.w, h.h);
	} else if (h.type == 3) {
		wSum = this->getRectangleSum(h.x + sx, h.y + sy, h.w, h.h);
		bSum = this->getRectangleSum(h.x + sx, h.y + sy + h.h, h.w, h.h);
	} else if (h.type == 4) {
		wSum = this->getRectangleSum(h.x + sx, h.y + sy, h.w, h.h) + 
			this->getRectangleSum(h.x + sx, h.y + sy + h.h * 2, h.w, h.h);
		bSum = this->getRectangleSum(h.x + sx, h.y + sy + h.h, h.w, h.h);
	} else {
		wSum = this->getRectangleSum(h.x + sx, h.y + sy, h.w, h.h) + 
			this->getRectangleSum(h.x + sx + h.w, h.y + sy + h.h, h.w, h.h);
		bSum = this->getRectangleSum(h.x + sx + h.w, h.y + sy, h.w, h.h) + 
			this->getRectangleSum(h.x + sx, h.y + sy + h.h, h.w, h.h);
	}
	float f = bSum - wSum;
	return f;
}