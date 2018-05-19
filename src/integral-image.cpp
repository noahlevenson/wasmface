#include <vector>
#include <cmath>
#include <iostream>

#include "integral-image.h"
#include "utility.h"
#include "haar-like.h"


// Make normal
IntegralImage::IntegralImage(float inputBuf[], int w, int h, int size, bool squared) {
	this->data.resize(w, std::vector<float>(h, 0));
	// Table to cache cumulative column values
	std::vector<float> sumTable(size);
	for (int i = 3; i < size; i += 4) {
		auto vec2 = offsetToVec2(i - 3, w);
		int x = vec2[0];
		int y = vec2[1];
		// Value of our integral image at (x,y) equals
		// the accumulated sum of the previous pixels in the column above it 
		// plus the accumulated sum of the previous pixels to the left
		float yP = y - 1 < 0 ? 0 : sumTable[i - w * 4]; // northern neighbor
		float xP = x - 1 < 0 ? 0 : this->data[x - 1][y]; // left hand neighbor
		sumTable[i] = !squared ? yP + inputBuf[i] : yP + std::pow(inputBuf[i], 2);
		this->data[x][y] = xP + sumTable[i];	
	}
}

// TODO: Make this a switch statement
// Compute the sum of pixels within a rectangle of arbitrary size over this integral image 
// Rectangle corners are enumerated clockwise from top left: a, b, c, d
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

// TODO: Make this a switch statement
// Compute the value of a Haar-like feature over this integral image
// The 5 types correspond to the 5 types generated during training phase:
// type 1: 2 rectangles horizontally
// type 2: 3 rectangles horizontally
// type 3: 2 rectangles vertically
// type 4: 3 rectangles vertically
// type 5: 4 rectangles in a 2x2 grid
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