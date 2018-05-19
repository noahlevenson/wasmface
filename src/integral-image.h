#pragma once

#include <vector>

#include "haar-like.h"

class IntegralImage {
	public:
		IntegralImage(float inputBuf[], int w, int h, int size, bool squared);
		float computeFeature(Haarlike& haarlike, int sx, int sy);
		std::vector<Haarlike> computeEntireFeatureSet(int s, int sx, int sy);
		float getRectangleSum(int x, int y, int w, int h);
		std::vector<std::vector<float>> data;
};