#pragma once

#include <vector>

#include "weak-classifier.h"
#include "integral-image.h"
#include "cascade-classifier.h"

class CascadeClassifier;

class StrongClassifier {
	public:
		StrongClassifier();
		void scale(float factor);
		void add(WeakClassifier weakClassifier, float weight);
		bool classify(IntegralImage& integral, int sx, int sy, float mean, float sd);
		void optimizeThreshold(std::vector<IntegralImage>& positiveValidationSet, float targetFNR);
		float getFPR(std::vector<IntegralImage>& negativeValidationSet);
		float getFNR(std::vector<IntegralImage>& positiveValidationSet);
		std::vector<WeakClassifier> weakClassifiers;
		std::vector<float> weights;
		float threshold;
};