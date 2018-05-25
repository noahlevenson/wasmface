#pragma once

#include <vector>

#include "integral-image.h"
#include "strong-classifier.h"

class StrongClassifier;

class CascadeClassifier {
	public:
		CascadeClassifier(int baseResolution);
		CascadeClassifier(int baseResolution, std::vector<StrongClassifier> sc);
		void scale(float factor);
		void add(StrongClassifier sc);
		void removeLast();
		bool classify(IntegralImage& integral, int sx, int sy, float mean, float sd);
		float getFPR(std::vector<IntegralImage>& negativeValidationSet);
		float getFNR(std::vector<IntegralImage>& positiveValidationSet);
		int baseResolution;
		std::vector<StrongClassifier> strongClassifiers;
};