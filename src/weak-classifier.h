#pragma once

#include "haar-like.h"

class WeakClassifier {
	public:
		WeakClassifier(Haarlike haarlike, float f, bool label, float weight);
		WeakClassifier();
		int classify(float featureValue);
		void scale(float factor);
		Haarlike haarlike;
		bool label;
		float weight;
		float threshold;
		int polarity;
		float minErr;
};

bool comparePotentialWeakClassifiers(const WeakClassifier& a, const WeakClassifier& b);