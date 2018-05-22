#include <algorithm>
#include <vector>

#include "strong-classifier.h"
#include "integral-image.h"
#include "weak-classifier.h"
#include "cascade-classifier.h"

StrongClassifier::StrongClassifier() {
	this->threshold = 0;
}	

void StrongClassifier::scale(float factor) {
	for (int i = 0; i < this->weakClassifiers.size(); i += 1) {
		this->weakClassifiers[i].scale(factor);
	}
}

void StrongClassifier::add(WeakClassifier weakClassifier, float weight) {
	this->weakClassifiers.push_back(weakClassifier);
	this->weights.push_back(weight);
}

bool StrongClassifier::classify(IntegralImage& integral, int sx, int sy, float mean, float sd) {
	float score = 0;
	for (int i = 0; i < this->weakClassifiers.size(); i += 1) {
		auto f = integral.computeFeature(this->weakClassifiers[i].haarlike, sx, sy);
		if (this->weakClassifiers[i].haarlike.type == 2) {
			f += float(this->weakClassifiers[i].haarlike.w * 3) * float(this->weakClassifiers[i].haarlike.h) * mean / 3.0f;
		} else if (this->weakClassifiers[i].haarlike.type == 4) {
			f += float(this->weakClassifiers[i].haarlike.w) * float(this->weakClassifiers[i].haarlike.h  * 3) * mean / 3.0f;
		}
		if (sd != 0) f /= sd;
		score += this->weakClassifiers[i].classify(f) * this->weights[i];
	}

	if (score >= this->threshold) return true;
	else return false;
}

void StrongClassifier::optimizeThreshold(std::vector<IntegralImage>& positiveValidationSet, CascadeClassifier& cc, float maxFNR) {
	std::vector<float> scores(positiveValidationSet.size(), 0);
	for (int i = 0; i < positiveValidationSet.size(); i += 1) {
		for (int j = 0; j < this->weakClassifiers.size(); j += 1) {
			auto f = positiveValidationSet[i].computeFeature(this->weakClassifiers[j].haarlike, 0, 0);
			scores[i] += float(this->weakClassifiers[j].classify(f)) * float(this->weights[j]);
		}
	}

	std::sort(scores.begin(), scores.end());
	int idx = maxFNR * positiveValidationSet.size();
	float thresh = scores[idx];
	while (idx > 0 && scores[idx] == thresh) {
		idx -= 1;	
	}

	this->threshold = scores[idx]; 
}

float StrongClassifier::getFPR(int baseResolution, std::vector<IntegralImage>& negativeSet) {
	int falsePositives = 0;
	for (int i = 0; i < negativeSet.size(); i += 1) {
		if (this->classify(negativeSet[i], 0, 0, 0, 1) == true) falsePositives += 1;
	}
	return (float)falsePositives / (float)negativeSet.size();  
}

float StrongClassifier::getFNR(int baseResolution, std::vector<IntegralImage>& positiveValidationSet) {
	int falseNegatives = 0;
	for (int i = 0; i < positiveValidationSet.size(); i += 1) {
		if (this->classify(positiveValidationSet[i], 0, 0, 0, 1) == false) falseNegatives += 1;
	}
	return (float)falseNegatives / (float)positiveValidationSet.size();
}