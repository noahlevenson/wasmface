#include <algorithm>
#include <vector>

#include "strong-classifier.h"
#include "integral-image.h"
#include "weak-classifier.h"
#include "cascade-classifier.h"

/**
 * Constructor
 */
StrongClassifier::StrongClassifier() {
	this->threshold = 0;
}	

/**
 * Destructively scale a strong classifier relative to its base resolution
 * @param {Float} factor The factor by which to scale
 */
void StrongClassifier::scale(float factor) {
	for (int i = 0; i < this->weakClassifiers.size(); i += 1) {
		this->weakClassifiers[i].scale(factor);
	}
}

/**
 * Add a weak classifier to a strong classifier
 * @param {WeakClassifier} weakClassifier The weak classifier to add
 * @param {Float}          weight         The voting weight to associate with this weak classifier
 */
void StrongClassifier::add(WeakClassifier weakClassifier, float weight) {
	this->weakClassifiers.push_back(weakClassifier);
	this->weights.push_back(weight);
}

/**
 * Classify a region of an integral image
 * @param  {IntegralImage}  integral The integral image to classify
 * @param  {Int}            sx       Subwindow x offset
 * @param  {Int}            sy       Subwindow y offset
 * @param  {Float}          mean     The mean of the values within the subwindow (for post-normalization)
 * @param  {Float}          sd       The standard deviation for the values within the subwindow (for post normalization)
 * @return {Bool}                    True for positive detection, false for negative
 */
bool StrongClassifier::classify(IntegralImage& integral, int sx, int sy, float mean, float sd) {
	float score = 0;
	for (int i = 0; i < this->weakClassifiers.size(); i += 1) {
		float f = integral.computeFeature(this->weakClassifiers[i].haarlike, sx, sy);
		if (this->weakClassifiers[i].haarlike.type == 2) {
			f += (this->weakClassifiers[i].haarlike.w * 3 * this->weakClassifiers[i].haarlike.h * mean) / 3;
		} else if (this->weakClassifiers[i].haarlike.type == 4) {
			f += (this->weakClassifiers[i].haarlike.w * this->weakClassifiers[i].haarlike.h * 3 * mean) / 3;
		}
		if (sd != 0) f /= sd;
		score += this->weakClassifiers[i].classify(f) * this->weights[i];
	}

	if (score >= this->threshold) return true;
	else return false;
}

/**
 * Optimize a strong classifier's threshold for minimal false negative rate
 * @param {std::vector<IntegralImage>} positiveValidationSet A set of positive images to test               
 * @param {Float}                      maxFNR                Normalized target maximum false negative rate
 */
void StrongClassifier::optimizeThreshold(std::vector<IntegralImage>& positiveValidationSet, float maxFNR) {
	std::vector<float> scores(positiveValidationSet.size(), 0);
	for (int i = 0; i < positiveValidationSet.size(); i += 1) {
		for (int j = 0; j < this->weakClassifiers.size(); j += 1) {
			auto f = positiveValidationSet[i].computeFeature(this->weakClassifiers[j].haarlike, 0, 0);
			scores[i] += this->weakClassifiers[j].classify(f) * this->weights[j];
		}
	}

	std::sort(scores.begin(), scores.end());
	int idx = maxFNR * positiveValidationSet.size();
	float thresh = scores[idx];
	while (idx > 0 && scores[idx] == thresh) idx -= 1;	
	
	this->threshold = scores[idx]; 
}

/**
 * Get false positive rate for a strong classifier
 * @param  {std::vector<IntegralImage>} negativealidationSet A set of negative images to test
 * @return {Float}                                           Normalized false positive rate
 */
float StrongClassifier::getFPR(std::vector<IntegralImage>& negativeValidationSet) {
	int falsePositives = 0;
	for (int i = 0; i < negativeValidationSet.size(); i += 1) {
		if (this->classify(negativeValidationSet[i], 0, 0, 0, 1) == true) falsePositives += 1;
	}
	return (float)falsePositives / negativeValidationSet.size();  
}

/**
 * Get false negative rate for a strong classifier
 * @param  {std::vector<IntegralImage>} positiveValidationSet A set of positive images to test
 * @return {Float}                                            Normalized false negative rate
 */
float StrongClassifier::getFNR(std::vector<IntegralImage>& positiveValidationSet) {
	int falseNegatives = 0;
	for (int i = 0; i < positiveValidationSet.size(); i += 1) {
		if (this->classify(positiveValidationSet[i], 0, 0, 0, 1) == false) falseNegatives += 1;
	}
	return (float)falseNegatives / positiveValidationSet.size();
}