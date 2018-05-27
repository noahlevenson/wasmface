#include <vector>

#include "cascade-classifier.h"
#include "integral-image.h"
#include "strong-classifier.h"

/**
 * Constructor
 * @param {Int} baseResolution Base resolution for the cascade classifer
 */
CascadeClassifier::CascadeClassifier(int baseResolution) {
	this->baseResolution = baseResolution;
}

/**
 * Constructor
 * @param {Int}                           baseResolution Base resolution for the cascade classifier
 * @param {std::vector<StrongClassifier>} sc             A set of strong classifiers to add as layers
 */
CascadeClassifier::CascadeClassifier(int baseResolution, std::vector<StrongClassifier> sc) {
	this->baseResolution = baseResolution;
	this->strongClassifiers = sc;
}

/**
 * Destructively scale a cascade classifier relative to its base resolution
 * @param {Float} factor The factor by which to scale
 */
void CascadeClassifier::scale(float factor) {
	this->baseResolution *= factor;
	for (int i = 0; i < this->strongClassifiers.size(); i += 1) this->strongClassifiers[i].scale(factor);
}

/**
 * Add a strong classifier as a layer to a cascade classifier
 * @param {StrongClassifier} sc The strong classifier to add
 */
void CascadeClassifier::add(StrongClassifier sc) {
	this->strongClassifiers.push_back(sc);
}

/**
 * Remove the most recently added strong classifier associated with a cascade classifier
 */
void CascadeClassifier::removeLast() {
	this->strongClassifiers.pop_back();
}

/**
 * Classify a region of an integral image
 * @param  {IntegralImage} integral The integral image to classify
 * @param  {Int}           sx       Subwindow x offset
 * @param  {Int}           sy       Subwindow y offset
 * @param  {Float}         mean     The mean of the values within the subwindow (for post-normalization)
 * @param  {Float}         sd       The standard deviation of the values within the subwindow (for post normalization)
 * @return {Bool}                   True for positive detection, false for negative
 */
bool CascadeClassifier::classify(IntegralImage& integral, int sx, int sy, float mean, float sd) {
	for (int i = 0; i < this->strongClassifiers.size(); i += 1) {
		if (this->strongClassifiers[i].classify(integral, sx, sy, mean, sd) == false) return false;
	}
	return true;
}

/**
 * Get false positive rate for a cascade classifier
 * @param  {std::vector<IntegralImage>} negativeValidationSet A set of negative images to test
 * @return {Float}                                            Normalized false positive rate
 */
float CascadeClassifier::getFPR(std::vector<IntegralImage>& negativeValidationSet) {
	int falsePositives = 0;
	for (int i = 0; i < negativeValidationSet.size(); i += 1) {
		if (this->classify(negativeValidationSet[i], 0, 0, 0, 1) == true) falsePositives += 1;
	}
	return (float)falsePositives / negativeValidationSet.size();
}

/**
 * Get false negative rate for a cascade classifier
 * @param  {std::vector<IntegralImage>} positiveValidationSet A set of positive images to test
 * @return {Float}                                            Normalized false negative rate
 */
float CascadeClassifier::getFNR(std::vector<IntegralImage>& positiveValidationSet) {
	int falseNegatives = 0;
	for (int i = 0; i < positiveValidationSet.size(); i += 1) {
		if (this->classify(positiveValidationSet[i], 0, 0, 0, 1) == false) falseNegatives += 1;
	}
	return (float)falseNegatives / positiveValidationSet.size();
}