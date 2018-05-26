#include "weak-classifier.h"
#include "haar-like.h"

/**
 * Constructor
 * @param {Haarlike} haarlike A Haar-like feature
 * @param {Float}    f        The computed feature value
 * @param {Bool}     label    True/false corresponds to positive/negative classification
 * @param {Float}    weight   The voting weight associated with this weak classifier
 */
WeakClassifier::WeakClassifier(Haarlike haarlike, float f, bool label, float weight) {
	this->haarlike = haarlike;
	this->threshold = f;
	this->label = label;
	this->weight = weight;
	this->minErr = 1;
	this->polarity = 0;
}

/**
 * Constructor
 */
WeakClassifier::WeakClassifier() {

}

/**
 * Classify a feature value
 * @param  {Float} featureValue The feature value
 * @return {Int}                1 is a positive classification, -1 is negative
 */
int WeakClassifier::classify(float featureValue) {
	if (featureValue * float(this->polarity) < this->threshold * float(this->polarity)) return 1;
	else return -1;
}

/**
 * Scale a weak classifier relative to its base resolution
 * @param {Float} factor The factor by which to scale
 */
void WeakClassifier::scale(float factor) {
	this->threshold *= (factor * factor);
	this->haarlike.scale(factor);
}

/**
 * Compare weak classifiers based on their threshold value
 * @param  {WeakClassifier} a First weak classifier
 * @param  {WeakClassifier} b Second weak classifier
 * @return {Bool}             True if the first weak classifier's threshold is the smaller of the two
 */
bool comparePotentialWeakClassifiers(const WeakClassifier& a, const WeakClassifier& b) {
	return a.threshold < b.threshold;
}