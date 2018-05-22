#include <iostream>

#include "weak-classifier.h"
#include "haar-like.h"


WeakClassifier::WeakClassifier(Haarlike haarlike, float f, bool label, float weight) {
	this->haarlike = haarlike;
	this->threshold = f;
	this->label = label;
	this->weight = weight;
	this->minErr = 1;
	this->polarity = 0;
}

WeakClassifier::WeakClassifier() {

}

int WeakClassifier::classify(float featureValue) {
	if (featureValue * float(this->polarity) < this->threshold * float(this->polarity)) return 1;
	else return -1;
}

void WeakClassifier::scale(float factor) {
	this->threshold *= (factor * factor);
	this->haarlike.scale(factor);
}

bool comparePotentialWeakClassifiers(const WeakClassifier& a, const WeakClassifier& b) {
	return a.threshold < b.threshold;
}