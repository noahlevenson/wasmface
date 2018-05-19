#include <iostream>

#include "weak-classifier.h"
#include "haar-like.h"

// Constructor
WeakClassifier::WeakClassifier(Haarlike haarlike, float f, bool label, float weight) {
	this->haarlike = haarlike;
	this->f = f;
	this->label = label;
	this->weight = weight;
	this->minErr = 179;
	this->threshold = 179;
	this->polarity = 1;
}

// Default constructor - do we need this?  Maybe we need this to build an empty WC that we can populate
// during construction for detection
WeakClassifier::WeakClassifier() {

}

int WeakClassifier::classify(float featureValue) {
	if (featureValue * float(this->polarity) < this->threshold * float(this->polarity)) {
		return 1;
	} else {
		return -1;
	}
}

void WeakClassifier::scale(float factor) {
	this->threshold *= (factor * factor);
	this->haarlike.scale(factor);
}

bool comparePotentialWeakClassifiers(const WeakClassifier& a, const WeakClassifier& b) {
	return a.f < b.f;
}