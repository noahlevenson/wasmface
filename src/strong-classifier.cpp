#include <algorithm>
#include <vector>
#include <iostream> 

#include "strong-classifier.h"
#include "integral-image.h"
#include "weak-classifier.h"
#include "cascade-classifier.h"

// Constructor
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
	// Experimental: Every time we add a weak classifier to a strong classifier, we update
	// the strong classifier's threshold value, which should be (0.5 * sum of weights aka alpha)
	// float sumOfWeights = 0;
	// for (int i = 0; i < this->weights.size(); i += 1) sumOfWeights += this->weights[i];
	// std::cout << "StrongClassifier::add says that current sum of weights is " << sumOfWeights << " and so threshold should be " << sumOfWeights / (float)2 << std::endl;
	// this->threshold = sumOfWeights / (float)2.0;
}

// TODO: Why do we take arguments for subwindow size and subwindow offsets? Why doesn't this
// match the arity of WeakClassifier::classify ? Is it for future frontend implementation and scaling etc?
bool StrongClassifier::classify(IntegralImage integral, int sx, int sy, float mean, float sd) {
	float score = 0;
	for (int i = 0; i < this->weakClassifiers.size(); i += 1) {
		auto f = integral.computeFeature(this->weakClassifiers[i].haarlike, sx, sy);

		// TODO: Should the below feature value modification be moved into the weakclassifier class?
		if (this->weakClassifiers[i].haarlike.type == 2) {
			f += float(this->weakClassifiers[i].haarlike.w * 3) * float(this->weakClassifiers[i].haarlike.h) * mean / 3.0f;
		}

		if (this->weakClassifiers[i].haarlike.type == 4) {
			f += float(this->weakClassifiers[i].haarlike.w) * float(this->weakClassifiers[i].haarlike.h  * 3) * mean / 3.0f;
		}

		if (sd != 0) f /= sd;

		// TODO: should we make weak classifier::classify return a float?
		score += float(this->weakClassifiers[i].classify(f)) * float(this->weights[i]);
	}

	if (score >= this->threshold) {
		return true;
	} else {
		return false;
	}
}

// TODO: Is this right?!  NO IT IS VERY BROKEN
// We should make strong classifiers more aware of the cascade classifier that they belong to
// rather than passing a cascade classifier as an argument... this is really gross
void StrongClassifier::optimizeThreshold(std::vector<IntegralImage>& positiveValidationSet, CascadeClassifier& cc, float maxFNR) {
	std::cout << "\nTrying to optimize strong classifier's threshold until the current cascade classifier achieves a max FNR of " << maxFNR << std::endl;
	std::vector<float> scores(positiveValidationSet.size(), 0);
	//std::vector<int> fValues;
	for (int i = 0; i < positiveValidationSet.size(); i += 1) {
		for (int j = 0; j < this->weakClassifiers.size(); j += 1) {
			auto f = positiveValidationSet[i].computeFeature(this->weakClassifiers[j].haarlike, 0, 0);
			// TODO: no need to cast here
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

	// this->temp.clear();
	// this->ftemp.clear();
	// auto tempFNR = this->getFNR(ct);   // temp, for debugging
	//std::sort(this->temp.begin(), this->temp.end());

	std::cout << "... OK, optimized strong classifier's threshold to " << this->threshold << std::endl;
}

// Get the rate of false positives for this strong classifier
// TODO: This should operate on a validation set?! Or at least be parameterized to specify
// what set of negative images you want to evaluate over?
float StrongClassifier::getFPR(int baseResolution, std::vector<IntegralImage>& negativeSet) {
	int falsePositives = 0;
	for (int i = 0; i < negativeSet.size(); i += 1) {
		if (this->classify(negativeSet[i], 0, 0, 0, 1) == true) falsePositives += 1;
	}
	return (float)falsePositives / (float)negativeSet.size();  
}

// Get the rate of false negatives for this strong classifier
// TODO: This should operate on a validation set?! Or at least be parameterized to specify
// what set of positive images you want to evaluate over?
float StrongClassifier::getFNR(int baseResolution, std::vector<IntegralImage>& positiveValidationSet) {
	int falseNegatives = 0;
	for (int i = 0; i < positiveValidationSet.size(); i += 1) {
		if (this->classify(positiveValidationSet[i], 0, 0, 0, 1) == false) falseNegatives += 1;
	}
	return (float)falseNegatives / (float)positiveValidationSet.size();
}