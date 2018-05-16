#include <vector>
#include <iostream>

#include "cascade-classifier.h"
#include "integral-image.h"
#include "strong-classifier.h"

CascadeClassifier::CascadeClassifier(int baseResolution) {
	this->baseResolution = baseResolution;
}

CascadeClassifier::CascadeClassifier(int baseResolution, std::vector<StrongClassifier> sc) {
	this->baseResolution = baseResolution;
	this->strongClassifiers = sc;
}

void CascadeClassifier::scale(float factor) {
	for (int i = 0; i < this->strongClassifiers.size(); i += 1) {
		this->strongClassifiers[i].scale(factor);
	} 
}

void CascadeClassifier::add(StrongClassifier sc) {
	this->strongClassifiers.push_back(sc);
}

void CascadeClassifier::removeLast() {
	this->strongClassifiers.pop_back();
}

bool CascadeClassifier::classify(IntegralImage integral, int sx, int sy, float mean, float sd) {
	for (int i = 0; i < this->strongClassifiers.size(); i += 1) {
		if (this->strongClassifiers[i].classify(integral, sx, sy, mean, sd) == false) {
			std::cout << "cascade classifier's " << i << "th strong classifier classified false\n";
			return false;
		}
	}
	return true;
}

float CascadeClassifier::getFPR(std::vector<IntegralImage>& negativeValidationSet) {
	int falsePositives = 0;
	for (int i = 0; i < negativeValidationSet.size(); i += 1) {
		if (this->classify(negativeValidationSet[i], 0, 0, 0, 1) == true) falsePositives += 1;
	}
	std::cout << "*** I just found " << falsePositives << " false positives in the negative validation set out of " << negativeValidationSet.size() << std::endl;
	return (float)falsePositives / (float)negativeValidationSet.size();
}

float CascadeClassifier::getFNR(std::vector<IntegralImage>& positiveValidationSet) {
	int falseNegatives = 0;
	for (int i = 0; i < positiveValidationSet.size(); i += 1) {
		if (this->classify(positiveValidationSet[i], 0, 0, 0, 1) == false) falseNegatives += 1;
	}
	std::cout << "*** I just found " << falseNegatives << " false negatives in the positive validation set out of " << positiveValidationSet.size() << std::endl;
	return (float)falseNegatives / (float)positiveValidationSet.size();
}
