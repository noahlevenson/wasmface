#pragma once

#include <vector>
#include <string>
#include <experimental/filesystem>

#include "../lib/CImg.h"

#include "haar-like.h"
#include "integral-image.h"
#include "weak-classifier.h"
#include "cascade-classifier.h"
#include "strong-classifier.h"

void collectLocalExamples(const std::experimental::filesystem::path& path, std::vector<std::string>& destination);

void computeAllIntegrals(
						int baseResolution, 
						int negativeSetSize,
						std::vector<IntegralImage>& negativeSet, 
						std::vector<IntegralImage>& positiveSet, 
						std::vector<IntegralImage>& negativeValidationSet,
						std::vector<IntegralImage>& positiveValidationSet,
						std::vector<std::string>& negativeExamplePaths,
						std::vector<std::string>& positiveExamplePaths, 
						std::vector<std::string>& negativeValidationExamplePaths,
						std::vector<std::string>& positiveValidationExamplePaths
						);

IntegralImage makeNormalizedIntegralImage(unsigned char inputBuf[], int w, int h);

unsigned char* cimgToHTMLImageData(cimg_library::CImg<unsigned char>& image);

std::vector<Haarlike> generateFeatures(int s);

void rebuildSets(
				CascadeClassifier& cascadeClassifier, 
				std::vector<IntegralImage>& negativeSet, 
				std::vector<std::string>& negativeExamplePaths,
				int negativeSetSize
				);
std::string cascadeToJSON(CascadeClassifier& cascadeClassifier, float fpr, float fnr);

WeakClassifier findOptimalWCThreshold(
									  std::vector<WeakClassifier> potentialWeakClassifiers, 
									  std::vector<float> posWeights, 
									  std::vector<float> negWeights
									  );

StrongClassifier adaBoost(
						CascadeClassifier& cascadeClassifier, 
						std::vector<IntegralImage>& positiveSet,
						std::vector<IntegralImage>& negativeSet,
						std::vector<IntegralImage>& positiveValidationSet,
						std::vector<IntegralImage>& negativeValidationSet,
						std::vector<Haarlike>& featureSet,
						float targetMaxFPR, 
						float targetMaxFNR, 
						int maxFeatures);