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

void getImagePaths(const std::experimental::filesystem::path& path, std::vector<std::string>& destination);
std::vector<IntegralImage> computeIntegrals(std::vector<std::string>& paths, int baseResolution);
std::vector<IntegralImage> computeIntegralsRand(std::vector<std::string>& paths, int baseResolution, int setSize);
std::vector<IntegralImage> computeIntegralsGrid(std::vector<std::string>& paths, int baseResolution);
unsigned char* cimgToHTMLImageData(cimg_library::CImg<unsigned char>& image);
std::vector<Haarlike> generateFeatures(int s);
std::string cascadeToJSON(CascadeClassifier& cascadeClassifier, float fpr, float fnr);
WeakClassifier optimizeWC(std::vector<WeakClassifier> potentialWCs, std::vector<float> posWeights, std::vector<float> negWeights);
StrongClassifier adaBoost(CascadeClassifier& cascadeClassifier, std::vector<IntegralImage>& positiveSet, 
                          std::vector<IntegralImage>& negativeSet, std::vector<IntegralImage>& positiveValidationSet, 
                          std::vector<IntegralImage>& negativeValidationSet, std::vector<Haarlike>& featureSet,
                          float targetMaxFPR, float targetMaxFNR, int maxFeatures);