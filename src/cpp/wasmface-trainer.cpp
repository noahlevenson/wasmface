#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <experimental/filesystem>
#include <experimental/random>
#include <fstream>

#include "../../lib/CImg.h"
#include "../../lib/json.hpp"

#include "wasmface-trainer.h"
#include "utility.h"
#include "integral-image.h"
#include "weak-classifier.h"
#include "strong-classifier.h"

/**
 * Recursively scan a local directory for image files and store their paths
 * @param {std::experimental::filesystem::path} path        Initial path to local directory
 * @param {std::vector<std::string>}            destination Where to accumulate found image paths
 */
void getImagePaths(const std::experimental::filesystem::path& path, std::vector<std::string>& destination) {
	if (std::experimental::filesystem::is_directory(path)) {
		for (auto& childPath: std::experimental::filesystem::directory_iterator(path)) {
			getImagePaths(childPath.path(), destination);
		}	
	} else {
		if (path.extension() == ".jpg" || path.extension() == ".JPG" || path.extension() == ".ppm" || path.extension() == ".PPM") {
			std::cout << "\nFound file --> " << path.string() << std::endl;
				destination.push_back(path.string());	
		}
	}
}

/**
 * Compute variance normalized integral images for a set of local images
 * Assumes local images are 1:1 aspect ratio and any pixel dimensions
 * @param  {std::vector<std::string>}   paths          A set of paths to local image files
 * @param  {Int}                        baseResolution Images will be resized to baseResolution x baseResolution pixels before computation
 * @return {std::vector<IntegralImage>}                A set of integral images
 */
std::vector<IntegralImage> computeIntegrals(std::vector<std::string>& paths, int baseResolution) {
	int size = baseResolution * baseResolution * 4;
	std::vector<IntegralImage> finalIntegrals;
	for (int i = 0; i < paths.size(); i += 1) {
		std::cout << "computeIntegrals(): creating pre-normalized integral image for local image file " << i + 1 << "/" << paths.size() << std::endl;
		cimg_library::CImg<unsigned char> image(paths[i].c_str());
		image.resize(baseResolution, baseResolution);
		auto inputBuf = cimgToHTMLImageData(image);
		auto normalized = imageDataToNormalizedBuffer(inputBuf, baseResolution, baseResolution);
		IntegralImage integral(normalized, baseResolution, baseResolution, size, false);
		finalIntegrals.push_back(integral);
	 	delete [] inputBuf;
	 	delete [] normalized;
	}
	return finalIntegrals;
}

/**
 * Compute variance normalized integral images generated from randomized subwindows of a set of local images
 * Assumes local images are any aspect ratio and of pixel dimensions > baseResolution
 * @param  {std::vector<std::string>}   paths          A set of paths to local image files
 * @param  {Int}                        baseResolution Subwindows will be baseResolution x baseResolution pixels
 * @param  {Int}                        setSize        Number of integral images to generate
 * @return {std::vector<IntegralImage>}                A set of integral images
 */
std::vector<IntegralImage> computeIntegralsRand(std::vector<std::string>& paths, int baseResolution, int setSize) {
	int size = baseResolution * baseResolution * 4;
	std::vector<IntegralImage> finalIntegrals;
	int count = 0;
	while (finalIntegrals.size() < setSize) {
		for (int i = 0; i < paths.size() && finalIntegrals.size() < setSize; i += 1, count += 1) {
			std::cout << "computeIntegralsRand(): creating pre-normalized integral image " << count + 1 << "/" << setSize << std::endl;
			cimg_library::CImg<unsigned char> image(paths[i].c_str());
			int xCrop = std::experimental::randint(0, image.width() - baseResolution);
			int yCrop = std::experimental::randint(0, image.height() - baseResolution);
			image.crop(xCrop, yCrop, xCrop + baseResolution - 1, yCrop + baseResolution - 1);
			auto inputBuf = cimgToHTMLImageData(image);
			auto normalized = imageDataToNormalizedBuffer(inputBuf, baseResolution, baseResolution);
			IntegralImage integral(normalized, baseResolution, baseResolution, size, false);
			finalIntegrals.push_back(integral);
			delete [] inputBuf;
			delete [] normalized;
		}
	}
	return finalIntegrals;
}

/**
 * Compute variance normalized integral images generated from every non-overlapping subwindow of a set of local images
 * Assumes local images are any aspect ratio and of pixel dimensions > baseResolution
 * @param  {std::vector<std::string>}   paths          A set of paths to local image files
 * @param                               baseResolution Subwindows will be baseResolution x baseResolution pixels
 * @return {std::vector<IntegralImage>}                A set of integral images       
 */
std::vector<IntegralImage> computeIntegralsGrid(std::vector<std::string>& paths, int baseResolution) {
	int size = baseResolution * baseResolution * 4;
	std::vector<IntegralImage> finalIntegrals;
	int count = 0;
	for (int i = 0; i < paths.size(); i += 1) {
		cimg_library::CImg<unsigned char> image(paths[i].c_str());
		for (int h = 0; h < image.height() - baseResolution; h += baseResolution) {
			for (int w = 0; w < image.width() - baseResolution; w += baseResolution, count += 1) {
				std::cout << "computeIntegralsGrid(): creating pre-normalized integral image " << count << " from " << paths.size() << " local image files\n";
				auto crop = image.get_crop(w, h, w + baseResolution - 1, h + baseResolution - 1);
				auto inputBuf = cimgToHTMLImageData(crop);
				auto normalized = imageDataToNormalizedBuffer(inputBuf, baseResolution, baseResolution);
				IntegralImage integral(normalized, baseResolution, baseResolution, size, false);
				finalIntegrals.push_back(integral);
				delete [] inputBuf;
				delete [] normalized;
			}
		}
	}
	return finalIntegrals;
}

/**
 * Convert a CImg object to an HTML5 ImageData pseudograyscale buffer (discard RGB and store luma in 4th byte)
 * Assumes the color space of the input object is RGB or grayscale
 * @param  {cimg_library::CImg<unsigned char>} image The CImg object
 * @return {unsigned char*}                          Pointer to a new buffer
 */
unsigned char* cimgToHTMLImageData(cimg_library::CImg<unsigned char>& image) {
	unsigned char* imageData = new unsigned char[image.width() * image.height() * 4];
	for (int i = 0; i < image.height(); i += 1) {
		for (int j = 0; j < image.width(); j += 1) {
			int offset = i * image.width() * 4 + (j * 4);
			unsigned char luma;
			if (image._spectrum == 1) {
				luma = image(j, i);
			} else if (image._spectrum == 3) {
				unsigned char r = image(j, i, 0, 0);
				unsigned char g = image(j, i, 0, 1);
				unsigned char b = image(j, i, 0, 2);
				luma = rgbToLuma(r, g, b);
			}
			imageData[offset] = 0;              // R
			imageData[offset + 1] = 0;          // G
			imageData[offset + 2] = 0;          // B
			imageData[offset + 3] = 255 - luma; // A - note inversion for HTML5 canvas
		}
	}
	return imageData;
}

/**
 * Generate a complete set of Haar-like features (types 1, 2, 3, 4 & 5) as constrained by a given window
 * @param  {Int}                   s The window size
 * @return {std::vector<Haarlike>}   A set of Haar-like features
 */
std::vector<Haarlike> generateFeatures(int s) {
	std::vector<Haarlike> featureSet;
	for (int i = 0; i < s; i += 1) {
		for (int j = 0; j < s; j += 1) {
			for (int h = 1; i + h <= s; h += 1) {
				for (int w = 1; w * 2 + j <= s; w += 1) {
					featureSet.push_back(Haarlike(j, i, w, h, 1));
				}
				for (int w = 1; w * 3 + j <= s; w += 1) {
					featureSet.push_back(Haarlike(j, i, w, h, 2));
				}
			}
			for (int h = 1; i + h * 2 <= s; h += 1) {
				for (int w = 1; j + w <= s; w += 1) {
					featureSet.push_back(Haarlike(j, i, w, h, 3));
				}
				for (int w = 1; j + w * 2 <= s; w += 1) {
					featureSet.push_back(Haarlike(j, i, w, h, 5));
				}
			}
			for (int h = 1; i + h * 3 <= s; h += 1) {
				for (int w = 1; j + w <= s; w += 1) {
					featureSet.push_back(Haarlike(j, i, w, h, 4));
				}
			}	
		}
	}
	return featureSet;
}

/**
 * Serialize a cascade classifier object
 * @param  {CascadeClassifier} cascadeClassifier The cascade classifier to serialize
 * @param  {Float}             fpr               Arbitrary false positive rate to include
 * @param  {Float}             fnr               Arbitrary false negative rate to include
 * @return {std::string}                         A stringified JSON object
 */
std::string cascadeToJSON(CascadeClassifier& cascadeClassifier, float fpr, float fnr) {
	nlohmann::json ccJSON;
	ccJSON["strongClassifiers"] = nlohmann::json::array();
	ccJSON["baseResolution"] = cascadeClassifier.baseResolution;
	ccJSON["fpr"] = fpr;
	ccJSON["fnr"] = fnr;
	for (int i = 0; i < cascadeClassifier.strongClassifiers.size(); i += 1) {
		nlohmann::json scJSON;
		scJSON["weakClassifiers"] = nlohmann::json::array();
		scJSON["weights"] = nlohmann::json::array();
		scJSON["threshold"] = cascadeClassifier.strongClassifiers[i].threshold;
		for (int j = 0; j < cascadeClassifier.strongClassifiers[i].weakClassifiers.size(); j += 1) {
			nlohmann::json wcJSON;
			wcJSON["type"] = cascadeClassifier.strongClassifiers[i].weakClassifiers[j].haarlike.type;
			wcJSON["w"] = cascadeClassifier.strongClassifiers[i].weakClassifiers[j].haarlike.w;
			wcJSON["h"] = cascadeClassifier.strongClassifiers[i].weakClassifiers[j].haarlike.h;
			wcJSON["x"] = cascadeClassifier.strongClassifiers[i].weakClassifiers[j].haarlike.x;
			wcJSON["y"] = cascadeClassifier.strongClassifiers[i].weakClassifiers[j].haarlike.y;
			wcJSON["threshold"] = cascadeClassifier.strongClassifiers[i].weakClassifiers[j].threshold;
			wcJSON["polarity"] = cascadeClassifier.strongClassifiers[i].weakClassifiers[j].polarity;
			scJSON["weakClassifiers"].push_back(wcJSON);
			scJSON["weights"].push_back(cascadeClassifier.strongClassifiers[i].weights[j]);
		}
		ccJSON["strongClassifiers"].push_back(scJSON);
	}
	return ccJSON.dump();
}

/**
 * Evaluate a set of potential weak classifiers and select the threshold and polarity that minimizes classification error
 * @param  {std::vector<WeakClassifier>} potentialWCs A set of potential weak classifiers 
 * @param  {std::vector<float>}          posWeights   A set of positive weights as specified by AdaBoost
 * @param  {std::vector<float>}          negWeights   A set of negative weights as specified by AdaBoost
 * @return WeakClassifier                             The optimal weak classifier
 */
WeakClassifier optimizeWC(std::vector<WeakClassifier> potentialWCs, std::vector<float> posWeights, std::vector<float> negWeights) {
	std::sort(potentialWCs.begin(), potentialWCs.end(), comparePotentialWeakClassifiers);
	
	float sumPosWeights = 0;
	float sumNegWeights = 0;
	for (int j = 0; j < posWeights.size(); j += 1) sumPosWeights += posWeights[j];
	for (int j = 0; j < negWeights.size(); j += 1) sumNegWeights += negWeights[j];
	
	float sumPosWeightsBelow = 0;
	float sumNegWeightsBelow = 0;
	if (potentialWCs[0].label) sumPosWeightsBelow += potentialWCs[0].weight;
	else sumNegWeightsBelow += potentialWCs[0].weight;
	
	int bestWC;
	float minErr = 1;
	for (int j = 0; j < potentialWCs.size(); j += 1) {
		if (j > 0) {
			if (potentialWCs[j].label) sumPosWeightsBelow += potentialWCs[j].weight;
			else sumNegWeightsBelow += potentialWCs[j].weight;
		}

		float negErr = sumNegWeightsBelow + sumPosWeights - sumPosWeightsBelow;
		float posErr = sumPosWeightsBelow + sumNegWeights - sumNegWeightsBelow;
		
		if (posErr < negErr) {
			if (posErr < minErr) {
				minErr = posErr;
				potentialWCs[j].minErr = posErr;
				potentialWCs[j].polarity = -1;  
				bestWC = j;
			}
		} else {
			if (negErr < minErr) {
				minErr = negErr;
				potentialWCs[j].minErr = negErr;
				potentialWCs[j].polarity = 1;  
				bestWC = j;
 			}
		}
	}
	return potentialWCs[bestWC];
}

/**
 * Modified version of the AdaBoost learning algorithm
 * Builds a strong classifier from a weighted combination of the best weak classifiers
 * @param  {CascadeClassifier}          cascadeClassifier     The cascade classifier being trained
 * @param  {std::vector<IntegralImage>} positiveSet           A set of positive training images
 * @param  {std::vector<IntegralImage>} negativeSet           A set of negative training images
 * @param  {std::vector<IntegralImage>} positiveValidationSet A set of positive validation images
 * @param  {std::vector<IntegralImage>} negativeValidationSet A set of negative validation images
 * @param  {std::vector<Haarlike>}      featureSet            A set of Haar-like features
 * @param  {Float}                      targetMaxFPR          Normalized target max false positive rate
 * @param  {Float}                      targetMaxFNR          Normalized target max false negative rate
 * @param  {Int}                        maxFeatures           Maximum features for the strong classifier
 * @return {StrongClassifier}                                 A strong classifier
 */
StrongClassifier adaBoost(CascadeClassifier& cascadeClassifier, std::vector<IntegralImage>& positiveSet, 
                          std::vector<IntegralImage>& negativeSet, std::vector<IntegralImage>& positiveValidationSet, 
                          std::vector<IntegralImage>& negativeValidationSet, std::vector<Haarlike>& featureSet,
                          float targetMaxFPR, float targetMaxFNR, int maxFeatures) {
	StrongClassifier strongClassifier;
	
	// Initialize the weights
	std::vector<float> posWeights(positiveSet.size());
	std::vector<float> negWeights(negativeSet.size());
	std::fill(posWeights.begin(), posWeights.end(), 1.0f / 2.0f * positiveSet.size());
	std::fill(negWeights.begin(), negWeights.end(), 1.0f / 2.0f * negativeSet.size());

	float lastOverallFPR;

	if (cascadeClassifier.strongClassifiers.size() > 0) lastOverallFPR = cascadeClassifier.getFPR(negativeValidationSet);
	else lastOverallFPR = 1.0;
	
	float currentOverallFPR = lastOverallFPR;
	float currentOverallFNR = 0;  

	while (currentOverallFPR > targetMaxFPR * lastOverallFPR && strongClassifier.weakClassifiers.size() < maxFeatures) {
		std::cout << "\nSelecting new WC! Our current FPR is " << currentOverallFPR <<
			" and we'll stop adding WCs when we hit an FPR of " << targetMaxFPR * lastOverallFPR << std::endl;

		// Normalize the weights
		float wSum = 0;
		for (int j = 0; j < positiveSet.size(); j += 1) wSum += posWeights[j];
		for (int j = 0; j < negativeSet.size(); j += 1) wSum += negWeights[j];
		for (int j = 0; j < positiveSet.size(); j += 1) posWeights[j] /= wSum;
		for (int j = 0; j < negativeSet.size(); j += 1) negWeights[j] /= wSum;
		
		WeakClassifier bestWC;	
		float bestWCminError = 1.0;   	
		for (int i = 0; i < featureSet.size(); i += 1) {
			std::vector<WeakClassifier> potentialWCs;
			for (int j = 0; j < positiveSet.size(); j += 1) {
				auto f = positiveSet[j].computeFeature(featureSet[i], 0, 0);
				WeakClassifier wc(featureSet[i], f, true, posWeights[j]);
				potentialWCs.push_back(wc); 
			}
			for (int j = 0; j < negativeSet.size(); j += 1) {
				auto f = negativeSet[j].computeFeature(featureSet[i], 0, 0);
				WeakClassifier wc(featureSet[i], f, false, negWeights[j]);
				potentialWCs.push_back(wc);
			}
			
			auto wc = optimizeWC(potentialWCs, posWeights, negWeights);
			
			if (wc.minErr < bestWCminError) {
				bestWCminError = wc.minErr;
				bestWC = wc;
			}
		}
		
		// Update the weights
		float beta = bestWCminError / (1.0f - bestWCminError);

		int correctlyClassifiedPositives = 0;
		for (int j = 0; j < positiveSet.size(); j += 1) {
			if (bestWC.classify(positiveSet[j].computeFeature(bestWC.haarlike, 0, 0)) == 1) {
				posWeights[j] *= beta;
				correctlyClassifiedPositives += 1;
			}
		}

		int correctlyClassifiedNegatives = 0;
		for (int j = 0; j < negativeSet.size(); j += 1) {
			if (bestWC.classify(negativeSet[j].computeFeature(bestWC.haarlike, 0, 0)) == -1) {
				negWeights[j] *= beta;
				correctlyClassifiedNegatives += 1;
			} 
		}	

		std::cout << "\nThe new best WC correctly classified " << correctlyClassifiedPositives << "/" << 
			positiveSet.size() << " positives and " << correctlyClassifiedNegatives << "/" << 
				negativeSet.size() << " negatives\n";

		float alpha = std::log(1.0f / beta);
		strongClassifier.add(bestWC, alpha);
		strongClassifier.optimizeThreshold(positiveValidationSet, targetMaxFNR);
		cascadeClassifier.add(strongClassifier);
		currentOverallFPR = cascadeClassifier.getFPR(negativeValidationSet);
		currentOverallFNR = cascadeClassifier.getFNR(positiveValidationSet);
		cascadeClassifier.removeLast();
		
		std::cout << "\nSC after optimization produces a cascade classifier with an FPR of " <<
			currentOverallFPR << " and an FNR of " << currentOverallFNR << std::endl;
	}	
	return strongClassifier;
}

/**
 * Main function
 * By default we create a 30 layer cascade with sensible targets for FPR and max features per layer
 * TODO: parameterize number of layers, target max FPR for the cascade, target max FNR per layer,
 * [target FPR per layer], [max features per layer], output director for model and model filename
 * @param  {Int}   argc
 * @param  {Char*} argv
 * @return {Int}
 */
int main(int argc, char* argv[]) {
	if (argc < 13) {
		std::cout << "\nError: not enough parameters!\n";
		return 0;
	}

	float maxFNRPerLayer = 0.01f; 
	float targetMaxFPROverall = 0.00001f;

	int targetLayers = 30;
	
	std::vector<float> targetFPRs(30);
	targetFPRs[0] = 0.5f;
	targetFPRs[1] = 0.25f;
	for (int i = 2; i < targetFPRs.size(); i += 1) targetFPRs[i] = 0.615425314f;

	std::vector<int> featuresPerLayer = {
                                         5, 10, 30, 50, 50, 50, 100, 100, 100, 200,
                                         200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
                                         200, 200, 200, 200, 200, 200, 200, 200, 200, 200 
                                        };

	std::experimental::filesystem::path pathToPositives;
	std::experimental::filesystem::path pathToNegatives;
	std::experimental::filesystem::path pathToValidationNegatives;
	std::experimental::filesystem::path pathToValidationPositives;

	int baseResolution;
	int negativeSetSize;

	for (int i = 1; i < argc; i += 1) {
		if (std::strcmp(argv[i], "--p") == 0) {
			pathToPositives = std::experimental::filesystem::path(argv[i + 1]);
		} else if (std::strcmp(argv[i], "--n") == 0) {
			pathToNegatives = std::experimental::filesystem::path(argv[i + 1]);
		} else if (std::strcmp(argv[i], "--vp") == 0) {
			pathToValidationPositives = std::experimental::filesystem::path(argv[i + 1]);
		} else if (std::strcmp(argv[i], "--vn") == 0) {
			pathToValidationNegatives = std::experimental::filesystem::path(argv[i + 1]);
		} else if (std::strcmp(argv[i], "--s") == 0) {
			negativeSetSize = std::atoi(argv[i + 1]);
		} else if (std::strcmp(argv[i], "--b") == 0) {
			baseResolution = std::atoi(argv[i + 1]);
		} else {
			std::cout << "\nError: unknown argument '" << argv[i] << "'\n";
			return 0;
		}
		i += 1;
	}
	
	std::cout << "\nWasmface\n";
	std::cout << "Cascade classifier training\n";
	std::cout << "Performing recursive file scan for paths:\n";
	std::cout << "'" << pathToPositives.string() << "'\n";
	std::cout << "'" << pathToNegatives.string() << "'\n";
	std::cout << "'" << pathToValidationPositives.string() << "'\n";
	std::cout << "'" << pathToValidationNegatives.string() << "'\n";
	std::cout << "(I sure hope there are only .jpg or .ppm files in there)\n";

	std::vector<std::string> positiveExamplePaths, negativeExamplePaths, negativeValidationExamplePaths, positiveValidationExamplePaths;
	getImagePaths(pathToPositives, positiveExamplePaths);
	getImagePaths(pathToNegatives, negativeExamplePaths);
	getImagePaths(pathToValidationNegatives, negativeValidationExamplePaths);
	getImagePaths(pathToValidationPositives, positiveValidationExamplePaths);

	std::cout << "\nPositive training images found: " << positiveExamplePaths.size() << std::endl;
	std::cout << "Negative training images found: " << negativeExamplePaths.size() << std::endl;
	std::cout << "Positive validation images found: " << positiveValidationExamplePaths.size() << std::endl;
	std::cout << "Negative validation images found: " << negativeValidationExamplePaths.size() << std::endl;

	auto positiveSet = computeIntegrals(positiveExamplePaths, baseResolution);
	auto negativeSet = computeIntegralsRand(negativeExamplePaths, baseResolution, negativeSetSize);
	auto positiveValidationSet = computeIntegrals(positiveValidationExamplePaths, baseResolution);
	auto negativeValidationSet = computeIntegralsGrid(negativeValidationExamplePaths, baseResolution);

	std::cout << "\nDone!\n";
	std::cout << "\nGenerating feature set for base resolution " << baseResolution << "px...\n";
	auto featureSet = generateFeatures(24);
	std::cout << featureSet.size() << " features generated!\n";
	std::cout << "\nInitializing new cascade classifier composed of " << targetLayers << 
		" layers, with a target max false positive rate of " << targetMaxFPROverall << std::endl;

	CascadeClassifier cascadeClassifier(24);

	float currentOverallFPR = 1.0;
	float currentOverallFNR = 1.0;
	for (int i = 0; i < targetLayers; i += 1) {
		std::cout << "\nRunning AdaBoost to train a new SC...\n";
	
		auto sc = adaBoost(cascadeClassifier, 
						   positiveSet,
						   negativeSet,
						   positiveValidationSet,
						   negativeValidationSet,
						   featureSet,
						   targetFPRs[i], 
						   maxFNRPerLayer, 
						   featuresPerLayer[i]);
	
		cascadeClassifier.add(sc);

		currentOverallFPR = cascadeClassifier.getFPR(negativeValidationSet);
		currentOverallFNR = cascadeClassifier.getFNR(positiveValidationSet);

		std::ofstream modelFile;
		std::string filename = "../../models/my-model-" + std::to_string(i + 1) + "-layers.js";
		modelFile.open(filename, std::ios_base::trunc);
		std::string json = cascadeToJSON(cascadeClassifier, currentOverallFPR, currentOverallFNR);
		modelFile << "const wasmfaceModel = " << json;
		modelFile.close();
			
		std::cout << "\n --> Added a new SC with " << sc.weakClassifiers.size() << 
			" WCs to the CC! Current CC now has " << cascadeClassifier.strongClassifiers.size() << 
				" layers! FPR: " << currentOverallFPR << " FNR: " << currentOverallFNR << std::endl;

		// Remove images from the positive training set that the new strong classifier incorrectly classifies
		for (int i = 0; i < positiveSet.size(); i += 1) {
			if (!sc.classify(positiveSet[i], 0, 0, 0, 1)) positiveSet.erase(positiveSet.begin() + i);
		}

		// Remove images from the negative training set that the new strong classifier correctly classifies
		for (int i = 0; i < negativeSet.size(); i += 1) {
			if (!sc.classify(negativeSet[i], 0, 0, 0, 1)) negativeSet.erase(negativeSet.begin() + i);
		}
		
		// Crop subwindows from our negative images that cause false positives and add them to our negative
		// training set until we hit our target negative training set size
		int size = cascadeClassifier.baseResolution * cascadeClassifier.baseResolution * 4;
		int stepSize = 1;
		for (int i = 0; i < negativeExamplePaths.size() && negativeSet.size() < negativeSetSize; i += 1) {
			std::cout << "Rebuilding negative training set! Checking image " << i + 1 << "/" << negativeExamplePaths.size() << " for suitable subwindows...\n";
			cimg_library::CImg<unsigned char> image(negativeExamplePaths[i].c_str());
			for (int h = 0; h < image.height() - cascadeClassifier.baseResolution && negativeSet.size() < negativeSetSize; h += stepSize) {
				for (int w = 0; w < image.width() - cascadeClassifier.baseResolution && negativeSet.size() < negativeSetSize; w += stepSize) {
					auto crop = image.get_crop(w, h, w + cascadeClassifier.baseResolution - 1, h + cascadeClassifier.baseResolution - 1);
					auto inputBuf = cimgToHTMLImageData(crop);
					auto normalized = imageDataToNormalizedBuffer(inputBuf, cascadeClassifier.baseResolution, cascadeClassifier.baseResolution);
					IntegralImage integral(normalized, cascadeClassifier.baseResolution, cascadeClassifier.baseResolution, size, false);
					delete [] inputBuf;
					delete [] normalized;
					if (cascadeClassifier.classify(integral, 0, 0, 0, 1) == true) {
						negativeSet.push_back(integral);
						std::cout << "Currently have " << negativeSet.size() << " negative examples out of " << negativeSetSize << std::endl;
					}
				}
			}
		}

		if (negativeSet.size() < negativeSetSize) std::cout << "\n---COULD NOT REBUILD A COMPLETE NEGATIVE SET!---\n";
	}
	std::cout << "Successfully trained a cascade with an FPR of " << currentOverallFPR << std::endl;
	return 0;
}