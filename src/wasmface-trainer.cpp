// g++ wf-train-cascade.cpp utility.cpp integral-image.cpp haar-like.cpp weak-classifier.cpp strong-classifier.cpp cascade-classifier.cpp -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -std=c++17 "-lstdc++fs" -o ../bin/wf-train-cascade

// g++ -g wf-train-cascade.cpp utility.cpp integral-image.cpp haar-like.cpp weak-classifier.cpp strong-classifier.cpp cascade-classifier.cpp -O0 -L/usr/X11R6/lib -lm -lpthread -lX11 -std=c++17 "-lstdc++fs" -o ../bin/wf-train-cascade

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <experimental/filesystem>
#include <experimental/random>
#include <iomanip>
#include <fstream>

#include "../lib/CImg.h"
#include "../lib/json.hpp"

#include "wasmface-trainer.h"
#include "utility.h"
#include "integral-image.h"
#include "weak-classifier.h"
#include "strong-classifier.h"

// Recursively scan a local directory for JPG files and store their paths
void collectLocalExamples(const std::experimental::filesystem::path& path, std::vector<std::string>& destination) {
	if (std::experimental::filesystem::is_directory(path)) {
		for (auto& childPath: std::experimental::filesystem::directory_iterator(path)) {
			collectLocalExamples(childPath.path(), destination);
		}	
	} else {
		if (path.extension() == ".jpg" || path.extension() == ".JPG" || path.extension() == ".ppm") {
			std::cout << "\nFound file --> " << path.string() << std::endl;
				destination.push_back(path.string());	
		}
	}
}

// This is awful - we need one generalized function for this task that can be called several times from main()
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
						) {
	int size = baseResolution * baseResolution * 4;
	// Precompute and cache all integrals for positive example images
	for (int i = 0; i < positiveExamplePaths.size(); i += 1) {
		std::cout << "Caching integral image for positive example " << i << "/" << positiveExamplePaths.size() - 1 << std::endl;
		cimg_library::CImg<unsigned char> image(positiveExamplePaths[i].c_str());

		// EXPERIMENTAL: CROP IN CLOSER ON LFWA 
		//image.crop(39, 59, 189, 209);
		image.resize(baseResolution, baseResolution);
		auto inputBuf = cimgToHTMLImageData(image);
		auto normalized = imageDataToNormalizedBuffer(inputBuf, baseResolution, baseResolution);
		IntegralImage integral(normalized, baseResolution, baseResolution, size, false);
		positiveSet.push_back(integral);
	 	delete [] inputBuf;
	 	delete [] normalized;
	}

	// Precompute and cache all integrals for the positive validation set
	for (int i = 0; i < positiveValidationExamplePaths.size(); i += 1) {
		std::cout << "Caching integral image for positive validation set example " << i << "/" << positiveValidationExamplePaths.size() - 1 << std::endl;
		cimg_library::CImg<unsigned char> image(positiveValidationExamplePaths[i].c_str());

		// EXPERIMENTAL: CROP IN CLOSER ON LFWA 
		//image.crop(39, 59, 189, 209);
		image.resize(baseResolution, baseResolution);

		auto inputBuf = cimgToHTMLImageData(image);
		auto normalized = imageDataToNormalizedBuffer(inputBuf, baseResolution, baseResolution);
		IntegralImage integral(normalized, baseResolution, baseResolution, size, false);
		positiveValidationSet.push_back(integral);
	 	delete [] inputBuf;
	 	delete [] normalized;
	}


	// Precompute and cache all integrals for negative example images
	// TODO: We can manually specify the size of our negative set per step if we want
	// The move: just load a bunch of random subwindows
	int count = 0;
	while (negativeSet.size() < negativeSetSize) {
		for (int i = 0; i < negativeExamplePaths.size() && negativeSet.size() < negativeSetSize; i += 1) {
			cimg_library::CImg<unsigned char> image(negativeExamplePaths[i].c_str());
			// Experimental: get random x/y position to anchor the crop
			int xCrop = std::experimental::randint(0, image.width() - 24);
			int yCrop = std::experimental::randint(0, image.height() - 24);
			image.crop(xCrop, yCrop, xCrop + 23, yCrop + 23); // We crop out a random subwindow from each example image
			auto inputBuf = cimgToHTMLImageData(image);
			auto normalized = imageDataToNormalizedBuffer(inputBuf, baseResolution, baseResolution);
			IntegralImage integral(normalized, baseResolution, baseResolution, size, false);
			negativeSet.push_back(integral);
			delete [] inputBuf;
			delete [] normalized;
			std::cout << "Caching integral image for negative example " << count << "/" << negativeSetSize << std::endl;
			count += 1;
		}
	}

	// Precompute and cache all integrals for the negative validation set
	// TODO: Right now we just chop up all available raw images into 
	// subwindows and that's the size of our validation set... we can do better than this
	int vcount = 0;
	for (int i = 0; i < negativeValidationExamplePaths.size(); i += 1) {
		cimg_library::CImg<unsigned char> image(negativeValidationExamplePaths[i].c_str());
		for (int h = 0; h < image.height() - baseResolution; h += baseResolution) {
			for (int w = 0; w < image.width() - baseResolution; w += baseResolution) {
				auto crop = image.get_crop(w, h, w + baseResolution - 1, h + baseResolution - 1);
				auto inputBuf = cimgToHTMLImageData(crop);
				auto normalized = imageDataToNormalizedBuffer(inputBuf, baseResolution, baseResolution);
				IntegralImage integral(normalized, baseResolution, baseResolution, size, false);
				negativeValidationSet.push_back(integral);
				delete [] inputBuf;
				delete [] normalized;
				std::cout << "Caching integral image for negative validation set example " << vcount << std::endl;
				vcount += 1;
			}
		}
	}
}

// Convert CImg image format to an HTML imagedata-style 1D buffer of bytes
// Returns a pointer to an image buffer in pseudogreyscale format (luma in 4th byte)
unsigned char* cimgToHTMLImageData(cimg_library::CImg<unsigned char>& image) {
	if (image._spectrum == 1) {
		// Input is a greyscale 1-channel image
		unsigned char* imageData = new unsigned char[image.width() * image.height() * 4];
		for (int i = 0; i < image.height(); i += 1) {
			for (int j = 0; j < image.width(); j += 1) {
				int offset = (i * image.width() * 4) + (j * 4);
				imageData[offset] = 0; 						// R
				imageData[offset + 1] = 0;					// G
				imageData[offset + 2] = 0;					// B
				imageData[offset + 3] = 255 - image(j, i);	// A - note inversion for HTML5 canvas
			}
		}
		return imageData;
 	} else if (image._spectrum == 3) {
		// Input is a RGB 3-channel image
		unsigned char* imageData = new unsigned char[image.width() * image.height() * 4];
		for (int i = 0; i < image.height(); i += 1) {
			for (int j = 0; j < image.width(); j += 1) {
				unsigned char r = image(j, i, 0, 0);
				unsigned char g = image(j, i, 0, 1);
				unsigned char b = image(j, i, 0, 2);
				int luma = r * 0.2126 + g * 0.7152 + b * 0.0722;   // We should be using the utility function here
				int offset = (i * image.width() * 4) + (j * 4);
				imageData[offset] = 0; 					// R
				imageData[offset + 1] = 0;				// G
				imageData[offset + 2] = 0;				// B
				imageData[offset + 3] = 255 - luma;		// A - note inversion for HTML5 canvas
			}
		}
		return imageData;
	}
}

// Generate a complete ~160,000 feature set of Haar-like features (Type A, B, C, D, E as described in the Wang paper) 
// over all scales and positions for a given subwindow size
// returns a std::vector of Haarlike objects
std::vector<Haarlike> generateFeatures(int s) {
	std::vector<Haarlike> featureSet;
	// Type A
	for (int i = 0; i < s; i += 1) {
		for (int j = 0; j < s; j += 1) {
			for (int h = 1; i + h <= s; h += 1) {
				for (int w = 1; w * 2 + j <= s; w += 1) {
					Haarlike haarlike = Haarlike(s, j, i, w, h, 1);
					featureSet.push_back(haarlike);
				}
			}
		}
	}
	// Type B
	for (int i = 0; i < s; i += 1) {
		for (int j = 0; j < s; j += 1) {
			for (int h = 1; i + h <= s; h += 1) {
				for (int w = 1; w * 3 + j <= s; w += 1) {
					Haarlike haarlike = Haarlike(s, j, i, w, h, 2);
					featureSet.push_back(haarlike);
				}
			}
		}
	}
	// Type C
	for (int i = 0; i < s; i += 1) {
		for (int j = 0; j < s; j += 1) {
			for (int h = 1; i + h * 2 <= s; h += 1) {
				for (int w = 1; j + w <= s; w += 1) {
					Haarlike haarlike = Haarlike(s, j, i, w, h, 3);
					featureSet.push_back(haarlike);
				}
			}
		}
	}
	// Type D
	for (int i = 0; i < s; i += 1) {
		for (int j = 0; j < s; j += 1) {
			for (int h = 1; i + h * 3 <= s; h += 1) {
				for (int w = 1; j + w <= s; w += 1) {
					Haarlike haarlike = Haarlike(s, j, i, w, h, 4);
					featureSet.push_back(haarlike);
				}
			}
		}
	}
	// Type E
	for (int i = 0; i < s; i += 1) {
		for (int j = 0; j < s; j += 1) {
			for (int h = 1; i + h * 2 <= s; h += 1) {
				for (int w = 1; j + w * 2 <= s; w += 1) {
					Haarlike haarlike = Haarlike(s, j, i, w, h, 5);
					featureSet.push_back(haarlike);
				}
			}
		}
	}
	return featureSet;
}

// This is horrible!!! Refactor and write a nice simple generalized function!
void rebuildSets(
				CascadeClassifier& cascadeClassifier, 
				std::vector<IntegralImage>& negativeSet, 
				std::vector<std::string>& negativeExamplePaths,
				int negativeSetSize
				) {
	int size = cascadeClassifier.baseResolution * cascadeClassifier.baseResolution * 4;
	// Delete correct detections from the negative set
	for (int i = 0; i < negativeSet.size(); i += 1) {
		if (cascadeClassifier.classify(negativeSet[i], 0, 0, 0, 1) == false) negativeSet.erase(negativeSet.begin() + i);
	}
	
	// Examine every base resolution-sized subwindow of every negative example 
	// for subwindows that cause false positives and put them in our negative set
	int stepSize = 1;
	for (int i = 0; i < negativeExamplePaths.size() && negativeSet.size() < negativeSetSize; i += 1) {
		std::cout << "Rebuilding negative training set; examining raw negative image " << i << "/" << negativeExamplePaths.size() << " for suitable subwindows...\n";
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
					std::cout << "Currently have " << negativeSet.size() << " negative examples out of " << negativeSetSize - 1 << std::endl;
				}
			}
		}
	}

	if (negativeSet.size() < negativeSetSize) {
		std::cout << "\n---COULD NOT REBUILD A COMPLETE NEGATIVE SET!---\n";
	} 
}

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

WeakClassifier findOptimalWCThreshold(
									  std::vector<WeakClassifier> potentialWeakClassifiers, 
									  std::vector<float> posWeights, 
									  std::vector<float> negWeights
									  ) {
	
	std::sort(potentialWeakClassifiers.begin(), potentialWeakClassifiers.end(), comparePotentialWeakClassifiers);

	// Maintain a sum of all positive weights, a sum of all negative weights, and 
	// a sum of positive and negative weights below each sorted example image
	float sumPosWeights = 0;
	float sumNegWeights = 0;
	for (int j = 0; j < posWeights.size(); j += 1) sumPosWeights += posWeights[j];
	for (int j = 0; j < negWeights.size(); j += 1) sumNegWeights += negWeights[j];

	float sumPosWeightsBelow = 0;
	float sumNegWeightsBelow = 0;

	// std::cout << "sum pos weights: " << sumPosWeights << std::endl;
	// std::cout << "sum neg weights: " << sumNegWeights << std::endl;

	if (potentialWeakClassifiers[0].label == true ) sumPosWeightsBelow += potentialWeakClassifiers[0].weight;
	else sumNegWeightsBelow += potentialWeakClassifiers[0].weight;

	float minErr = 1;	
	int bestWC;

	for (int j = 0; j < potentialWeakClassifiers.size(); j += 1) {

		// NEW MATH HERE - IS THIS RIGHT???
		if (j > 0) {
			if (potentialWeakClassifiers[j].label == true) sumPosWeightsBelow += potentialWeakClassifiers[j].weight;
			else sumNegWeightsBelow += potentialWeakClassifiers[j].weight;
		}

		// Experimental: correct for floating point rounding error
		float posResult = sumPosWeights - sumPosWeightsBelow < 0 ? 0 : sumPosWeights - sumPosWeightsBelow;
		float negResult = sumNegWeights - sumNegWeightsBelow < 0 ? 0 : sumNegWeights - sumNegWeightsBelow;

		float negErr = sumNegWeightsBelow + posResult;
		float posErr = sumPosWeightsBelow + negResult;

		// DEBUGGING
		// if (posErr < 0) {
		// 	std::cout << std::setprecision(100) << "\n!!! posErr is negative! posErr = " << posErr << ", sumPosWeights = " << sumPosWeights <<
		// 		", sumNegWeights = " << sumNegWeights << ", sumPosWeightsBelow = " << sumPosWeightsBelow << 
		// 			", sumNegWeightsBelow = " << sumNegWeightsBelow << std::endl;
		// }
		// if (negErr < 0) {
		// 	std::cout << std::setprecision(100) << "\n!!! negErr is negative! negErr = " << negErr << ", sumPosWeights = " << sumPosWeights <<
		// 		", sumNegWeights = " << sumNegWeights << ", sumPosWeightsBelow = " << sumPosWeightsBelow << 
		// 			", sumNegWeightsBelow = " << sumNegWeightsBelow << std::endl;
		// }

		if (posErr < negErr) {
			if (posErr < minErr) {
				minErr = posErr;
				potentialWeakClassifiers[j].threshold = potentialWeakClassifiers[j].f; // TODO: This is dumb + redundant
				potentialWeakClassifiers[j].polarity = -1;  
				potentialWeakClassifiers[j].minErr = posErr;
				bestWC = j;
			}
		} else {
			if (negErr < minErr) {
				minErr = negErr;
				potentialWeakClassifiers[j].threshold = potentialWeakClassifiers[j].f; // TODO: This is dumb + redundant
				potentialWeakClassifiers[j].polarity = 1;  
				potentialWeakClassifiers[j].minErr = negErr;
				bestWC = j;
 			}
		}
	}
	return potentialWeakClassifiers[bestWC];
}

StrongClassifier adaBoost(
						CascadeClassifier& cascadeClassifier, 
						std::vector<IntegralImage>& positiveSet,
						std::vector<IntegralImage>& negativeSet,
						std::vector<IntegralImage>& positiveValidationSet,
						std::vector<IntegralImage>& negativeValidationSet,
						std::vector<Haarlike>& featureSet,
						float targetMaxFPR, 
						float targetMaxFNR, 
						int maxFeatures) {
	
	StrongClassifier* strongClassifier = new StrongClassifier();

	// Initialize weights
	std::vector<float> posWeights(positiveSet.size());
	std::vector<float> negWeights(negativeSet.size());
	std::fill(posWeights.begin(), posWeights.end(), (float)1 / (float)2 * (float)positiveSet.size());
	std::fill(negWeights.begin(), negWeights.end(), (float)1 / (float)2 * (float)negativeSet.size());

	float lastOverallFPR;

	if (cascadeClassifier.strongClassifiers.size() > 0){
		lastOverallFPR = cascadeClassifier.getFPR(negativeValidationSet);
	} else {
		lastOverallFPR = 1.0;
	}

	float currentOverallFPR = lastOverallFPR;
	float currentOverallFNR = 0;  // TODO: do we even use this?

	//  && strongClassifier->weakClassifiers.size() < maxFeatures
	// || strongClassifier->weakClassifiers.size() < minFeatures
	while (currentOverallFPR > targetMaxFPR * lastOverallFPR && strongClassifier->weakClassifiers.size() < maxFeatures) {
		// Stop adding new weak classifiers if all negative samples are correctly classified
		// I don't think this is described in the paper but I inferred it from an existing implementation
		// TODO: What the fuck is this?  We should probably remove it
    	// if (strongClassifier->weakClassifiers.size() > 0 && strongClassifier->getFPR(*this) == 0) {
    	// 	std::cout << "All negative examples classified correctly! Could not achieve target FPR for this round.\n";
     //  		return *strongClassifier;
    	// }
      
		std::cout << "\nAbout to select an WC to add to our current SC... our current FPR is " << currentOverallFPR 
			<< " and we will stop adding WCs when we hit an FPR of " << targetMaxFPR * lastOverallFPR << std::endl;

		float bestWCminError = 1;   // TODO: some of these identifiers are not necessary and we should initialize them 
		int championClassifier;		// in a nice clean line
		WeakClassifier bestWC;

		// Normalize the weights!
		float wSum = 0;
		for (int j = 0; j < positiveSet.size(); j += 1) wSum += posWeights[j];
		for (int j = 0; j < negativeSet.size(); j += 1) wSum += negWeights[j];
		for (int j = 0; j < positiveSet.size(); j += 1) posWeights[j] /= wSum;
		for (int j = 0; j < negativeSet.size(); j += 1) negWeights[j] /= wSum;
			
		// Do it for each of the ~162,336 features
		for (int i = 0; i < featureSet.size(); i += 1) {
			// Collect all potential weak classifiers for this feature
			std::vector<WeakClassifier> potentialWeakClassifiers;
			for (int j = 0; j < positiveSet.size(); j += 1) {
				auto f = positiveSet[j].computeFeature(featureSet[i], 0, 0);
				WeakClassifier wc(featureSet[i], f, true, posWeights[j]);
				potentialWeakClassifiers.push_back(wc); 
				//posFeatures[j] = f;
			}
			for (int j = 0; j < negativeSet.size(); j += 1) {
				auto f = negativeSet[j].computeFeature(featureSet[i], 0, 0);
				WeakClassifier wc(featureSet[i], f, false, negWeights[j]);
				potentialWeakClassifiers.push_back(wc);
				//negFeatures[j] = f;
			}
			// Pick the best weak classifier of all the potentials
			// std::cout << "\n\nTraining...\n";
			// std::cout << "Finding optimal threshold for feature " << i << " across " << this->positiveSet.size() - 1+ this->negativeSet.size() - 1 << " example images\n";
			auto wc = findOptimalWCThreshold(potentialWeakClassifiers, posWeights, negWeights);
			
			if (wc.minErr < bestWCminError) {
				//std::cout << "This is a new champion classifier!\n";
				championClassifier = i;
				bestWCminError = wc.minErr;
				bestWC = wc;
			}
			// std::cout << "Best weak classifier for feature " << i << ": " << "threshold " << wc.threshold << ", polarity " << wc.polarity << std::endl; 
			// std::cout << "Current champion classifier: feature " << championClassifier << std::endl;

			// std::cout << "type: " << featureSet[championClassifier].type << std::endl;
			// std::cout << "width: " << featureSet[championClassifier].w << std::endl;
			// std::cout << "height: " << featureSet[championClassifier].h << std::endl;
			// std::cout << "x offset: " << featureSet[championClassifier].x << std::endl;
			// std::cout << "y offset: " << featureSet[championClassifier].y << std::endl;

			// std::cout << "Strong classifier is currently composed of " << strongClassifier->weakClassifiers.size() << " weak classifiers\n";
		}
		// Now it's time to update the weights!
		float beta = bestWCminError / ((float)1.0 - bestWCminError);

		// This is a hack I inferred from some other source code... it could be breaking everything!
		if (beta < 1.0 / 100000000) {
			std::cout << "\nBad news! Beta was too small and we had to set it to a min value :(\n";
			beta = 1.0 / 100000000;
		}

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

		std::cout << "\nThe new best WC correctly classified " << correctlyClassifiedPositives << "/" << positiveSet.size() 
			<< " positive examples and " << correctlyClassifiedNegatives << "/" << negativeSet.size() << " negative examples\n";

		// Add the best weak classifier from this round to our strong classifier
		float alpha = std::log((float)1.0 / (float)beta);
		std::cout << "Adding a new WC to the SC with an alpha (SC weight) of " << alpha << std::endl;
		std::cout << "By the way, beta was: " << beta << std::endl;
		std::cout << "And the min error for this WC was " << bestWCminError << std::endl;

		strongClassifier->add(bestWC, alpha);
		strongClassifier->optimizeThreshold(positiveValidationSet, cascadeClassifier, targetMaxFNR);
		cascadeClassifier.add(*strongClassifier);
		//std::cout << "... which achiveed a FNR for the current cascade classifier of " << cascadeClassifier.getFNR(*this) << std::endl;
		currentOverallFPR = cascadeClassifier.getFPR(negativeValidationSet);
		currentOverallFNR = cascadeClassifier.getFNR(positiveValidationSet);
		cascadeClassifier.removeLast();
		std::cout << "\nWIP strong classifier after threshold optimization produces a cascade classifier with an FPR of "
			<< currentOverallFPR << " and an FNR of " << currentOverallFNR << " against a threshold of " << 
				strongClassifier->threshold << std::endl;
	}	
	return *strongClassifier;
}

int main() {
	float maxFNRPerLayer = 0.01; // the correct number here is really more like 0.01
	float targetMaxFPROverall = 0.00001;
	int targetLayers = 10;
	std::vector<float> targetFPRs = {0.5, .25, 0.303886886, 0.303886886, 0.303886886, 0.303886886, 0.303886886, 0.303886886, 0.303886886, 0.303886886};
	std::vector<int> featuresPerLayer = {3, 6, 30, 40, 40, 40, 50, 60, 60, 60};
	int negativeSetSize = 10000; // Number of negative example images to use per step

	// TODO: Add command line arguments for number of example images to include in each round???

	// TODO: set these via command line argument
	// std::experimental::filesystem::path pathToPositives("/home/noah/Desktop/testsets/positives"); // TODO: set via command line argument
	// std::experimental::filesystem::path pathToNegatives("/home/noah/Desktop/testsets/negatives"); //TODO: set via command line argument
	// std::experimental::filesystem::path pathToValidationNegatives("/home/noah/Desktop/testsets/negatives-validation");
	// std::experimental::filesystem::path pathToValidationPositives("/home/noah/Desktop/testsets/positives-validation");

	// std::experimental::filesystem::path pathToPositives("/home/noah/Desktop/datasets/lfwa"); // TODO: set via command line argument
	// std::experimental::filesystem::path pathToNegatives("/home/noah/Desktop/datasets/negatives"); //TODO: set via command line argument
	// std::experimental::filesystem::path pathToValidationNegatives("/home/noah/Desktop/datasets/negative-validation");
	// std::experimental::filesystem::path pathToValidationPositives("/home/noah/Desktop/datasets/lfwa-validation");

	std::experimental::filesystem::path pathToPositives("/home/noah/Desktop/datasets/lfwcrop"); // TODO: set via command line argument
	std::experimental::filesystem::path pathToNegatives("/home/noah/Desktop/datasets/negatives"); //TODO: set via command line argument
	std::experimental::filesystem::path pathToValidationNegatives("/home/noah/Desktop/datasets/negative-validation");
	std::experimental::filesystem::path pathToValidationPositives("/home/noah/Desktop/datasets/lfwcrop-validation");



	int baseResolution = 24;

	//CascadeTrainer ct(24); // TODO: base resolution should be set via command line argument

	std::cout << "Wasmface\n";
	std::cout << "Cascade classifier training\n";
	std::cout << "Performing recursive file scan for paths '" << pathToPositives.string() << "' and '" << pathToNegatives.string() << "'" << std::endl;
	std::cout << "(I sure hope there are only .jpg files in there)\n";

	std::vector<std::string> positiveExamplePaths;
	std::vector<std::string> negativeExamplePaths;
	std::vector<std::string> negativeValidationExamplePaths;
	std::vector<std::string> positiveValidationExamplePaths;

	collectLocalExamples(pathToPositives, positiveExamplePaths);
	collectLocalExamples(pathToNegatives, negativeExamplePaths);
	collectLocalExamples(pathToValidationNegatives, negativeValidationExamplePaths);
	collectLocalExamples(pathToValidationPositives, positiveValidationExamplePaths);

	std::cout << "\nPositive training images found: " << positiveExamplePaths.size() << std::endl;
	std::cout << "Negative training images found: " << negativeExamplePaths.size() << std::endl;
	std::cout << "Positive validation images found: " << positiveValidationExamplePaths.size() << std::endl;
	std::cout << "Negative validation images found: " << negativeValidationExamplePaths.size() << std::endl;

	std::vector<IntegralImage> negativeSet;
	std::vector<IntegralImage> positiveSet;
	std::vector<IntegralImage> negativeValidationSet;
	std::vector<IntegralImage> positiveValidationSet;

	computeAllIntegrals(
						baseResolution, 
						negativeSetSize,
						negativeSet, 
						positiveSet, 
						negativeValidationSet,
						positiveValidationSet,
						negativeExamplePaths,
						positiveExamplePaths, 
						negativeValidationExamplePaths,
						positiveValidationExamplePaths
						);
	
	std::cout << "\nDone!\n";

	std::cout << "\nGenerating feature set for base resolution " << 24 << "px...\n";
	
	auto featureSet = generateFeatures(24);
	
	std::cout << featureSet.size() << " features generated!\n";

	std::cout << "\nInitializing new cascade classifier composed of " << targetLayers << " layers, with a target max false positive rate of " << targetMaxFPROverall << std::endl;

	CascadeClassifier cascadeClassifier(24);

	float currentOverallFPR = 1.0;
	float currentOverallFNR = 1.0;

	for (int i = 0; i < targetLayers; i += 1) {
		std::cout << "\nRunning AdaBoost to create a new strong classifier ...\n";
		// We create a new strong classifier, deciding that it is done when
		// it meets our target rates for false positives and false negatives
		auto sc = adaBoost(
						   cascadeClassifier, 
						   positiveSet,
						   negativeSet,
						   positiveValidationSet,
						   negativeValidationSet,
						   featureSet,
						   targetFPRs[i], 
						   maxFNRPerLayer, 
						   featuresPerLayer[i]
						   );
	
		// Add the new strong classifier as a layer to our cascade classifier
		// and get the current rate of false positives for the cascade classifier
		cascadeClassifier.add(sc);

		currentOverallFPR = cascadeClassifier.getFPR(negativeValidationSet);

		currentOverallFNR = cascadeClassifier.getFNR(positiveValidationSet);

		std::ofstream modelFile;
		std::string filename = "../models/human-face-" + std::to_string(i + 1) + "-layers.js";
		modelFile.open(filename, std::ios_base::trunc);
		std::string json = cascadeToJSON(cascadeClassifier, currentOverallFPR, currentOverallFNR);
		modelFile << "const wasmfaceModel = " << json;
		modelFile.close();
			
		std::cout << "\n --> Added a new strong classifier with " << sc.weakClassifiers.size() << 
			" WCs to the cascade classifier! Current cascade classifier now has " << cascadeClassifier.strongClassifiers.size() << 
			" layers; it produces an overall false positive rate of " << currentOverallFPR << 
				" and an overall false negative rate of " << currentOverallFNR << std::endl;

		// TODO: I think we're also supposed to use this SC to eliminate false detections
		// from the negative set (aka correct classifications???)
		for (int i = 0; i < negativeSet.size(); i += 1) {
			if (sc.classify(negativeSet[i], 0, 0, 0, 1) == false) negativeSet.erase(negativeSet.begin() + i);
		}

		// TODO: we're now removing images from our positive set that caused false negatives...
		// not sure how we manage the size of our positive set???
		for (int i = 0; i < positiveSet.size(); i += 1) {
			if (sc.classify(positiveSet[i], 0, 0, 0, 1) == false) positiveSet.erase(positiveSet.begin() + i);
		}

		// for (int i = 0; i < positiveValidationSet.size(); i += 1) {
		// 	if (sc.classify(positiveValidationSet[i], baseResolution, 0, 0) == false) positiveValidationSet.erase(positiveValidationSet.begin() + i);
		// }

		rebuildSets(cascadeClassifier, negativeSet, negativeExamplePaths, negativeSetSize);
		
	}
	std::cout << "Successfully trained a cascade with an FPR of " << currentOverallFPR << std::endl;
	return 0;
}