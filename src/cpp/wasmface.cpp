#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <emscripten/emscripten.h>

#include "../../lib/json.hpp"

#include "wasmface.h"
#include "utility.h"
#include "integral-image.h"
#include "strong-classifier.h"
#include "cascade-classifier.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compare two pointers based on their dereferenced values
 * @param  {Int*} a First pointer
 * @param  {Int*} b Second pointer
 * @return {Bool}   True if the first pointer's dereferenced value is the smaller of the two
 */
bool compareDereferencedPtrs(int* a, int* b) {
	return *a < *b;
}

/**
 * Apply non-maximum suppression to a set of 1:1 aspect ratio bounding boxes
 * Bounding boxes are represented as [x, y, s] where s = width and height
 * @param  {std::vector<std::array<int, 3>>} boxes   The set of bounding boxes
 * @param  {Float}                           thresh  The minimum overlap ratio required for suppression
 * @param  {Float}                           nthresh The minimum number of neighboring boxes required for suppression
 * @return {std::vector<std::array<int, 3>>}         The suppressed set of bounding boxes
 */
std::vector<std::array<int, 3>> nonMaxSuppression(std::vector<std::array<int, 3>>& boxes, float thresh, int nthresh) {
	int len = boxes.size();
	if (!len) return boxes;

	// Destructure our bounding boxes into arrays representing upper left and lower right coords
	int* x1 = new int[len];
	int* y1 = new int[len];
	int* x2 = new int[len];
	int* y2 = new int[len];
	int* area = new int[len];
	for (int i = 0; i < len; i += 1) {
		x1[i] = boxes[i][0];
		y1[i] = boxes[i][1];
		x2[i] = boxes[i][0] + boxes[i][2] - 1;
		y2[i] = boxes[i][1] + boxes[i][2] - 1;
		area[i] = std::pow(boxes[i][2], 2);
	}

	// Create an array of the indices that would sort our y2 coords
	int** ptrs = new int*[len];
	for (int i = 0; i < len; i += 1) ptrs[i] = &y2[i];
	std::sort(ptrs, ptrs + len, compareDereferencedPtrs);
	std::vector<int> ind(len);
	for (int i = 0; i < len; i += 1) ind[i] = ptrs[i] - &y2[0];

	std::vector<std::pair<int, int>> pick;
	while (ind.size() > 0) {
		int last = ind.size() - 1;
		int n = ind[last]; 
		pick.push_back(std::pair<int, int> (n, 0));  
		
		// Suppress bounding boxes that overlap 
		int neighborsCount = 0;
		std::vector<int> suppress = {last};
		for (int i = 0; i < last; i += 1) {
			int j = ind[i];
			int xx1 = std::max(x1[n], x1[j]);
			int yy1 = std::max(y1[n], y1[j]);
			int xx2 = std::min(x2[n], x2[j]);
			int yy2 = std::min(y2[n], y2[j]);
			int w = std::max(0, xx2 - xx1 + 1);
			int h = std::max(0, yy2 - yy1 + 1);

			float overlap = float(w * h) / area[j];

			if (overlap > thresh) {
				suppress.push_back(i);
				neighborsCount += 1;
			}
		}

		for (int i = 0; i < suppress.size(); i += 1) ind.erase(ind.begin() + suppress[i]);
		pick.back().second = neighborsCount;
	}

	delete [] x1;
	delete [] y1;
	delete [] x2;
	delete [] y2;
	delete [] area;
	delete [] ptrs;

	// Also suppress boxes that do not have the minimum number of neighbors
	std::vector<std::array<int, 3>> result;
	for (int i = 0; i < pick.size(); i += 1) {
		if (pick[i].second >= nthresh) result.push_back(boxes[pick[i].first]);
	}
	return result;
} 

/**
 * Deserialize and construct a cascade classifier object
 * @param  {Char*}              model A serialized cascade classifier object
 * @return {CascadeClassifier*}       A pointer to a new cascade classifier object
 */
EMSCRIPTEN_KEEPALIVE CascadeClassifier* create(char model[]) {
	auto ccJSON = nlohmann::json::parse(model);

	std::vector<StrongClassifier> sc;
	for (int i = 0; i < ccJSON["strongClassifiers"].size(); i += 1) {
		StrongClassifier strongClassifier;
		strongClassifier.threshold = ccJSON["strongClassifiers"][i]["threshold"];
		for (int j = 0; j < ccJSON["strongClassifiers"][i]["weakClassifiers"].size(); j += 1) {
			WeakClassifier weakClassifier;
			weakClassifier.haarlike.type = ccJSON["strongClassifiers"][i]["weakClassifiers"][j]["type"];
			weakClassifier.haarlike.w = ccJSON["strongClassifiers"][i]["weakClassifiers"][j]["w"];
			weakClassifier.haarlike.h = ccJSON["strongClassifiers"][i]["weakClassifiers"][j]["h"];
			weakClassifier.haarlike.x = ccJSON["strongClassifiers"][i]["weakClassifiers"][j]["x"];
			weakClassifier.haarlike.y = ccJSON["strongClassifiers"][i]["weakClassifiers"][j]["y"];
			weakClassifier.threshold = ccJSON["strongClassifiers"][i]["weakClassifiers"][j]["threshold"];
			weakClassifier.polarity = ccJSON["strongClassifiers"][i]["weakClassifiers"][j]["polarity"];
			strongClassifier.weakClassifiers.push_back(weakClassifier);
			strongClassifier.weights.push_back(ccJSON["strongClassifiers"][i]["weights"][j]);
		}
		sc.push_back(strongClassifier);
	}

	CascadeClassifier* cc = new CascadeClassifier(ccJSON["baseResolution"], sc);
	return cc;
}

/**
 * Destroy a cascade classifier object
 * Provided as a JavaScript-callable function
 * @param {CascadeClassifier*} cc Pointer to the cascade classifier to destroy
 */
EMSCRIPTEN_KEEPALIVE void destroy(CascadeClassifier* cc) {
	delete cc;
}

/**
 * Use a cascade classifier to detect objects in an HTML5 ImageData buffer
 * @param  {Unsigned char*}     inputBuf Pointer to an HTML5 ImageData buffer
 * @param  {Int}                w        Width of the ImageData object
 * @param  {Int}                h        Height of the ImageData object
 * @param  {CascadeClassifier*} cco      Pointer to a cascade classifier object
 * @param  {Float}              step     Detector scale step to apply
 * @param  {Float}              delta    Detector sweep delta to apply
 * @param  {Bool}               pp       True applies post processing
 * @param  {Float}              othresh  Overlap threshold for post processing
 * @param  {Float}              nthresh  Neighbor threshold for post processing
 * @return {uint16_t*}                   Pointer to an array of bounding box geometry
 */
EMSCRIPTEN_KEEPALIVE uint16_t* detect(unsigned char inputBuf[], int w, int h, CascadeClassifier* cco, 
                                      float step, float delta, bool pp, float othresh, int nthresh) {
	CascadeClassifier* cc = new CascadeClassifier(*cco);
	
	int byteSize = w * h * 4;
	auto fpgs = toGrayscaleFloat(inputBuf, w, h);
	auto integral = IntegralImage(fpgs, w, h, byteSize, false);
	auto integralSquared = IntegralImage(fpgs, w, h, byteSize, true);
	delete [] fpgs;

	// Sweep and scale the detector over the post-normalized input image and collect detections
	std::vector<std::array<int, 3>> roi;
	while (cc->baseResolution < w && cc->baseResolution < h) {
		for (int y = 0; y < h - cc->baseResolution; y += step * delta) {
			for (int x = 0; x < w - cc->baseResolution; x += step * delta) {
				float sum = integral.getRectangleSum(x, y, cc->baseResolution, cc->baseResolution);
				float squaredSum = integralSquared.getRectangleSum(x, y, cc->baseResolution, cc->baseResolution);
				float area = std::pow(cc->baseResolution, 2);
				float mean = sum / area;
				float sd = std::sqrt(squaredSum / area - std::pow(mean, 2));
				bool c = cc->classify(integral, x, y, mean, sd);
				
				if (c) {
					std::array<int, 3> bounding = {x, y, cc->baseResolution};
					roi.push_back(bounding);
				}
			}
		}
		cc->scale(step);
	}

	if (pp) roi = nonMaxSuppression(roi, othresh, nthresh);

	// We return a 1D array on the heap with its length stashed as the first element
	int blen = roi.size() * 3 + 1;
	uint16_t* boxes = new uint16_t[blen];
	boxes[0] = blen;
	for (int i = 0, j = 1; i < roi.size(); i += 1, j += 3) {
		boxes[j] = roi[i][0];
		boxes[j + 1] = roi[i][1];
		boxes[j + 2] = roi[i][2];
	}
	
	delete cc;
	return boxes;
}

/**
 * Main function
 * @return {Int}
 */
int main() {
	std::cout << "Made with Wasmface\n";
	return 0;
}

#ifdef __cplusplus
}
#endif