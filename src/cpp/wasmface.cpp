// source ./emsdk_env.sh --build=Release

// emcc wasmface.cpp cascade-classifier.cpp haar-like.cpp integral-image.cpp strong-classifier.cpp utility.cpp weak-classifier.cpp -s TOTAL_MEMORY=1024MB -s "EXTRA_EXPORTED_RUNTIME_METHODS=['ccall', 'cwrap', 'allocate']" -s WASM=1 -O3 -std=c++1z -o ../../demo/wasmface.js

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

bool compareDereferencedPtrs(int* a, int* b) {
	return *a < *b;
}

std::vector<std::array<int, 3>> nonMaxSuppression(std::vector<std::array<int, 3>>& boxes, float thresh, int nthresh) {
	int len = boxes.size();
	if (!len) return boxes;

	// Destructure our bounding boxes into arrays of coords and calculate areas
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

	// To create an array of the indexes that would sort our y2 coords:
	// Create an array of pointers to our y2 coords, sort by the value they
	// point to, then subtract the pointer to the start of the y2 coords array from each pointer
	int** ptrs = new int*[len];
	for (int i = 0; i < len; i += 1) ptrs[i] = &y2[i];
	std::sort(ptrs, ptrs + len, compareDereferencedPtrs);
	std::vector<int> ind(len);
	for (int i = 0; i < len; i += 1) ind[i] = ptrs[i] - &y2[0];

	// for (int i = 0; i < ind.size(); i += 1) std::cout << ind[i] << ", ";
	// 	std::cout << std::endl;

	std::vector<std::pair<int, int>> pick;
	while (ind.size() > 0) {
		int last = ind.size() - 1;
		int n = ind[last]; // This is selecting the bounding box that is lowest on screen?
		pick.push_back(std::pair<int, int> (n, 0));  // And we just select it and pick it as a keeper?
		
		std::vector<int> suppress = {last};

		int neighborsCount = 0;

		for (int i = 0; i < last; i += 1) {
			int j = ind[i];

			// What's bigger, the upper left corner of the lowest on-screen box, or the upper left
			// corner of the box we're currently evaluating?
			int xx1 = std::max(x1[n], x1[j]);
			int yy1 = std::max(y1[n], y1[j]);
			// What's smaller, the lower right corner of the lowest on-screen box, or the lower right
			// corner of the box we're currently evaluating?
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

	std::vector<std::array<int, 3>> result;
	for (int i = 0; i < pick.size(); i += 1) {
		if (pick[i].second > nthresh) result.push_back(boxes[pick[i].first]);
	}
	return result;
} 

// TODO: implement a destructor
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

EMSCRIPTEN_KEEPALIVE int detect(unsigned char inputBuf[], int w, int h, CascadeClassifier* cco, 
							    float step, float delta, float othresh, int nthresh) {
	CascadeClassifier* cc = new CascadeClassifier(*cco);
	int byteSize = w * h * 4;
	auto fpgs = toGrayscaleFloat(inputBuf, w, h);
	auto integral = IntegralImage(fpgs, w, h, byteSize, false);
	auto integralSquared = IntegralImage(fpgs, w, h, byteSize, true);
	delete [] fpgs;

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

	// for (int i = 0; i < roi.size(); i += 1) {
	// 	EM_ASM({
	// 		outputOverlayCtx.strokeStyle = '#ff0000';
	// 		outputOverlayCtx.lineWidth = 6;
	// 		outputOverlayCtx.beginPath();
	// 		outputOverlayCtx.rect($0, $1, $2, $2); 
	// 		outputOverlayCtx.stroke();
	// 	},roi[i][0], roi[i][1], roi[i][2]);
	// }

	auto boxes = nonMaxSuppression(roi, othresh, nthresh);

	for (int i = 0; i < boxes.size(); i += 1) {
		EM_ASM({
			outputOverlayCtx.strokeStyle = '#77ff33';
			outputOverlayCtx.lineWidth = 4;
			outputOverlayCtx.beginPath();
			outputOverlayCtx.rect($0, $1, $2, $2); 
			outputOverlayCtx.stroke();
		}, boxes[i][0], boxes[i][1], boxes[i][2]);
	}

	delete cc;
	return 1;
}

int main() {
	std::cout << "Made with Wasmface\n";
	return 0;
}

#ifdef __cplusplus
}
#endif