// source ./emsdk_env.sh --build=Release

// emcc wasmface.cpp cascade-classifier.cpp haar-like.cpp integral-image.cpp strong-classifier.cpp utility.cpp weak-classifier.cpp -s TOTAL_MEMORY=1024MB -s "EXTRA_EXPORTED_RUNTIME_METHODS=['ccall', 'cwrap']" -s WASM=1 -O3 -std=c++1z -o ../demo2/wasmface.js

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <emscripten/emscripten.h>

#include "../lib/json.hpp"

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

	std::vector<int> pick;
	std::vector<int> neighbors;
	while (ind.size() > 0) {
		int last = ind.size() - 1;
		int n = ind[last]; // This is selecting the bounding box that is lowest on screen?
		pick.push_back(n);  // And we just select it and pick it as a keeper?
		
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
		neighbors.push_back(neighborsCount);
	}

	delete [] x1;
	delete [] y1;
	delete [] x2;
	delete [] y2;
	delete [] area;
	delete [] ptrs;

	std::vector<std::array<int, 3>> result;
	for (int i = 0; i < pick.size(); i += 1) {
		if (neighbors[i] > nthresh) result.push_back(boxes[pick[i]]);
	}
	return result;
} 

EMSCRIPTEN_KEEPALIVE int detect(unsigned char inputBuf[], int w, int h) {
	int byteSize = w * h * 4;

	auto fpgs = toGrayscaleFloat(inputBuf, w, h);
	auto integral = IntegralImage(fpgs, w, h, byteSize, false);
	auto integralSquared = IntegralImage(fpgs, w, h, byteSize, true);
	delete [] fpgs;

	// For development only: We rebuild a cascade model on every single function call
	// TODO: Construct the model outside of this function on init
	auto ccJSON = nlohmann::json::parse("{\"baseResolution\":24,\"fnr\":0.02670878730714321,\"fpr\":0.0020242915488779545,\"strongClassifiers\":[{\"threshold\":-5.338443756103516,\"weakClassifiers\":[{\"h\":4,\"polarity\":-1,\"threshold\":46.99998474121094,\"type\":4,\"w\":15,\"x\":5,\"y\":0},{\"h\":7,\"polarity\":-1,\"threshold\":6.804267883300781,\"type\":1,\"w\":3,\"x\":12,\"y\":1},{\"h\":6,\"polarity\":1,\"threshold\":-3.7276840209960938,\"type\":3,\"w\":2,\"x\":5,\"y\":3}],\"weights\":[2.2908742427825928,1.641132116317749,1.4064373970031738]},{\"threshold\":-3.394113063812256,\"weakClassifiers\":[{\"h\":4,\"polarity\":-1,\"threshold\":46.99998474121094,\"type\":4,\"w\":15,\"x\":5,\"y\":0},{\"h\":7,\"polarity\":-1,\"threshold\":6.804267883300781,\"type\":1,\"w\":3,\"x\":12,\"y\":1},{\"h\":6,\"polarity\":1,\"threshold\":-3.7276840209960938,\"type\":3,\"w\":2,\"x\":5,\"y\":3},{\"h\":1,\"polarity\":1,\"threshold\":-3.2086563110351563,\"type\":1,\"w\":5,\"x\":0,\"y\":0},{\"h\":6,\"polarity\":1,\"threshold\":-38.41594696044922,\"type\":4,\"w\":24,\"x\":0,\"y\":3}],\"weights\":[2.2908742427825928,1.641132116317749,1.4064373970031738,0.9787285923957825,0.9656018018722534]},{\"threshold\":-2.8806662559509277,\"weakClassifiers\":[{\"h\":4,\"polarity\":-1,\"threshold\":65.62350463867188,\"type\":4,\"w\":18,\"x\":3,\"y\":0},{\"h\":7,\"polarity\":1,\"threshold\":-5.908620834350586,\"type\":1,\"w\":3,\"x\":7,\"y\":1},{\"h\":4,\"polarity\":1,\"threshold\":-7.937835693359375,\"type\":3,\"w\":5,\"x\":17,\"y\":4},{\"h\":24,\"polarity\":1,\"threshold\":-16.340595245361328,\"type\":1,\"w\":2,\"x\":0,\"y\":0},{\"h\":7,\"polarity\":1,\"threshold\":-41.822994232177734,\"type\":4,\"w\":24,\"x\":0,\"y\":2},{\"h\":2,\"polarity\":1,\"threshold\":-2.0539302825927734,\"type\":5,\"w\":4,\"x\":15,\"y\":19},{\"h\":3,\"polarity\":1,\"threshold\":-5.110252380371094,\"type\":5,\"w\":6,\"x\":0,\"y\":0}],\"weights\":[1.9568027257919312,1.3985168933868408,1.2337216138839722,0.953900158405304,0.9304466247558594,0.872736930847168,0.8590916395187378]},{\"threshold\":-3.3162031173706055,\"weakClassifiers\":[{\"h\":4,\"polarity\":-1,\"threshold\":85.45214080810547,\"type\":4,\"w\":19,\"x\":3,\"y\":0},{\"h\":7,\"polarity\":-1,\"threshold\":9.815784454345703,\"type\":1,\"w\":4,\"x\":11,\"y\":2},{\"h\":1,\"polarity\":-1,\"threshold\":4.44256591796875,\"type\":4,\"w\":21,\"x\":2,\"y\":9},{\"h\":3,\"polarity\":1,\"threshold\":-7.1866607666015625,\"type\":1,\"w\":5,\"x\":0,\"y\":21},{\"h\":3,\"polarity\":-1,\"threshold\":17.759613037109375,\"type\":3,\"w\":14,\"x\":5,\"y\":0},{\"h\":3,\"polarity\":1,\"threshold\":-2.4973526000976563,\"type\":1,\"w\":2,\"x\":9,\"y\":5},{\"h\":7,\"polarity\":1,\"threshold\":-2.724864959716797,\"type\":3,\"w\":3,\"x\":4,\"y\":3},{\"h\":3,\"polarity\":1,\"threshold\":-6.4412994384765625,\"type\":5,\"w\":5,\"x\":13,\"y\":18},{\"h\":1,\"polarity\":-1,\"threshold\":0.18508052825927734,\"type\":2,\"w\":1,\"x\":11,\"y\":0},{\"h\":1,\"polarity\":-1,\"threshold\":2.824970245361328,\"type\":3,\"w\":6,\"x\":10,\"y\":17},{\"h\":3,\"polarity\":1,\"threshold\":-17.30077362060547,\"type\":3,\"w\":18,\"x\":3,\"y\":5},{\"h\":24,\"polarity\":-1,\"threshold\":3.240013599395752,\"type\":1,\"w\":1,\"x\":22,\"y\":0},{\"h\":2,\"polarity\":1,\"threshold\":-1.141357421875,\"type\":1,\"w\":6,\"x\":10,\"y\":14},{\"h\":23,\"polarity\":-1,\"threshold\":14.669891357421875,\"type\":2,\"w\":4,\"x\":3,\"y\":1},{\"h\":3,\"polarity\":1,\"threshold\":-0.8436336517333984,\"type\":3,\"w\":21,\"x\":2,\"y\":7}],\"weights\":[1.4254679679870605,1.380274772644043,1.0303212404251099,1.0379102230072021,1.008321762084961,0.9063416123390198,0.91350919008255,0.8710759282112122,0.6678094863891602,0.7356999516487122,0.7481126189231873,0.745220959186554,0.698285698890686,0.6261972784996033,0.6888036131858826]},{\"threshold\":-4.0039825439453125,\"weakClassifiers\":[{\"h\":4,\"polarity\":-1,\"threshold\":3.87274169921875,\"type\":1,\"w\":2,\"x\":12,\"y\":4},{\"h\":3,\"polarity\":-1,\"threshold\":27.969200134277344,\"type\":3,\"w\":18,\"x\":2,\"y\":0},{\"h\":3,\"polarity\":-1,\"threshold\":4.84210205078125,\"type\":5,\"w\":4,\"x\":3,\"y\":18},{\"h\":1,\"polarity\":-1,\"threshold\":6.283714294433594,\"type\":4,\"w\":21,\"x\":2,\"y\":9},{\"h\":5,\"polarity\":-1,\"threshold\":4.497568607330322,\"type\":5,\"w\":3,\"x\":1,\"y\":5},{\"h\":1,\"polarity\":-1,\"threshold\":2.1843223571777344,\"type\":3,\"w\":5,\"x\":16,\"y\":4},{\"h\":9,\"polarity\":-1,\"threshold\":12.147953033447266,\"type\":5,\"w\":5,\"x\":11,\"y\":1},{\"h\":1,\"polarity\":-1,\"threshold\":0.5918188095092773,\"type\":2,\"w\":3,\"x\":8,\"y\":0},{\"h\":3,\"polarity\":-1,\"threshold\":3.1399707794189453,\"type\":1,\"w\":3,\"x\":18,\"y\":21},{\"h\":1,\"polarity\":-1,\"threshold\":3.8450660705566406,\"type\":4,\"w\":24,\"x\":0,\"y\":10},{\"h\":14,\"polarity\":1,\"threshold\":-10.57080078125,\"type\":1,\"w\":2,\"x\":0,\"y\":10},{\"h\":3,\"polarity\":-1,\"threshold\":2.8512954711914063,\"type\":5,\"w\":3,\"x\":18,\"y\":0},{\"h\":8,\"polarity\":1,\"threshold\":-7.563283920288086,\"type\":1,\"w\":3,\"x\":7,\"y\":1},{\"h\":1,\"polarity\":1,\"threshold\":-0.24581146240234375,\"type\":3,\"w\":5,\"x\":10,\"y\":15},{\"h\":5,\"polarity\":-1,\"threshold\":-2.7397727966308594,\"type\":3,\"w\":22,\"x\":2,\"y\":0},{\"h\":1,\"polarity\":1,\"threshold\":-2.5729827880859375,\"type\":3,\"w\":6,\"x\":11,\"y\":19},{\"h\":7,\"polarity\":1,\"threshold\":4.575675964355469,\"type\":3,\"w\":18,\"x\":2,\"y\":1},{\"h\":1,\"polarity\":-1,\"threshold\":1.6214828491210938,\"type\":5,\"w\":4,\"x\":12,\"y\":19},{\"h\":4,\"polarity\":-1,\"threshold\":-12.719207763671875,\"type\":3,\"w\":19,\"x\":1,\"y\":8},{\"h\":23,\"polarity\":-1,\"threshold\":3.3862085342407227,\"type\":1,\"w\":1,\"x\":22,\"y\":0},{\"h\":1,\"polarity\":1,\"threshold\":-0.01470947265625,\"type\":5,\"w\":1,\"x\":18,\"y\":22},{\"h\":5,\"polarity\":-1,\"threshold\":2.2784805297851563,\"type\":3,\"w\":5,\"x\":6,\"y\":12},{\"h\":22,\"polarity\":1,\"threshold\":-21.003694534301758,\"type\":2,\"w\":3,\"x\":8,\"y\":2},{\"h\":1,\"polarity\":1,\"threshold\":0.1878722906112671,\"type\":3,\"w\":1,\"x\":23,\"y\":0},{\"h\":5,\"polarity\":1,\"threshold\":-2.722888469696045,\"type\":1,\"w\":2,\"x\":0,\"y\":0},{\"h\":2,\"polarity\":1,\"threshold\":-4.782917022705078,\"type\":3,\"w\":14,\"x\":0,\"y\":6},{\"h\":1,\"polarity\":-1,\"threshold\":0.4070587158203125,\"type\":5,\"w\":1,\"x\":19,\"y\":5},{\"h\":1,\"polarity\":-1,\"threshold\":-0.3169288635253906,\"type\":3,\"w\":5,\"x\":19,\"y\":13},{\"h\":1,\"polarity\":-1,\"threshold\":0.3866729736328125,\"type\":5,\"w\":1,\"x\":17,\"y\":5},{\"h\":7,\"polarity\":1,\"threshold\":10.477012634277344,\"type\":3,\"w\":12,\"x\":12,\"y\":3},{\"h\":1,\"polarity\":1,\"threshold\":-0.23751449584960938,\"type\":5,\"w\":1,\"x\":8,\"y\":5},{\"h\":2,\"polarity\":-1,\"threshold\":0.04652595520019531,\"type\":3,\"w\":5,\"x\":10,\"y\":0},{\"h\":1,\"polarity\":-1,\"threshold\":0.3122386932373047,\"type\":5,\"w\":1,\"x\":8,\"y\":5},{\"h\":9,\"polarity\":-1,\"threshold\":-15.551589965820313,\"type\":3,\"w\":6,\"x\":11,\"y\":2},{\"h\":1,\"polarity\":-1,\"threshold\":1.232034683227539,\"type\":5,\"w\":5,\"x\":4,\"y\":22},{\"h\":1,\"polarity\":-1,\"threshold\":3.0517578125e-05,\"type\":3,\"w\":2,\"x\":4,\"y\":14},{\"h\":14,\"polarity\":-1,\"threshold\":37.32847595214844,\"type\":2,\"w\":7,\"x\":0,\"y\":10},{\"h\":5,\"polarity\":1,\"threshold\":25.257413864135742,\"type\":5,\"w\":11,\"x\":1,\"y\":10}],\"weights\":[1.2109524011611938,1.0512163639068604,1.0441310405731201,1.075690507888794,0.839128851890564,0.7703808546066284,0.7994118928909302,0.7098618149757385,0.7132012844085693,0.7009995579719543,0.6797688007354736,0.6311349868774414,0.6245829463005066,0.6035668253898621,0.5661067962646484,0.5644234418869019,0.7251126766204834,0.6803959608078003,0.6189637184143066,0.630258321762085,0.49700143933296204,0.5097678899765015,0.5477471351623535,0.5354641675949097,0.5509768724441528,0.5107681155204773,0.6311917304992676,0.564540445804596,0.5506783127784729,0.5834228992462158,0.573786735534668,0.5154768824577332,0.5457881689071655,0.5779250860214233,0.5483381748199463,0.4981406629085541,0.5031972527503967,0.5443611145019531]},{\"threshold\":-3.3386247158050537,\"weakClassifiers\":[{\"h\":2,\"polarity\":1,\"threshold\":-15.351604461669922,\"type\":2,\"w\":5,\"x\":5,\"y\":5},{\"h\":1,\"polarity\":-1,\"threshold\":1.15325927734375,\"type\":2,\"w\":2,\"x\":8,\"y\":0},{\"h\":14,\"polarity\":1,\"threshold\":-15.934492111206055,\"type\":1,\"w\":3,\"x\":0,\"y\":10},{\"h\":4,\"polarity\":1,\"threshold\":-7.2823028564453125,\"type\":5,\"w\":5,\"x\":0,\"y\":0},{\"h\":6,\"polarity\":1,\"threshold\":-76.69715118408203,\"type\":4,\"w\":23,\"x\":0,\"y\":3},{\"h\":2,\"polarity\":1,\"threshold\":-12.625783920288086,\"type\":4,\"w\":7,\"x\":9,\"y\":14},{\"h\":8,\"polarity\":1,\"threshold\":-6.026630401611328,\"type\":1,\"w\":3,\"x\":7,\"y\":0},{\"h\":1,\"polarity\":-1,\"threshold\":1.3811416625976563,\"type\":1,\"w\":4,\"x\":16,\"y\":23},{\"h\":3,\"polarity\":-1,\"threshold\":4.3078460693359375,\"type\":5,\"w\":4,\"x\":16,\"y\":0},{\"h\":1,\"polarity\":-1,\"threshold\":6.103515625e-05,\"type\":1,\"w\":1,\"x\":21,\"y\":12},{\"h\":1,\"polarity\":-1,\"threshold\":3.0517578125e-05,\"type\":1,\"w\":1,\"x\":7,\"y\":11},{\"h\":1,\"polarity\":-1,\"threshold\":1.2381935119628906,\"type\":3,\"w\":3,\"x\":11,\"y\":19},{\"h\":1,\"polarity\":-1,\"threshold\":-0.7305335998535156,\"type\":3,\"w\":5,\"x\":1,\"y\":12},{\"h\":1,\"polarity\":1,\"threshold\":-3.035050392150879,\"type\":3,\"w\":6,\"x\":11,\"y\":19},{\"h\":1,\"polarity\":1,\"threshold\":1.9676437377929688,\"type\":3,\"w\":6,\"x\":0,\"y\":12},{\"h\":1,\"polarity\":1,\"threshold\":-5.0635223388671875,\"type\":1,\"w\":6,\"x\":0,\"y\":23},{\"h\":3,\"polarity\":1,\"threshold\":-14.705753326416016,\"type\":3,\"w\":15,\"x\":9,\"y\":5},{\"h\":5,\"polarity\":1,\"threshold\":-4.232185363769531,\"type\":1,\"w\":2,\"x\":10,\"y\":3},{\"h\":2,\"polarity\":1,\"threshold\":-1.5173473358154297,\"type\":1,\"w\":3,\"x\":14,\"y\":13},{\"h\":1,\"polarity\":-1,\"threshold\":0.013789176940917969,\"type\":1,\"w\":1,\"x\":22,\"y\":0},{\"h\":1,\"polarity\":-1,\"threshold\":0.38714599609375,\"type\":5,\"w\":1,\"x\":17,\"y\":5},{\"h\":1,\"polarity\":-1,\"threshold\":-1.4066734313964844,\"type\":3,\"w\":5,\"x\":10,\"y\":9},{\"h\":1,\"polarity\":-1,\"threshold\":1.49017333984375,\"type\":3,\"w\":2,\"x\":12,\"y\":13},{\"h\":2,\"polarity\":-1,\"threshold\":5.561219692230225,\"type\":3,\"w\":15,\"x\":3,\"y\":0},{\"h\":1,\"polarity\":1,\"threshold\":0.30498701333999634,\"type\":3,\"w\":2,\"x\":0,\"y\":0},{\"h\":9,\"polarity\":1,\"threshold\":-30.97545623779297,\"type\":1,\"w\":5,\"x\":0,\"y\":0},{\"h\":6,\"polarity\":-1,\"threshold\":1.0097274780273438,\"type\":3,\"w\":7,\"x\":4,\"y\":10},{\"h\":1,\"polarity\":1,\"threshold\":-0.37926483154296875,\"type\":5,\"w\":1,\"x\":17,\"y\":5},{\"h\":1,\"polarity\":1,\"threshold\":1.1037158966064453,\"type\":3,\"w\":3,\"x\":21,\"y\":12},{\"h\":22,\"polarity\":-1,\"threshold\":33.04938888549805,\"type\":1,\"w\":5,\"x\":0,\"y\":1},{\"h\":1,\"polarity\":-1,\"threshold\":-0.29681396484375,\"type\":3,\"w\":2,\"x\":22,\"y\":12},{\"h\":1,\"polarity\":1,\"threshold\":-0.9320993423461914,\"type\":3,\"w\":3,\"x\":9,\"y\":14},{\"h\":1,\"polarity\":1,\"threshold\":0.5599803924560547,\"type\":3,\"w\":3,\"x\":21,\"y\":0},{\"h\":1,\"polarity\":-1,\"threshold\":0.39935970306396484,\"type\":5,\"w\":1,\"x\":13,\"y\":14},{\"h\":1,\"polarity\":1,\"threshold\":0.6061859130859375,\"type\":3,\"w\":4,\"x\":10,\"y\":9},{\"h\":1,\"polarity\":-1,\"threshold\":0.5176620483398438,\"type\":5,\"w\":1,\"x\":7,\"y\":5},{\"h\":1,\"polarity\":-1,\"threshold\":-0.6722579002380371,\"type\":1,\"w\":2,\"x\":10,\"y\":0},{\"h\":1,\"polarity\":1,\"threshold\":-1.6933403015136719,\"type\":3,\"w\":6,\"x\":10,\"y\":17},{\"h\":1,\"polarity\":1,\"threshold\":2.35870361328125,\"type\":3,\"w\":10,\"x\":0,\"y\":15},{\"h\":1,\"polarity\":1,\"threshold\":-0.3822765350341797,\"type\":5,\"w\":1,\"x\":7,\"y\":5}],\"weights\":[1.0400972366333008,0.9411505460739136,0.8484309911727905,0.755580484867096,0.7047975659370422,0.7734078764915466,0.7072058916091919,0.6739394068717957,0.5927983522415161,0.5669746398925781,0.5272623896598816,0.5185450315475464,0.6354384422302246,0.5989407896995544,0.5919733047485352,0.610155463218689,0.5545832514762878,0.5710381269454956,0.5113619565963745,0.47504723072052,0.4721526801586151,0.5524135828018188,0.5512796640396118,0.5312499403953552,0.4875601530075073,0.6214157938957214,0.5233892798423767,0.5247217416763306,0.5227070450782776,0.5456815958023071,0.48451247811317444,0.5571720600128174,0.4724200665950775,0.5042135715484619,0.500564694404602,0.5067459344863892,0.5258789658546448,0.4820851683616638,0.5040133595466614,0.4942642152309418]}]}");

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
			// need to confirm that we're not reversing the order of WCs - may need to use insert()
			strongClassifier.weakClassifiers.push_back(weakClassifier);
			strongClassifier.weights.push_back(ccJSON["strongClassifiers"][i]["weights"][j]);
		}
		// need to confirm that we're not reversing the order of WCs - may need to use insert()
		sc.push_back(strongClassifier);
	}

	CascadeClassifier cc(ccJSON["baseResolution"], sc);

	float step = 2.0;
	float delta = 2.0;
	std::vector<std::array<int, 3>> roi;

	while (cc.baseResolution < w && cc.baseResolution < h) {
		for (int y = 0; y < h - cc.baseResolution; y += step * delta) {
			for (int x = 0; x < w - cc.baseResolution; x += step * delta) {
				float sum = integral.getRectangleSum(x, y, cc.baseResolution, cc.baseResolution);
				float squaredSum = integralSquared.getRectangleSum(x, y, cc.baseResolution, cc.baseResolution);
				float area = std::pow(cc.baseResolution, 2);
				float mean = sum / area;
				float sd = std::sqrt(squaredSum / area - std::pow(mean, 2));
				bool c = cc.classify(integral, x, y, mean, sd);
				
				if (c) {
					std::array<int, 3> bounding = {x, y, cc.baseResolution};
					roi.push_back(bounding);
				}
			}
		}
		cc.scale(step);
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

	auto boxes = nonMaxSuppression(roi, 0.1, 30);

	for (int i = 0; i < boxes.size(); i += 1) {
		EM_ASM({
			outputOverlayCtx.strokeStyle = '#77ff33';
			outputOverlayCtx.lineWidth = 4;
			outputOverlayCtx.beginPath();
			outputOverlayCtx.rect($0, $1, $2, $2); 
			outputOverlayCtx.stroke();
		}, boxes[i][0], boxes[i][1], boxes[i][2]);
	}

	return 1;
}

int main() {
	std::cout << "Made with Wasmface\n";
	return 0;
}

#ifdef __cplusplus
}
#endif