let myWasmface, loopId;

const outputOverlayCanvas = document.getElementById("output-overlay-canvas");
const inputCanvas = document.getElementById("input-canvas");
const video = document.getElementById("video");
const stepOutput = document.getElementById("scale-step-output");
const deltaOutput = document.getElementById("delta-output");
const stepSlider = document.getElementById("scale-step");
const deltaSlider = document.getElementById("delta");
const overlapOutput = document.getElementById("overlap-output");
const neighborOutput = document.getElementById("neighbor-output");
const overlapSlider = document.getElementById("overlap");
const neighborSlider = document.getElementById("neighbor");

const applypp = document.getElementById("applypp");
let pp = 1;

const outputOverlayCtx = outputOverlayCanvas.getContext("2d");
const inputCtx = inputCanvas.getContext("2d");

navigator.mediaDevices.getUserMedia({video: true}).then(stream => {
	video.srcObject = stream;
	video.onloadedmetadata = () => {
		outputOverlayCanvas.width = video.videoWidth;
		outputOverlayCanvas.height = video.videoHeight;
		inputCanvas.width = video.videoWidth;
		inputCanvas.height = video.videoHeight;
	}
}).catch(err => alert(err));

stepSlider.oninput = () => stepOutput.innerHTML = stepSlider.value;
deltaSlider.oninput = () => deltaOutput.innerHTML = deltaSlider.value;
overlapSlider.oninput = () => overlapOutput.innerHTML = overlapSlider.value;
neighborSlider.oninput = () => neighborOutput.innerHTML = neighborSlider.value;
applypp.onchange = () => pp = applypp.checked ? 1 : 0;

function start() {
	if (!myWasmface) myWasmface = new Wasmface(humanFace);
	if (!loopId) loopId = window.requestAnimationFrame(update);
		
	function update() {
		inputCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
		outputOverlayCtx.clearRect(0, 0, outputOverlayCanvas.width, outputOverlayCanvas.height);
		const boxes = myWasmface.detect(inputCtx, pp, overlapSlider.value, neighborSlider.value, stepSlider.value, deltaSlider.value);
		
		for (let i = 0, len = boxes.length; i < len; i += 1) {
			outputOverlayCtx.strokeStyle = "#77ff33";
			outputOverlayCtx.lineWidth = 4;
			outputOverlayCtx.beginPath();
			outputOverlayCtx.rect(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][2]); 
			outputOverlayCtx.stroke();
		}

		loopId = window.requestAnimationFrame(update);
	}
}

function stop() {
	window.cancelAnimationFrame(loopId);
	loopId = null;
	outputOverlayCtx.clearRect(0, 0, outputOverlayCanvas.width, outputOverlayCanvas.height);
}