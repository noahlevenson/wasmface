const wildcardCanvas = document.getElementById("wildcard-canvas");
const jenniferlopezCanvas = document.getElementById("jenniferlopez-canvas");
const lancebassCanvas = document.getElementById("lancebass-canvas");
const noahlevensonCanvas = document.getElementById("noahlevenson-canvas");
const pikachuCanvas = document.getElementById("pikachu-canvas");
const robocopCanvas = document.getElementById("robocop-canvas");
const pizzaCanvas = document.getElementById("pizza-canvas");
const richardstallmanCanvas = document.getElementById("richardstallman-canvas");

const jenniferlopez = document.getElementById("jenniferlopez");
const lancebass = document.getElementById("lancebass");
const noahlevenson = document.getElementById("noahlevenson");
const pikachu = document.getElementById("pikachu");
const robocop = document.getElementById("robocop");
const pizza = document.getElementById("pizza");
const richardstallman = document.getElementById("richardstallman");

wildcardCanvas.width = 240;
wildcardCanvas.height = 240;

jenniferlopezCanvas.width = 240;
jenniferlopezCanvas.height = 240;

lancebassCanvas.width = 240;
lancebassCanvas.height = 240;

noahlevensonCanvas.width = 240;
noahlevensonCanvas.height = 240;

pikachuCanvas.width = 240;
pikachuCanvas.height = 240;

robocopCanvas.width = 240;
robocopCanvas.height = 240;

pizzaCanvas.width = 240;
pizzaCanvas.height = 240;

richardstallmanCanvas.width = 240;
richardstallmanCanvas.height = 240;

const wildcardCtx = wildcardCanvas.getContext("2d");
const jenniferlopezCtx = jenniferlopezCanvas.getContext("2d");
const lancebassCtx = lancebassCanvas.getContext("2d");
const noahlevensonCtx = noahlevensonCanvas.getContext("2d");
const pikachuCtx = pikachuCanvas.getContext("2d");
const robocopCtx = robocopCanvas.getContext("2d");
const pizzaCtx = pizzaCanvas.getContext("2d");
const richardstallmanCtx = richardstallmanCanvas.getContext("2d");

const wildcardResult = document.getElementById("wildcard-result");
const jenniferlopezResult = document.getElementById("jenniferlopez-result");
const lancebassResult = document.getElementById("lancebass-result");
const noahlevensonResult = document.getElementById("noahlevenson-result");
const pikachuResult = document.getElementById("pikachu-result");
const robocopResult = document.getElementById("robocop-result");
const pizzaResult = document.getElementById("pizza-result");
const richardstallmanResult = document.getElementById("richardstallman-result");

window.onload = () => {	
	wildcardCtx.drawImage(wildcard, 0, 0, 240, 240);
	jenniferlopezCtx.drawImage(jenniferlopez, 0, 0, 240, 240);
	lancebassCtx.drawImage(lancebass, 0, 0, 240, 240);
	noahlevensonCtx.drawImage(noahlevenson, 0, 0, 240, 240);
	pikachuCtx.drawImage(pikachu, 0, 0, 240, 240);
	robocopCtx.drawImage(robocop, 0, 0, 240, 240);
	pizzaCtx.drawImage(pizza, 0, 0, 240, 240);
	richardstallmanCtx.drawImage(richardstallman, 0, 0, 240, 240);
}

function go() {
	classify(wildcardCtx, wildcardResult);
	classify(jenniferlopezCtx, jenniferlopezResult);
	classify(lancebassCtx, lancebassResult);
	classify(noahlevensonCtx, noahlevensonResult);
	classify(pikachuCtx, pikachuResult);
	classify(robocopCtx, robocopResult);
	classify(pizzaCtx, pizzaResult);
	classify(richardstallmanCtx, richardstallmanResult);
}

function classify(context, outputDiv) {
	const inputImgData = context.getImageData(0, 0, 240, 240);
	const inputBuf = Module._malloc(inputImgData.data.length);
	Module.HEAPU8.set(inputImgData.data, inputBuf);
	const c = Module.ccall("isFace", "number", ["number"], [inputBuf]);
	outputDiv.style.color = c === 1 ? "#00cc00" : "#ff0000";
	outputDiv.innerHTML = c === 1 ? "YES" : "NO";
	Module._free(inputBuf);
}