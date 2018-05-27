/**
 * Constructor
 * @param {JSON Object} model A wasmface cascade classifier model
 */
function Wasmface(model) {
	const strptr = Module.allocate(intArrayFromString(JSON.stringify(model)), "i8", 0);
	this.ptr = Module.ccall("create", "number", ["number"], [strptr]);
	Module._free(strptr);
}

/**
 * Manually deallocate the heap memory associated with a cascade classifier 
 */
Wasmface.prototype.destroy = function() {
	Module.ccall("destroy", null, ["number"], [this.ptr]);
}

/**
 * Detect objects in an HTML5 canvas
 * @param  {Canvas context object} ctx     2D context for the canvas 
 * @param  {Number}                pp      1 for post processing, 0 for no post processing
 * @param  {Number}                othresh Overlap threshold for post processing
 * @param  {Number}                nthresh Neighbor threshold for post processing
 * @param  {Number}                step    Detector scale step to apply
 * @param  {Number}                delta   Detector sweep delta to apply
 * @return {Array}                         2D array of 1:1 aspect ratio bounding boxes [x, y, s] where s = width and height
 */
Wasmface.prototype.detect = function(ctx, pp = 1, othresh = 0.3, nthresh = 10, step = 2.0, delta = 2.0) {
	const inputImgData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
	const inputBuf = Module._malloc(inputImgData.data.length);
	Module.HEAPU8.set(inputImgData.data, inputBuf);

	const ptr = Module.ccall("detect", "number", 
                             ["number", "number", "number", "number", "number", "number", "number", "number", "number"], 
                             [inputBuf, ctx.canvas.width, ctx.canvas.height, this.ptr, step, delta, pp, othresh, nthresh])
	                         / Uint16Array.BYTES_PER_ELEMENT;

	const len = Module.HEAPU16[ptr];
	const boxes = [];
	for (let i = 1; i < len; i += 3) {
		const box = [Module.HEAPU16[ptr + i], Module.HEAPU16[ptr + i + 1], Module.HEAPU16[ptr + i + 2]];
		boxes.push(box);
	}

	Module._free(inputBuf);
	Module._free(ptr);

	return boxes;
}