function Wasmface(model) {
	const strptr = Module.allocate(intArrayFromString(model), "i8", 0);
	this.ptr = Module.ccall("create", "number", ["number"], [strptr]);
	Module._free(strptr);
}

Wasmface.prototype.detect = function(ctx, step = 2.0, delta = 2.0, pp = 1, othresh = 0.3, nthresh = 10) {
	const inputImgData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
	const inputBuf = Module._malloc(inputImgData.data.length);
	Module.HEAPU8.set(inputImgData.data, inputBuf);

	const ptr = Module.ccall("detect", "number", 
						   	 ["number", "number", "number", "number", "number", "number", "number", "number", "number"], 
						  	 [inputBuf, ctx.canvas.width, ctx.canvas.height, this.ptr, step, delta, pp, othresh, nthresh])
							 / Uint16Array.BYTES_PER_ELEMENT;

	len = Module.HEAPU16[ptr];
	const boxes = [];
	for (let i = 1; i < len; i += 3) {
		const box = [Module.HEAPU16[ptr + i], Module.HEAPU16[ptr + i + 1], Module.HEAPU16[ptr + i + 2]];
		boxes.push(box);
	}

	Module._free(inputBuf);
	Module._free(ptr);

	return boxes;
}