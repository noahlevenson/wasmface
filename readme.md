### wasmface
Fast WebAssembly face detection for HTML5 canvas

Demo: http://wasmface.noahlevenson.com

Wasmface is a fast computer vision library for detecting human faces and other objects in an HTML5 canvas element.

It's a WebAssembly implementation of the [Viola-Jones framework](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf) for rapid object detection. It's written in C++ for the Emscripten compiler.

For JavaScript developers: You don't need to know C++ to use Wasmface! Wasmface provides JavaScript wrappers that handle memory management and variable binding. Wasmface cascade classifier models are formatted as JSON.

For C++ developers: The Wasmface engine has zero dependencies and is written from scratch in C++17. Optimize it for your use case!

Wasmface comes with [human-face.js](https://github.com/noahlevenson/wasmface/src/models/human-face.js), a cascade classifier model trained to detect faces. It's a 20-layer cascade consisting of 300 features. The model was trained on ~13,000 positive examples from [LFW-a](https://www.openu.ac.il/home/hassner/data/lfwa/) and ~10,000 negative examples generated from stock photos.

You can also create your own models using [wasmface-trainer](https://github.com/noahlevenson/wasmface/src/wasmface-trainer.cpp), a native executable tool which implements the training phase of the framework. Viola-Jones is particularly suited for detecting human faces, but TK TK.

Features:

* Absolutely no GPU acceleration :stuck_out_tongue_winking_eye: Wasmface is an experiment in fast sequential processing
* Detection merging via non-maximum suppression
* Variance normalization (pre-applied during training, post-applied during detection)
* 5 types of Haar-like features
* Optimized for HTML5 pseudograyscale (luma in 4th byte)
* Integral images, feature scaling, and everything else described in the 2001 paper

<INSTRUCTIONS AND DOCS HERE>

<DEPENDENCIES>

I developed Wasmface as part of my research at [Recurse Center](https://recurse.com), which focused on WebAssembly in the domain of computer vision.

<CONTACT ETC>