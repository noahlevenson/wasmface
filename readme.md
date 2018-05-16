### wasmface
Fast WebAssembly face detection for HTML5 canvas

Wasmface is a fast computer vision library for detecting human faces and other objects within an HTML5 canvas element.

It's a WebAssembly implementation of the [Viola-Jones framework](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf) for rapid object detection, written in C++ for the Emscripten compiler.

For JavaScript developers: You don't need to know C++ or wasm to use Wasmface! Wasmface provides JavaScript wrappers for its C++ function calls which handle memory management and variable binding. Wasmface cascade classifier models are distributed in JSON format.

For C++ developers: The Wasmface engine has zero dependencies and is written from scratch in C++17. Happy optimizing!

Wasmface comes with a cascade classifier [model](https://github.com/noahlevenson/wasmface/src/models/human-face.js) trained to detect faces. It's a 20-layer cascade consisting of 300 features. The model was trained on ~13,000 positive examples from [LFW-a](https://www.openu.ac.il/home/hassner/data/lfwa/) and ~10,000 negative examples generated from stock photos.

You can create your own models using the included native executable tool, [wf-train-cascade](https://github.com/noahlevenson/wasmface/src/train-cascade.cpp), which implements the training phase of the framework. Viola-Jones is particularly suited for detecting human faces, but TK TK.

<INSTRUCTIONS AND DOCS HERE>

<DEPENDENCIES>

I developed Wasmface as part of my research at [Recurse Center](https://recurse.com), which focused on WebAssembly in the domain of computer vision.

<CONTACT ETC>