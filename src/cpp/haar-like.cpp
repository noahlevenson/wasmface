#include <iostream>

#include "haar-like.h"

/**
 * Constructor
 * @param {Int} x    X offset relative to subwindow
 * @param {Int} y    Y offset relative to subwindow
 * @param {Int} w    Constituent rectangle width
 * @param {Int} h    Constituent rectangle height
 * @param {Int} type Feature type 1-5 
 */
Haarlike::Haarlike(int x, int y, int w, int h, int type) {
	this->x = x; 
	this->y = y; 
	this->w = w; 
	this->h = h; 
	this->type = type; 
}

/**
 * Constructor
 */
Haarlike::Haarlike() {

}

/**
 * Destructively scale a Haarlike relative to its base resolution
 * @param {Float} factor The factor by which to scale
 */
void Haarlike::scale(float factor) {
	this->w *= factor;
	this->h *= factor;
	this->x *= factor;
	this->y *= factor;
}