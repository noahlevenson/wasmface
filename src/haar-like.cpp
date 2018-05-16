#include <iostream>

#include "haar-like.h"

// Constructor
// TODO: does subwindow size ever get used?
Haarlike::Haarlike(int s, int x, int y, int w, int h, int type) {
	this->s = s; // Subwindow size
	this->x = x; // X offset
	this->y = y; // Y offset
	this->w = w; // Width
	this->h = h; // Height
	this->type = type; // 1 - 5 : A - E
}

// Default constrctor - do we need this?
Haarlike::Haarlike() {

}

void Haarlike::scale(float factor) {
	this->w *= factor;
	this->h *= factor;
	this->x *= factor;
	this->y *= factor;
}