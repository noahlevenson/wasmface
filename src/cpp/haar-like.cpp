#include <iostream>

#include "haar-like.h"

// TODO: I don't think subwindow size is ever used
Haarlike::Haarlike(int s, int x, int y, int w, int h, int type) {
	this->s = s; 
	this->x = x; 
	this->y = y; 
	this->w = w; 
	this->h = h; 
	this->type = type; 
}

Haarlike::Haarlike() {

}

void Haarlike::scale(float factor) {
	this->w *= factor;
	this->h *= factor;
	this->x *= factor;
	this->y *= factor;
}