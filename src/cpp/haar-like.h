#pragma once

class Haarlike {
	public:
		Haarlike(int x, int y, int w, int h, int type);
		Haarlike();
		void scale(float factor);
		int x;
		int y;
		int w;
		int h;
		int type;
};