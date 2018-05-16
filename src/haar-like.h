#pragma once

class Haarlike {
	public:
		Haarlike(int s, int x, int y, int w, int h, int type);
		Haarlike();
		void scale(float factor);
		int s;
		int x;
		int y;
		int w;
		int h;
		int type;
};