#pragma once

struct MousePos
{
	float x;
	float y;

	bool avoid;

	MousePos(float X, float Y, bool Avoid)
	{
		x = X;
		y = Y;
		avoid = Avoid;
	}
};