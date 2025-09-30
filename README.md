# NN--Circle_and_Fractal
0.9998 makes a pretty good circle; will 1.002 make a decent fractal?

##
### Problem
Determine the shape with an area of 1 and the minimum perimeter possible.


<b>Method</b>:

Parameterize x(s) and y(s), for s from 0 to 1.

Use Neural Networks to express x(s) and y(s).

Use a loss term to enforce x(0) = x(1) = 0 and y(0) = y(1) = 0.

Use a loss term to enforce Area = 1.

Use a loss term to minimize the perimeter; the perimeter should be a factor (0.9998) times the perimeter in the previous training epoch.

Pre-train the model to represent the 4 corners of a square of length 1 centered at the origin. This pre-trained scenario violates the x(0) = 0, y(0) = 0 condition and is only used to decrease the likelihood of perimeter self-intersections that would lead to negative area.


<b>Result</b>:

The shape very nearly approximates a circle. The perimeter is 5.013 units, as expected.
Solved in _circle_v2.py_.


##
### Problem
Determine the shape with an area of 1 and the maximum perimeter possible.


<b>Method</b>:

Parameterize x(s) and y(s), for s from 0 to 1.

Use Neural Networks to express x(s) and y(s).

Use a loss term to enforce continuity and derivative continuity at x(0) = x(1), y(0) = y(1).

Use a loss term to enforce Area = 1.

Use a loss term to maximize the perimeter; the perimeter should be a factor (1.001) times the perimeter in the previous training epoch.

Use a loss term to bound the shape inside the square from (-2,-2) to (2,2).

Pre-train the model to represent the 4 corners of a square of length 1 centered at the origin.


<b>Result</b>:

The model did not train well. Every training resulted in many points with y(s) = -1 for no enforced reason, the perimeter didn’t increase much, and then the shape developed self-intersections causing the area (shoelace formula) to go to “NaN”. Did not resolve/complete.
Final attempt in _circle_v3.py_.

