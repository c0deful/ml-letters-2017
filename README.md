# LETTERS IMAGE RECOGNITION

This script uses tensorflow's `DNNCLassifier` to recognise capital letters from some corrupted data.
This was a task during University of Warsaw's Machine Learning course in 2017.

## Description of the data

Letters have the following attributes:

- `x.bar`: mean x of on pixels in box (integer)
- `y.bar`: mean y of on pixels in box (integer)
- `xybar`: mean x y correlation (integer)
- `x.edge`: mean edge count left to right (integer)
- `y.edge`: mean edge count bottom to top (integer)
- `a6,...,a15`: some linear combinations of 
    - horizontal position of box
    - vertical position of box
    - width of box
    - height of box
    - total number of on pixels
    - mean x variance
    - mean y variance
    - mean of x * x * y
    - mean of x * y * y
    - correlation of x-edge with y
    - correlation of y-edge with x
- `letter`: capital letter (26 values from A to Z)

All numeric values have been scaled to the range 0..20. The data (including test sets) are of reduced quality. In particular, values of attributes a6,...,a15 may be missing. 


## Requirements
The script requires python 2.7 and pip to run. Install dependencies using command
`pip install -r requirements.txt`

## Running full training and testing
To run interactive script for training and testing the classifier, execute
`python letters.py`
and provide training, testing and output file paths when prompted.

Example files can be found in the `data/` directory