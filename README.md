this repository shows loading of a radiology dataset with annotations (contours)

we assume that slices without annotations can be ignored 
(i.e. they are either not real true negatives, or we can ignore true negatives for this task)

## Dependencies
- pillow
- numpy
- pandas
- pytorch (in task 2)
- matplotlib (notebooks only)
- opencv
- numba


## Presentation 

## Notes:
### Assignment 1
[see here](asgn1.md)

### Assignment 2

#### Important Observations
Not all i-contours are contained within o-contours (11 images out of 46 are affected).
Sometimes it is drawing error (1--3 pixel of i-contour are outside the o-contour in 2 images),
but often it is a gross error (200--500 pixels, all in 9 images from 'SC-HF-I-6' case).
X-Y axis flipping does not explain this error.



