this repository shows loading of a radiology dataset with annotations (contours)

we assume that slices without annotations can be ignored 
(i.e. they are either not real true negatives, or we can ignore true negatives for this task)


## Presentation 

please refer to plain python files for final code and unit tests.

- [task1.py](task1.py)

- [task2.py](task2.py)

- [test_task2.py](test_task2.py)

prototyping and visualization is performed in Jupyter notebooks

## Notes:
### Task 1:

- How did you verify that you are parsing the contours correctly?
 
> First I assumed the library code is solid, and I decided against writing unit tests for it. 
> I verified the code visually, though by overlaying contour and the mask 
> (see last cells of the [task1 notebook](task1.ipynb))

- What changes did you make to the code, if any, in order to integrate it into our production code base?

> I did not have to make any changes to the provided `parsing.py` prototype.

### Task 2
- Did you change anything from the pipelines built in Parts 1 to better streamline the pipeline built in Part 2? If so, what? If not, is there anything that you can imagine changing in the future?
> Nothing so far

- How do you/did you verify that the pipeline was working correctly?
> i. Manually ensured the output conforms to the specifications;
> ii. Wrote unit tests for output format+shape and random shuffling
> iii. Inspected output visually, see last cell of the [task2 notebook](task2.ipynb)

- Given the pipeline you have built, can you see any deficiencies that you would change if you had more time? If not, can you think of any improvements/enhancements to the pipeline that you could build in?
> I would first convert the contours to run-length encoding (RLE) instead of constructing masks every time 
> to accelerate loading time. I would also explore other optimization with memory mapping or any framework-specific
> optimized data formats.




