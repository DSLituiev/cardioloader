this repository shows loading of a radiology dataset with annotations (contours)

we assume that slices without annotations can be ignored 
(i.e. they are either not real true negatives, or we can ignore true negatives for this task)

## Dependencies
- pillow
- numpy
- pandas
- pytorch (in task 2)
- matplotlib (notebooks only)


## Presentation 

please refer to plain python files for final code and unit tests.

- [task1.py](task1.py)

- [task2.py](task2.py)

- [test_task2.py](test_task2.py)

prototyping and visualization is performed in Jupyter notebooks

## Notes:
### Task 1:

> How did you verify that you are parsing the contours correctly?
 
First I assumed the library code is solid, and I decided against writing unit tests for it. 
I verified the code visually, by overlaying contour and the mask 
![overlay](task1.png)
(for code, see last cells of the [task1 notebook](task1.ipynb))

> What changes did you make to the code, if any, in order to integrate it into our production code base?

I did not have to make any changes to the provided `parsing.py` prototype.

### Task 2

> Did you change anything from the pipelines built in Parts 1 to better streamline the pipeline built in Part 2? If so, what? If not, is there anything that you can imagine changing in the future?

I wrapped the DataFrame and a reading function from previous task into a HeartDataset class,
so that it is easier to work with it downstream, specifically to use it with DataLoader class.

> How do you/did you verify that the pipeline was working correctly?

i. Manually ensured the output conforms to the specifications;

ii. Wrote unit tests for output format+shape and random shuffling

iii. Inspected output visually
![overlay](task2.png)
for code, see last cell of the [task2 notebook](task2.ipynb)

> Given the pipeline you have built, can you see any deficiencies that you would change if you had more time? If not, can you think of any improvements/enhancements to the pipeline that you could build in?

I would first convert the contours to run-length encoding (RLE) instead of constructing masks from contours on-flight
to accelerate loading time. I would also explore other optimization with memory mapping or any framework-specific
optimized data formats. I also found out that DataLoader has issues with random seed setting for shuffling while
writting a unit test -- another thing to fix.

