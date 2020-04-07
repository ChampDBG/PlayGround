import timeit, tqdm

## parameters
REPEAT = 100
NUMBER = 100000

if __name__ == "__main__":
    print('='*10 + ' pure python   ' + '='*10)
    print('Record 100 times of executing statement 100000 times')
    py_data = []
    for i in tqdm.tqdm(range(REPEAT), ncols = 70):
        py_data.append(timeit.timeit(stmt = 'from Fibonacci import py_fib\n' 'py_fib.fib(100000)', number = NUMBER))
    print('the fastest result is %s' % min(py_data))
    print('the slowest result is %s' % max(py_data))
    print('the average result is %s' % (sum(py_data)/REPEAT))
    
    print('='*10 + ' naive cython  ' + '='*10)
    print('Record %s times of executing statement %s times' % (REPEAT, NUMBER))
    cy_data = []
    for i in tqdm.tqdm(range(REPEAT), ncols = 70):
        cy_data.append(timeit.timeit(stmt = 'from Fibonacci import cy_fib\n' 'cy_fib.fib(100000)', number = NUMBER))
    print('the fastest result is %s' % min(cy_data))
    print('the slowest result is %s' % max(cy_data))
    print('the average result is %s' % (sum(cy_data)/REPEAT))
    
    print('='*10 + ' static cython ' + '='*10)
    print('Record 100 times of executing statement 100000 times')
    cdef_data = []
    for i in tqdm.tqdm(range(REPEAT), ncols = 70):
        cdef_data.append(timeit.timeit(stmt = 'from Fibonacci import cy_fib_static_type\n' 'cy_fib_static_type.fib(100000)', number = NUMBER))
    print('the fastest result is %s' % min(cdef_data))
    print('the slowest result is %s' % max(cdef_data))
    print('the average result is %s' % (sum(cdef_data)/REPEAT))

# the for loop could be change as timeit.repeat, but i want the progress bar effect.
# for information about timeit.repeat: https://docs.python.org/3/library/timeit.html#timeit.repeat
# reference for Cython: https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html#the-basics-of-cython