def fib(n):
    return cdef_fib(n)

cdef int cdef_fib(int n):
    """Print the Fibonacci series up to n."""
    cdef int a=0, b=1
    while b < n:
        a, b = b, a + b