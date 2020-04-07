def fib(n):
    """Print the Fibonacci series up to n."""
    a, b = 0, 1
    while b < n:
        a, b = b, a + b