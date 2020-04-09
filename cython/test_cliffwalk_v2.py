import timeit, tqdm

REPEAT = 30
NUMBER = 3000

if __name__ == '__main__':
    print('='*10 + ' pure python   ' + '='*10)
    print('Record %s times of executing cliffwalk %s times' % (REPEAT, NUMBER))
    py_data = []
    for i in tqdm.tqdm(range(REPEAT), ncols = 70):
        py_data.append(timeit.timeit(stmt = 
        'from cliffwalk_v2 import py_cliffwalk\n' 
        'py_cliffwalk.qlearn()\n'
        'py_cliffwalk.sarsa()'
        , number = NUMBER))
    print('the fastest result is %s' % min(py_data))
    print('the slowest result is %s' % max(py_data))
    print('the average result is %s' % (sum(py_data)/REPEAT))

    print('='*10 + ' naive cython   ' + '='*10)
    print('Record %s times of executing cliffwalk %s times' % (REPEAT, NUMBER))
    naive_cy_data = []
    for i in tqdm.tqdm(range(REPEAT), ncols = 70):
        naive_cy_data.append(timeit.timeit(stmt = 
        'from cliffwalk_v2 import ncy_cliffwalk\n' 
        'ncy_cliffwalk.qlearn()\n'
        'ncy_cliffwalk.sarsa()\n', number = NUMBER))
    print('the fastest result is %s' % min(naive_cy_data))
    print('the slowest result is %s' % max(naive_cy_data))
    print('the average result is %s' % (sum(naive_cy_data)/REPEAT))

    print('='*10 + ' cython   ' + '='*10)
    print('Record %s times of executing cliffwalk %s times' % (REPEAT, NUMBER))
    cy_data = []
    for i in tqdm.tqdm(range(REPEAT), ncols = 70):
        cy_data.append(timeit.timeit(stmt = 
        'from cliffwalk_v2 import cy_cliffwalk\n' 
        'cy_cliffwalk.qlearn()\n'
        'cy_cliffwalk.sarsa()\n', number = NUMBER))
    print('the fastest result is %s' % min(cy_data))
    print('the slowest result is %s' % max(cy_data))
    print('the average result is %s' % (sum(cy_data)/REPEAT))