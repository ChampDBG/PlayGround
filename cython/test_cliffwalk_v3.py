import timeit, tqdm

REPEAT = 30
NUMBER = 3000

if __name__ == '__main__':
    print('='*10 + ' pure python   ' + '='*10)
    print('Record %s times of executing cliffwalk %s times' % (REPEAT, NUMBER))
    py_data = []
    for i in tqdm.tqdm(range(REPEAT), ncols = 70):
        py_data.append(timeit.timeit(stmt = 
            'from cliffwalk_v3 import py_cliffwalk as pc\n' 
            'runQ = pc.cliffwalk(1000, "qlearn")\n' 
            'runQ.qlearn()\n'
            'runS = pc.cliffwalk(1000, "sarsa")\n'
            'runS.sarsa()', number = NUMBER))
    print('the fastest result is %s' % min(py_data))
    print('the slowest result is %s' % max(py_data))
    print('the average result is %s' % (sum(py_data)/REPEAT))

    print('='*10 + ' naive cython   ' + '='*10)
    print('Record %s times of executing cliffwalk %s times' % (REPEAT, NUMBER))
    naive_cy_data = []
    for i in tqdm.tqdm(range(REPEAT), ncols = 70):
        naive_cy_data.append(timeit.timeit(stmt = 
            'from cliffwalk_v3 import ncy_cliffwalk as ncc\n' 
            'run = ncc.cliffwalk(1000, "qlearn")\n' 
            'run.qlearn()\n'
            'run = ncc.cliffwalk(1000, "sarsa")\n' 
            'run.sarsa()', number = NUMBER))
    print('the fastest result is %s' % min(naive_cy_data))
    print('the slowest result is %s' % max(naive_cy_data))
    print('the average result is %s' % (sum(naive_cy_data)/REPEAT))

    print('='*10 + ' cython   ' + '='*10)
    print('Record %s times of executing cliffwalk %s times' % (REPEAT, NUMBER))
    cy_data = []
    for i in tqdm.tqdm(range(REPEAT), ncols = 70):
        cy_data.append(timeit.timeit(stmt = 
            'from cliffwalk_v3 import cy_cliffwalk_v3 as cc\n' 
            'run = cc.exec_cliffwalk(1000, "qlearn")\n' 
            'run = cc.exec_cliffwalk(1000, "sarsa")' , number = NUMBER))
    print('the fastest result is %s' % min(cy_data))
    print('the slowest result is %s' % max(cy_data))
    print('the average result is %s' % (sum(cy_data)/REPEAT))
