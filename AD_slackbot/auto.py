#from prep.source import *
from astropy.time import Time
import tempfile
import os
import glob
from build_rec import post as pst
from build_rec import build_rec as bs
import time



def run(post=True):
    ps= []
    if post:
        for l in ['ANT2023k4niot9ojjs2', 'ANT20231cmvku2jk9km']:
            ant = bs(antaresID=l)
            ps.append(ant.string)
        ps = '\n'.join(ps)
        pst(ps,channel='D05R7RK4K8T') # LAISS_AD_bot channel for testing

    return 0


# def run_sched(sep= 86400): # 24 hours in seconds
#     while True:
#         t1 = time.monotonic()
#         run()
#         t2 = time.monotonic()
#         td = t2-t1
#         time.sleep(sep-td) # calculation offset so it runs at the same time every day
#     return 0


if __name__ == '__main__':
    print('running')
    run()
    #run_sched()

def main():
    print('running')
    run()
    #run_sched()
    return 0