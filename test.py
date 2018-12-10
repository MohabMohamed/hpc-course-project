from mpi4py import MPI 
import imageio
import numpy as np
from Sobel import Sobel


comm = MPI.COMM_WORLD 

size = comm.Get_size()
rank = comm.Get_rank()
path = ''
if rank == 0:
    path = input('enter image\'s path : ')

s = Sobel(path)
s.Compute()