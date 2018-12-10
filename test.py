from mpi4py import MPI 
import imageio
import numpy as np
from Sobel import Sobel


comm = MPI.COMM_WORLD 

size = comm.Get_size()
rank = comm.Get_rank()


s = Sobel()
s.Compute()