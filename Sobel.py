from mpi4py import MPI 
import numpy as np 
import imageio 

comm = MPI.COMM_WORLD 

size = comm.Get_size()
rank = comm.Get_rank()

itemsize = MPI.DOUBLE.Get_size() 
imshape = None
if rank == 0:
    path=''
    path = input('enter image\'s path : ')
    im = imageio.imread(path).astype(dtype='d')
    im = np.pad(im,pad_width=((1,1),(1,1),(0,0)), mode='constant', constant_values=0)
    imsize = im.size
    imshape = im.shape
    nbyte = imsize * itemsize
 
else:
    nbyte = 0
imshape = comm.bcast(imshape,root=0)

win = MPI.Win.Allocate_shared(nbyte, itemsize, comm=comm)

buf, itemsize = win.Shared_query(0) 
assert itemsize == MPI.DOUBLE.Get_size() 
imshape[0]
arr = np.ndarray(buffer=buf, dtype='d', shape=imshape) 


if rank == 0:
    np.copyto(dst=arr,src=im)

comm.Barrier() 



