from mpi4py import MPI
import numpy as np
import imageio


class Sobel:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        itemsize = MPI.DOUBLE.Get_size()
        imshape = None
        if self.rank == 0:
            path = ''
            path = input('enter image\'s path : ')
            im = imageio.imread(path).astype(dtype='d')
            im = np.pad(im, pad_width=((1, 1), (1, 1), (0, 0)),
                        mode='constant', constant_values=0)
            imsize = im.size
            imshape = im.shape
            nbyte = imsize * itemsize

        else:
            nbyte = 0
        imshape = self.comm.bcast(imshape, root=0)

        win = MPI.Win.Allocate_shared(nbyte, itemsize, comm=self.comm)

        buf, itemsize = win.Shared_query(0)
        assert itemsize == MPI.DOUBLE.Get_size()
        imshape[0]
        self.arr = np.ndarray(buffer=buf, dtype='d', shape=imshape)

        if self.rank == 0:
            np.copyto(dst=self.arr, src=im)

        self.comm.Barrier()
        self.filter_v = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
        self.filter_h = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])
        print('init done p#',self.rank)

    def Sobel_v(self, start, end):
        '''
        start = start of the image  should be 1 for first index
        end = end of the image  should be image height -1 for last index
        '''
        print('Sobel V from P#',self.rank,' start : ',start,' end : ',end)
        res = np.empty(shape=(end-start, self.arr.shape[1]-2, self.arr.shape[2]), dtype='d')
        for i in range(start, end):
            for j in range(1, self.arr.shape[1]-1):
                for z in range(self.arr.shape[2]):
                    res[i-start, j-1, z] = np.sum(np.multiply(
                        self.arr[i-1:i+2, j-1:j+2, z], self.filter_v))
        print('sobel V end P#',self.rank)
        self.comm.send(res, dest=0, tag=1)

    def Sobel_h(self, start, end):
        '''
        start = start of the image  should be 1 for first index
        end = end of the image  should be image height -1 for last index
        '''
        print('Sobel H from P#',self.rank,' start : ',start,' end : ',end)
        res = np.empty(shape=(end-start, self.arr.shape[1]-2,self.arr.shape[2]),dtype='d')
        for i in range(start, end):
            for j in range(1, self.arr.shape[1]-1):
                for z in range(self.arr.shape[2]):
                    res[i-start, j-1, z] = np.sum(np.multiply(self.arr[i-1:i+2, j-1:j+2, z],self.filter_h))
        print('sobel H end P#',self.rank)
        self.comm.send(res,dest=0,tag=2)

    def __get_start_end_padded(self,rank):
        start = (rank-1) * int(self.arr.shape[0]/(self.size-1))
        if rank == 1 :
            start+=1
        if rank != self.size - 1:
            end = (rank) * int(self.arr.shape[0]/(self.size-1))
        else:
            end =int( self.arr.shape[0]-2)
        return (start,end)


    def Compute(self):
        if self.rank == 0:
            res_v=np.empty(shape=(self.arr.shape[0]-2,self.arr.shape[1]-2,self.arr.shape[2]),dtype='d')
            res_h=np.empty(shape=(self.arr.shape[0]-2,self.arr.shape[1]-2,self.arr.shape[2]),dtype='d')
            print('res_v.shape = ',res_v.shape)
            for i in range(1,self.size):
                start ,end = self.__get_start_end_padded(i)
                res_v[start:end,:,:]=self.comm.recv(source=i,tag=1)
                res_h[start:end,:,:]=self.comm.recv(source=i,tag=2)
                print(i,' heeeh')
            final_image = np.absolute(np.add(res_h,res_v))
            imageio.imsave('result.jpg',final_image.astype(dtype=np.uint8))
        else:
            start,end = self.__get_start_end_padded(self.rank)
            print('rank ',self.rank,)
            self.Sobel_v(start,end)
            self.Sobel_h(start,end)
        
