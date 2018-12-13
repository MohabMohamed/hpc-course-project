from mpi4py import MPI
import numpy as np
import imageio
import time

class Sobel:
    def __init__(self,path):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        itemsize = MPI.DOUBLE.Get_size()
        imshape = None
        gray_scaled_flag = False
        if self.rank == 0:
            self.time = time.process_time()
            im = imageio.imread(path).astype(dtype='d')
            pad = None
            if len(im.shape) ==2:
                pad=((1,1),(1,1))
                gray_scaled_flag =True
            else:
                pad=((1, 1), (1, 1), (0, 0))
            im = np.pad(im, pad_width=pad,
                        mode='constant', constant_values=0)
            imsize = im.shape[0]*im.shape[1]
            imshape = im.shape
            nbyte = imsize * itemsize

        else:
            nbyte = 0
            im=None
        imshape = self.comm.bcast(imshape, root=0)
        gray_scaled_flag= self.comm.bcast(gray_scaled_flag,root=0)
        win = MPI.Win.Allocate_shared(nbyte, itemsize, comm=self.comm)

        buf, itemsize = win.Shared_query(0)
        assert itemsize == MPI.DOUBLE.Get_size()
        
        self.arr = np.ndarray(buffer=buf, dtype='d', shape=imshape[0:2])
        if gray_scaled_flag ==False:
            self.gray_scale(im)
        else:
            if self.rank == 0:
                np.copyto(dst=self.arr, src=im)

        self.comm.Barrier()
        # if self.rank>0:
        #     self.noise_remove()
        # else:
        #     for i in range(1,self.size-1):
        #         start ,end = self.get_start_end_padded(i)
        #         self.arr[start+2:end+2,:]=self.comm.recv(source=i,tag=0)
        #     self.arr[end+2:,:]=self.comm.recv(source=self.size-1,tag=0)
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
        # if self.rank == 0:
        #     imageio.imsave('gray_scaled.jpg',self.arr.astype(dtype=np.uint8))
        #print('init done p#',self.rank)

    def gray_scale(self,im):
        if self.rank ==0:
            di=im.shape
            start=0
            end=0
            #print('full : ',di)
            for i in range(1,self.size-1):
                start=(i-1)*(di[0]/(self.size-1))
                end=(i)*(di[0]/(self.size-1))
                #print('i = ',i,' start = ',start,' end = ',end)
                self.comm.send(obj=im[int(start):int(end),:,:],dest=i)
            self.comm.send(obj=im[int(end):int(di[0]),:,:],dest=self.size-1)
        else:

            arr=self.comm.recv(source=0)
            arrdi=arr.shape
            start=int((self.rank-1)*(self.arr.shape[0]/(self.size-1)))
            #print('from self.rank ',self.rank,' di = ',arrdi)
            for i in range(arrdi[0]):
                for j in range(arrdi[1]):
                    self.arr[start+i,j]=(arr[i,j,0]/3+arr[i,j,1]/3+arr[i,j,2]/3)
                    
    def noise_remove(self):
        start,end= self.get_start_end_padded(self.rank)
        mean_filter = np.array([
            [1/9,1/9,1/9],
            [1/9,1/9,1/9],
            [1/9,1/9,1/9]
        ])
        res = np.zeros(shape=(end-start, self.arr.shape[1]), dtype='d')
        for i in range(start, end):
            for j in range(1, self.arr.shape[1]-1):
                res[i-start, j-1] = np.sum(np.multiply(
                    self.arr[i-1:i+2, j-1:j+2], mean_filter))
        self.comm.send(res, dest=0, tag=0)

    def Sobel_v(self, start, end):
        '''
        start = start of the image  should be 1 for first index
        end = end of the image  should be image height -1 for last index
        '''
        #print('Sobel V from P#',self.rank,' start : ',start,' end : ',end)
        res = np.empty(shape=(end-start, self.arr.shape[1]-2), dtype='d')
        for i in range(start, end):
            for j in range(1, self.arr.shape[1]-1):
                res[i-start, j-1] = self.threshhold(np.sum(np.multiply(
                    self.arr[i-1:i+2, j-1:j+2], self.filter_v)),20)
        #print('sobel V end P#',self.rank)
        self.comm.send(res, dest=0, tag=1)

    def Sobel_h(self, start, end):
        '''
        start = start of the image  should be 1 for first index
        end = end of the image  should be image height -1 for last index
        '''
        #print('Sobel H from P#',self.rank,' start : ',start,' end : ',end)
        res = np.empty(shape=(end-start, self.arr.shape[1]-2),dtype='d')
        for i in range(start, end):
            for j in range(1, self.arr.shape[1]-1):
                res[i-start, j-1] = self.threshhold(np.sum(np.multiply(self.arr[i-1:i+2, j-1:j+2],self.filter_h)),20)
        #print('sobel H end P#',self.rank)
        self.comm.send(res,dest=0,tag=2)

    def get_start_end_padded(self,rank):
        start = (rank-1) * int(self.arr.shape[0]/(self.size-1))
        if rank == 1 :
            start+=1
        if rank != self.size - 1:
            end = (rank) * int(self.arr.shape[0]/(self.size-1))
        else:
            end =int( self.arr.shape[0]-2)
        return (start,end)
    def threshhold(self,pix_val,thresh_val):
        if pix_val<thresh_val:
            return 0
        else:
            return 255

    def Compute(self):
        if self.rank == 0:
            res_v=np.empty(shape=(self.arr.shape[0]-2,self.arr.shape[1]-2),dtype='d')
            res_h=np.empty(shape=(self.arr.shape[0]-2,self.arr.shape[1]-2),dtype='d')
            #print('res_v.shape = ',res_v.shape)
            for i in range(1,self.size):
                start ,end = self.get_start_end_padded(i)
                res_v[start:end,:]=self.comm.recv(source=i,tag=1)
                res_h[start:end,:]=self.comm.recv(source=i,tag=2)
                #print(i,' heeeh')
            final_image = np.absolute(np.add(res_h,res_v))
            print('Sobel excution time : ',time.process_time()-self.time)
            imageio.imsave('result.jpg',final_image.astype(dtype=np.uint8))
        else:
            start,end = self.get_start_end_padded(self.rank)
            #print('rank ',self.rank,)
            self.Sobel_v(start,end)
            self.Sobel_h(start,end)
        
