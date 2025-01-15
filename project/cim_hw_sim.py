'''
File name: mnist_infer_int8_hw_sim.py
Descriptions: This python script implement quantized mnist-12 image classify neural network 
  by numpy & hardware model step by step and run on test sets to statistic top-1 accuracy. 
  Input image is (1,28,28) uint8 hand written decimal number, total 10000 samples. 
  Network have two conv layers and a flat layer. Weights data type is int8.
  This model can reach 98.95% accuracy
Author: Weijun
Date: 2024/12/03
Revision: 1.0.2
Revision History:
V1.0.2 - 2024/12/03 print loop counters values step by step for hardware circuit comparison
V1.0.1 - 2024/11/20 Exchange cim_conv2d h,g,v axis to well fit hardware behavior
V1.0.0 - 2024/11/12 Support hardware simulation with variables of feature map channels
V0.9.5 - 2024/10/28 Add hardware simulation model for data comparision 
V0.9.0 - 2024/07/03 Update bias calculation scheme in Flatten/Softmax layer
V0.8.0 - 2024/06/28 Initial version
'''

import os
import math
from tarfile import data_filter
import numpy as np

# Print image data, image should be in shape of (Height,Width)
def print_image_data(image):
    for row in image:
        pixel = "".join(["{:>4d}".format(int(i)) for i in row])
        print(pixel)

# Validate shape of feature map and conv core, and insert proper padding data
# fmap - feature maps in shape of (channel, y/row/height, x/column/width)
# core - convolution core in shape of (channel, y/row/height, x/column/width)
# pad - extra rows and columns of padding data to be inserted
# pair - if Ture, will insert padding data into both sides in y and x
# This padding scheme will affect to accuracy a little bit
def validate_fmap(fmap,core,pad,pair=True):
    assert(len(fmap.shape)==3 and len(core.shape)==3)
    assert(fmap.shape[1]>=core.shape[1] and fmap.shape[1]>=core.shape[1])
    #for slicex in fmap:
    s = fmap.shape
    if pad>0:
        r = np.zeros((s[0],pad,s[2]),dtype=fmap.dtype)
        #fmap = np.concatenate((r,fmap),axis=1) # same as hstack
        if(pair): fmap = np.hstack((r,fmap))
        fmap = np.hstack((fmap,r))

    s = fmap.shape
    if pad>0:
        c = np.zeros((s[0],s[1],pad),dtype=fmap.dtype)
        if(pair): fmap = np.concatenate((c,fmap),axis=2) 
        fmap = np.concatenate((fmap,c),axis=2)
    #print_image_data(fmap[0])
    return fmap

# Convolution core of 2D feature map
# fmpa - feature map with shape of (channel, y/row/height, x/column/width)
# core - conv core in shape of (channel, y/row/height, x/column/width)
# out - np array pointer to store output data
# stride - stride or step length for conv core to move along x/y 
def conv2d_core(fmap,core,out,stride=1):
    my = out.shape[0] # max y or row range
    mx = out.shape[1] # max x or col range 
    for y in range(my):
        for x in range(mx):
            sy = y * stride
            ey = sy + core.shape[1]
            sx = x * stride
            ex = x + core.shape[2]
            slicexy = fmap[:,sy:ey,sx:ex]
            mult_2d = slicexy * core
            out[y][x] = mult_2d.sum()

# Calculate all conv a completed layer of neural network
# fmpa - feature map with shape of (channel, y/row/height, x/column/width), dtype=np.unit8
# W - weights for conv core, dtype=np.int8 
# stride - stride or step length for conv core to move along x/y 
# pad - extra rows and columns of padding data to be inserted
# pair - if Ture (default), will insert padding data into both sides of each cord (even padding)
# output - conv output, dtype=np.int32
def conv2d_layer(fmap,W,stride=1,pad=0,pair=True):
    fmap = validate_fmap(fmap,W[0],pad,pair)
    s1 = int((fmap.shape[1]-W.shape[2]+1)/stride)
    s2 = int((fmap.shape[2]-W.shape[3]+1)/stride)
    ''' Attention: to avoid overflow, adder should be int32 type'''
    output = np.zeros((len(W),s1,s2),dtype=np.int32)
    for i in range(len(W)):
        conv2d_core(fmap,W[i],output[i],stride)
    return output

# Relu activation function for output from conv layer
# fmpa - feature map with shape of (channel, y/row/height, x/column/width)
def relu_layer(fmap):
    for i in range(len(fmap)):
        for j in range(len(fmap[i])):
            for k in range(len(fmap[i][j])):
                val = fmap[i][j][k]
                fmap[i][j][k] = val if val>0 else 0
                
# Max pool for conv outputs
# fmpa - feature map with shape of (channel, y/row/height, x/column/width), dtype=np.uint8
# shape - shape of filter of max pool 
# stride - stride or step length for conv core to move along x/y 
# pad - extra rows and columns of padding data to be inserted
# pair - if Ture (default), will insert padding data into both sides of each cord (even padding)
# output - output to next CNN layer, dtype=np.uint8
def max_pool_layer(fmap,shape,stride,pad=0,pair=True):
    fmap = validate_fmap(fmap,fmap,pad,pair)
    s1 = int((fmap.shape[1]-shape[0]+1)/stride)
    s2 = int((fmap.shape[2]-shape[1]+1)/stride)
    output = np.zeros((len(fmap),s1,s2),dtype=np.uint8)
    for i in range(len(output)):
        for y in range(s1):
            for x in range(s2):
                sy = y*stride       # Start of y cord
                ey = sy+shape[0]    # End of y cord
                sx = x*stride       # Start of x cord
                ex = sx+shape[1]    # End of x cord
                slicexy = fmap[i][sy:ey,sx:ex]
                output[i][y][x] = slicexy.max()
    return output

# To quantize from dtype int32 to uint8 for CNN layer output
# scale - scale factor in dtype float32
# zero - zero point of uint8 data type
# return - quantized data in uint8 type
'''
Attention: quantize_linear plays role of relu and clip data that exceed range of 255
to make sure data can fit into uint8 type. 
din - input data in shape of (C,Height/Row,Width/Col), dtype=np.int32 
scale - Correspond to onnx model QLinearConv y_scale, dtype=np.float32
zero - Correspond to onnx model QLinearConv y_zero_point, dtype=np.uint8
wscale - Correspond to onnx model QLinearConv w_scale, dtype=np.float32 (w_zero is set to 0)
dout - data output for next CNN layer, dtype=np.uint8
'''
def quantize_linear(din,scale,zero,wscale):
    dout = np.zeros(din.shape,dtype=np.uint8)
    din = (din * wscale / scale + zero)
    for i in range(len(din)):
        for j in range(len(din[i])):
            for k in range(len(din[i][j])):
                val = din[i][j][k]           
                val = int(val)
                val = val if val > 0 else 0    
                val = 255 if val > 255 else val 
                dout[i][j][k] = val 
    return dout
                
def load_mnist_data():
    file_path = "../mnist.npz" # This is a numpy.lib.npyio.NpzFile
    fp = np.load(file_path, allow_pickle=True)
    image_list, targets = fp['x_test'], fp['y_test']
    fp.close()
    print("Data set have", len(image_list), "samples, with shape of", 
    image_list[0].shape, image_list[0].dtype)
    return image_list, targets

def load_weight(fname):
    W = np.load(fname, allow_pickle=True) # This is a numpy.ndarray
    print("Shape of ndarray weight is",W.shape,W.dtype)
    return W

# Neural network model/layers to infer MNIST image
# image - mnist image data in shape of (1,28,28) dtype=uint
# model - weight data for neural network
'''
# For details of model, please open "mnist-12-int8.onnx"
# by on-line tool https://netron.app/
INT8 W =  Float32_W / W_Scale + Zero_Point 
CONV B = (Float32_B / W_Scale + Zero_Point) / (Pre_Y_Scale...) # This is to align with input x scale, 
# Because input x is already quanitized by div Pre_Y_Scale in previous layer
# Flatten/Softmax bias are calculated by different scheme compared with CONV
FLAT B =  Float32_B / B_Scale + Zero_Point # Since no scaling by Pre_Y_Scale, predict output should consider
# 1) Either to align FLAT B with Softmax dot() operation range: flat_cov_out.dot(F) + (B-Zero_Point)*B_Scale/(Pre_Y_Scale...)
# 2) Or to align Softmax to Float32 model output range: flat_cov_out.dot(F)*(Pre_Y_Scale...) + (B-Zero_Point)*B_Scale
# This script use above second option
'''
def model_infer(image,model):
    (W1,B1,W2,B2,F3,B3,WS1,WS2,WS3) = model
    
    # Conv layer 1 with relu and max pool 
    # Conv core shape is (8,1,5,5) with stride=1 and pad=2, 
    # max_pool filter is (2,2), stride=1, pad=1, pair=false
    out = conv2d_layer(image,W1,pad=2) + B1
    y_scale_1 = 3.679605722427368
    y_zero_point_1 = 0
    out = quantize_linear(out,y_scale_1,y_zero_point_1,WS1)
    out1 = max_pool_layer(out,(2,2),stride=2,pad=1,pair=False)
    #print_image_data(out1[0])

    # Conv layer 2 with relu and m pool 
    # Conv core shape is (16,8,5,5) with stride=1 and pad=2, 
    # max_pool filter is (3,3), stride=3, pad=0, pair=true  
    out = conv2d_layer(out1,W2,pad=2) + B2
    y_scale_2 = 9.886133193969727
    y_zero_point_2 = 0
    out = quantize_linear(out,y_scale_2,y_zero_point_2,WS2) 
    out2 = max_pool_layer(out,(3,3),stride=3,pad=0)
    #print_image_data(out2[0])

    # Flatten/Softmax layer - flatten by MatMul of factor in shape of (256,10)
    b3_scale = 0.00104560155887156
    b3_zero_point = 121
    out3 = out2.reshape((256)).dot(F3)*WS3*y_scale_1*y_scale_2 + (B3-b3_zero_point)*b3_scale
    #print(out3)
    
    # Predict result
    result = list(out3).index(out3.max())
    return result
  
'''
Below are function definitions for hardware simulatin purpose
1. Assumed CIM memory is 64 byte * 256, output mux can be 8/4/2/1
2. If feature map channel >1, stack all channels into one CIM memory; 
   reshape from (ch,row,col) to (block, row, col*ch)
3. Each vector is 64 byte numpy array
4. Bias is loaded into either GPR or weight memory
5. Feature and weight memory map - for details, refer to excel tables
   Assumed feature map is proper padded. Channels has 4 categories:
   1) 1 channel - memory output is mux8 of logic address (one logic address has 8bytes)
   2) 2~4 channels - memory output is mux2
   3) 5~64 channels - can fit into 1/2/4/8 logic address
   4) >64 channels - need N vec64 + vecX to compute 1D CNN
6. Vector and byte mask format 
   1) [11:4] - address mask bit in one vector; [3:0] - highest address bytes count, other address use all bytes
   2) [11:4] - address mask bit in one vector; [3:0] - bytes count in one address
   3) [11:4] - byte mask bit in one address; [3:0] - address count in one vector

'''

# Load weight from PSRAM/DDR into CIM core local memory
def cim_load_weight(din,mem=[]):
    #img_data = np.transpose(img_data, (2, 0, 1)) # from (row,col,ch) to (ch,row,col)
    shape=din.shape
    # Below transpose and reshape stack kernel row according to kernel# one by one 
    #dtran = np.transpose(din,(0,2,3,1))
    #dresh = dtran.reshape((shape[0],shape[2],shape[1]*shape[3]))
    #dresh = dresh.reshape((shape[0]*shape[2],shape[1]*shape[3]))
    #print(din[0:2],"\n------------------")
    # Input weight shape is (out channels, in channels, kernel rows, kernel columns)
    if shape[1]==1: # example of din shape (8,1,5,5)
        outgrp = math.ceil(shape[0]/8)
        krow = math.ceil(shape[3]/8)
        #assert len(mem)+outgrp*krow<256, "Current memory can't fit all weight data!"
        for r in range(shape[2]): # number of kernel rows
            for g in range(outgrp): # output channel groups 
                och = shape[0]-g*8 if shape[0]-g*8<=8 else 8
                for p in range(krow): # number of vec64 needed for one kernel row
                    vec = np.zeros((64,),dtype=np.int32)
                    kmx = shape[3]-p*8 if shape[3]-p*8<=8 else 8
                    for x in range(kmx): # number logic address in one vec64
                        for k in range(och): # number of output channels
                            vec[x*8+k] = din[k][0][r][x]
                    mem.append(vec)
    elif shape[1]<=4:# example of din shape (9,3,5,5)
        dtran = np.transpose(din,(2,0,3,1))
        dresh = dtran.reshape((shape[2],shape[0]*shape[3],shape[1]))    
        pcs = int(8/shape[1]) # one logic address can store 2 or 4 rows
        outgrp = math.ceil(shape[0]/pcs) # row numbers after group
        vseg = 8 # number of segment in one vector
        krow = math.ceil(outgrp*shape[3]/vseg) # number of vec64 needed for one kernel row*outgrp 
        #assert len(mem)+shape[2]*krow<256, "Current memory can't fit all weight data!"
        slen = int(64/vseg) # length of each vector segment
        glen = math.ceil(8/pcs)
        vecm = np.zeros((shape[2]*krow,64),dtype=np.int32) # vector memory 
        #for i in range(6): print(dtran[i][0][0:6])
        for r in range(shape[2]):  # number of kernel rows
            for c in range(shape[0]): # number of channel 
                for x in range(shape[3]): # number of data in one kernel row
                    ic = c%pcs
                    adr = int(c/pcs)*shape[3]+x
                    row = int(adr/vseg)
                    ofs = adr%vseg*slen+ic*glen
                    vecm[r*krow+row][ofs:ofs+shape[1]] = dtran[r][c][x]
        for row in vecm: 
            mem.append(row)
            #print(row)
        
    elif shape[1]<=64: # example of din shape (16,9,10,10) 
        dtran = np.transpose(din,(2,0,3,1))
        dresh = dtran.reshape((shape[2],shape[0]*shape[3],shape[1]))
        vseg = 1 if shape[1]>32 else (2 if shape[1]>16 else (4 if shape[1]>8 else 8)) 
        slen = int(64/vseg) # length of each vector segment
        krow = math.ceil(shape[0]*slen*shape[3]/64) # number of vec64 needed for one kernel row
        #assert len(mem)+krow*shape[2]<256, "Current memory can't fit all weight data!"
        for r in range(shape[2]): # number of kernel rows
            for p in range(krow): # number of vec64 needed for one kernel row*och
                vec = np.zeros((64,),dtype=np.int32)
                for s in range(vseg): # number of logic address per vec64
                    kx = p*vseg + s # index of current kernel data 
                    pcs = (p*vseg+s)*slen # start address of current vector segment in reshaped weights
                    vec[s*slen:s*slen+shape[1]] = dresh[r][kx]
                mem.append(vec)
                
    else: # shape[1]>64
        VSEG = 8 # number of segment/address in one vec64 
        UNIT = 8 # number of bytes in one logic address
        plen = math.ceil(shape[1]/UNIT) # number of logic address for one kernel data
        krow = math.ceil(shape[0]*shape[3]*plen/VSEG) # number of vec64 needed for kernel_row*channel
        vecm = np.zeros((shape[2]*krow,64),dtype=np.int32) # vector memory 
        dtran = np.transpose(din,(2,0,3,1))
        for r in range(shape[2]): # number of kernel rows
            for c in range(shape[0]): # number of channel 
                for x in range(shape[3]): # number of data in one kernel row
                    for p in range(plen): # number of logic address for one kernel data
                        adr = r*krow*VSEG + c*shape[3]*plen + x*plen + p
                        row = int(adr/VSEG)
                        ofs = adr%VSEG
                        byt = shape[1]%UNIT if p==plen-1 else UNIT # number of byte in current logic address
                        vecm[row][ofs*UNIT:ofs*UNIT+byt] = dtran[r][c][x][p*UNIT:p*UNIT+byt]
        #i=0          
        for rx in vecm: 
            mem.append(rx)
            #print(i,rx)
            #i=i+1
    return mem
    
# Load feature map from PSRAM/DDR into CIM core local memory 
# Assumed feature map is proper padded before load into local memory
# din - feature map data input, make sure dtype is unit8, otherwise some bit is cut in vec*vec operation
# ws - weight shape 
def cim_load_fmap(din,ws):
    shape=din.shape
    mem = []
    # Input feature map shape is (Channels, Rows, Columns)
    if shape[0]==1: # channel==1
        rowgrp = math.ceil(shape[1]/8) # one group can store 8 rows
        nvec = math.ceil(shape[2]/8) # number of vec64 needed for one feature map row
        vseg = 8 # number of seg per vec64 
        slen = int(64/vseg) # length of each vector segment
        #assert rowgrp*nvec<256, "Current memory can't fit all feature map data!"
        for g in range(rowgrp):
            for p in range(nvec):
                vec = np.zeros((64,),dtype=np.uint8)
                for s in range(vseg):
                    for r in range(8):
                        vec[s*slen+r] = din[0][g*8+r][p*vseg+s]
                mem.append(vec)
                #print(vec)
    elif shape[0]<=4: # channels are 2~4
        pcs = int(8/shape[0]) # one logic address can store 2 or 4 rows
        rowgrp = math.ceil(shape[1]/pcs) # row numbers after group
        nvec = math.ceil(shape[2]/8) # number of vec64 needed for one feature map row
        vseg = 8 # number of seg per vec64 
        slen = int(64/vseg) # length of each vector segment
        #assert rowgrp*nvec<256, "Current memory can't fit all feature map data!"
        for g in range(rowgrp):
            for p in range(nvec):
                vec = np.zeros((64,),dtype=np.uint8)
                for s in range(vseg):
                    for r in range(pcs):
                        for c in range(shape[0]):
                            vec[s*8+int(r*vseg/pcs)+c] = din[c][g*pcs+r][p*vseg+s]
                mem.append(vec)
    elif shape[0]<=64: # channels are 5~64
        vseg = 1 if shape[0]>32 else (2 if shape[0]>16 else (4 if shape[0]>8 else 8))  # number of seg per vec64 
        slen = int(64/vseg) # length of each vector segment, segment length 
        nvec = math.ceil(shape[2]/vseg) # number of vec64 needed for one feature map row
        #assert shape[1]*nvec<256, "Current memory can't fit all feature map data!"
        dtran = np.transpose(din,(1,2,0))
        #dresh = dtran.reshape((shape[1],shape[0]*shape[2]))
        #for i in range(5): print(dtran[i][0:5])
        for r in range(shape[1]):
            for p in range(nvec):
                vec = np.zeros((64,),dtype=np.uint8)
                for s in range(vseg):
                    if p*vseg+s>=shape[2]: continue
                    vec[s*slen:s*slen+shape[0]] = dtran[r][p*vseg+s]
                mem.append(vec)
                #print(vec)
    else: # channels >64, example (90,18,18)
        VSEG = 8 # number of segment/address in one vec64 
        UNIT = 8 # number of bytes in one logic address
        plen = math.ceil(shape[0]/UNIT) # number of logic address for one fm data
        nvec = math.ceil(shape[2]*plen/VSEG) # number of vec64 needed for one fearture map row
        dtran = np.transpose(din,(1,2,0))
        vecm = np.zeros((shape[1]*nvec,64),dtype=np.uint8) # vector memory 
        for r in range(shape[1]):
            for x in range(shape[2]):
                for p in range(plen):
                    adr = r*nvec*VSEG + x*plen + p
                    row = int(adr/VSEG)
                    ofs = adr%VSEG
                    byt = shape[0]%UNIT if p==plen-1 else UNIT # number of byte in current logic address
                    vecm[row][ofs*UNIT:ofs*UNIT+byt] = dtran[r][x][p*UNIT:p*UNIT+byt] 
        for row in vecm:
            mem.append(row)
    return mem

# Convolution core of CIM 
# 先按照1D CNN的方式逐行处理 feature map并存储在RAM中; 
# 2D CNN第二行和后续行的计算结果需要把当前行的MAC结果于RAM中之前的结果相加后再存储。
# 1) CH<=64 读取一个kernel Vec计算整个feature map row, 然后换下一个kernel Vec计算整个feature
# 2) CH>64 Kernel和Feature map地址同时增加
# mem0 - memory block 0, default is feature map 
# ws0 - data shape of memory 0,format(Channel, Krow, Kcol),example (8,32,32) 
# mem1 - memory block 1, default is weight
# ws1 - data shape of memory, format(out channel, in channel, rows, cols), example (16,8,5,5)
def cim_conv2d(mem0,ws0,mem1,ws1,stride=1,offset=0):
    shape=ws0
    kaddr=0 # kernel address, unit is 8 bytes
    kvcnt=0 # kernel vector counter, unit is 8 bytes
    khcnt=0 # kernel output channel counter
    kgcnt=0 # kernel output channel counter for input channel <=4 only 
    krcnt=0 # kernel row counter
    
    maskv=0 # vector mask inside one vector
    maskb=0 # byte mask inside one logical address
    
    faddr=0 # feature map address
    fvcnt=0 # feature map vector counter in one pixel
    fxcnt=0 # feature map pixel counter in one row
    frcnt=0 # feature map row coutner
    
    oaddr=0 # output feature address
    ohcnt=0 # output channel counter
    oxcnt=0 # output pixel counter in one row
    orcnt=0 # output row counter
    
    s1 = int((ws0[1]-ws1[2]+1)/stride)
    s2 = int((ws0[2]-ws1[3]+1)/stride)
    output = np.zeros((ws1[0],s1,s2),dtype=np.int32)
    
    UNIT = 8 # number of bytes in one logic address
    unit = UNIT
    VSEG = 8 # number of logic address in one vector

    if shape[0]<=4: 
        USEG = int(UNIT/shape[0]) # number of data rows in one address; or number of kernel channel in one address
        ULEN = math.ceil(UNIT/USEG) # length of one piece of data
        OHGRP = math.ceil(ws1[0]/USEG) # grouped output channel numbers  
        KVCNT = math.ceil(ws1[3]/VSEG) # number of vec64 for one_kernel_row 
        KHGLE = math.ceil(OHGRP*ws1[3]/VSEG) # number of vec64 for (one_kernel_row * OHGRP)
        KLAST = VSEG if ws1[3]%VSEG==0 else ws1[3]%VSEG
        RVCNT = math.ceil(ws0[2]/VSEG) # number of vec64 for one feature map row
        for kr in range(ws1[3]):
            for kh in range(OHGRP):
                for kg in range(USEG): # affect byte mask bit in format #3 
                    for kv in range(KVCNT):  
                        kvcnt=kv
                        maskv = KLAST if kv==KVCNT-1 else 8 # vector mask in format #3
                        kgcnt=kg
                        khcnt=kh*USEG+kg
                        if khcnt>=ws1[0]: continue 
                        #kaddr=KVCNT*VSEG*ws1[1]*kgcnt + KVCNT*kr*VSEG + VSEG*kvcnt + offset
                        kaddr=KHGLE*kr*VSEG + kh*ws1[3] + kv*VSEG + offset
                        maskb=khcnt 
                        #kernel filter access
                        krow0 = int(kaddr/VSEG)
                        krow1 = int((kaddr+maskv-1)/VSEG)
                        start = kaddr%VSEG 
                        if krow0==krow1:
                            kvec = mem1[krow0][start*unit:(start+maskv)*unit]
                        else:
                            kvec = np.zeros((maskv*unit,),dtype=np.int32)
                            kvec[0:(8-start)*unit] = mem1[krow0][start*unit:8*unit]
                            kvec[(8-start)*unit:maskv*unit] = mem1[krow1][0:(maskv+start-8)*unit]
                        kernel = np.zeros((ULEN*maskv,),dtype=np.int32)
                        for i in range(maskv):
                            kernel[i*ULEN:(i+1)*ULEN] = kvec[i*VSEG+kg*ULEN : i*VSEG+kg*ULEN+ULEN]
                        #if kr==0 and fr==0: print(kr,kv,kh,kg,"--",khcnt,krow0,krow1,kernel)
                        for fr in range(0,ws0[1]-ws1[2]+1,stride):
                            frcnt=fr+kr
                            for fx in range(0,ws0[2]-ws1[3]+1,stride):
                                faddr_offset = int(frcnt/USEG)*RVCNT*VSEG+kvcnt*VSEG
                                fxcnt=fx                         
                                faddr=faddr_offset+fxcnt
                                init=0 if kr==0 and kv==0 else output[khcnt][fr][fx]
                                frow0=int(faddr/8)
                                frow1=int((faddr+maskv-1)/8)
                                fstar=faddr%8
                                if frow0==frow1:
                                    fvec=mem0[frow0][fstar*unit:(fstar+maskv)*unit]
                                else:
                                    fvec=np.zeros((maskv*unit,),dtype=np.uint8)
                                    fvec[0:(8-fstar)*unit] = mem0[frow0][fstar*unit:8*unit]
                                    fvec[(8-fstar)*unit:maskv*unit] = mem0[frow1][0:(maskv+fstar-8)*unit]
                                fdata = np.zeros((ULEN*maskv,),dtype=np.uint8)
                                for i in range(maskv):
                                    fdata[i*ULEN:(i+1)*ULEN]=fvec[i*VSEG+frcnt%USEG*ULEN : i*VSEG+frcnt%USEG*ULEN+ULEN]
                                mult_2d=kernel*fdata                          
                                output[khcnt][fr][fx] = mult_2d.sum()+init 
                                # if khcnt==0 and kg==0 and fr==0 and fx==0: print(kr,fr,fx,kaddr,"\n",fdata,"\n",kernel)

    elif shape[0]<=64:
        ULEN = 64 if shape[0]>32 else (32 if shape[0]>16 else (16 if shape[0]>8 else 8))  # length of one piece of data
        KVCNT = math.ceil(ULEN*ws1[3]/64) #math.ceil(ws1[1]*ws1[3]/64) # number of vec64 for one_kernel_row 
        KRLEN = math.ceil(ws1[0]*ULEN*ws1[3]/UNIT) # number of address for (one_kernel_row * output_channel) 
        KALEN = math.ceil(ULEN*ws1[3]/UNIT) # logic address length of one_kernel_row
        KLAST = VSEG if KALEN%VSEG==0 else KALEN%VSEG # logic address for last segment of kernel row
        RVCNT = math.ceil(ws0[2]*ULEN/64) # number of vec64 for one feature map row
        print("RC,RS,HC,HS,VC,VS:",ws1[3],KRLEN,ws1[0],KALEN,KVCNT,VSEG,ws0,ws1)
        print("YC,YS,XC,XS:",int((ws0[1]-ws1[2]+1)/stride),RVCNT*VSEG*stride,int((ws0[2]-ws1[3]+1)/stride),int(ULEN/VSEG))
        for kr in range(ws1[3]):
            for kh in range(ws1[0]):
                khcnt=kh
                for kv in range(KVCNT):
                    kvcnt=kv
                    adnum=KLAST if kv==KVCNT-1 else VSEG # HW maskv - address numbers in one vector for mask format #1 
                    lastb= 8 if ws1[1]*ws1[3]%8==0 else ws1[1]*ws1[3]%8  # HW maskb - highest address bytes of mask format #1 
                    kaddr=kr*KRLEN + khcnt*KALEN + kvcnt*VSEG + offset
                    #print(kr,kh,kv,"----",kaddr,adnum,maskb)
                    krow0 = int(kaddr/VSEG)
                    krow1 = int((kaddr+adnum-1)/VSEG)
                    start = kaddr%VSEG 
                    if krow0==krow1:
                        kernel = mem1[krow0][start*unit:(start+adnum)*unit]                      
                    else:
                        kernel = np.zeros((adnum*unit,),dtype=np.int32)
                        kernel[0:(8-start)*unit] = mem1[krow0][start*unit:8*unit]
                        kernel[(8-start)*unit:adnum*unit] = mem1[krow1][0:(adnum+start-8)*unit]
                    #print(kr,kh,kernel)
                    if((kr<2 and kh<2) or (kr==ws1[3]-1 and kh==ws1[0]-1)): 
                        print("\nWeigh------------")
                        array_to_hex_string(kernel.tolist())                    
                    for fr in range(0,ws0[1]-ws1[2]+1,stride):
                        frcnt=fr+kr
                        faddr_offset = frcnt*RVCNT*VSEG + kvcnt*VSEG  
                        for fx in range(0,ws0[2]-ws1[3]+1,stride):
                            fxcnt=fx
                            faddr=faddr_offset+int(fxcnt*ULEN/VSEG)
                            init=0 if kr==0 and kv==0  else output[kh][fr][fx]
                            frow0=int(faddr/VSEG)
                            frow1=int((faddr+adnum-1)/VSEG)
                            fstar=faddr%VSEG
                            if frow0==frow1:
                                fdata=mem0[frow0][fstar*unit:(fstar+adnum)*unit]
                            else:
                                fdata=np.zeros((adnum*unit,),dtype=np.uint8)
                                fdata[0:(8-fstar)*unit] = mem0[frow0][fstar*unit:8*unit]
                                fdata[(8-fstar)*unit:adnum*unit] = mem0[frow1][0:(adnum+fstar-8)*unit]
                            mult_2d=kernel*fdata                          
                            output[kh][fr][fx] = mult_2d.sum()+init
                            #if kh==0 and fr==0 and fx==0: print(kr,kv,"--",fr,fx,adnum,"\n",kernel,"\n",fdata)
                            adro = kh*(ws0[1]-ws1[2]+1)*(ws0[2]-ws1[3]+1)+fr*(ws0[2]-ws1[3]+1)+fx
                            if((kr<2 and kh<2) or (kr==ws1[3]-1 and kh==ws1[0]-1)): 
                                array_to_hex_string(fdata.tolist())
                                print("R,H,V,Y,X:",kr,kh,kv,fr,fx,"VLEN,PSUM,ADRO,ADRW,ADRV",adnum,(kr!=0 or kv!=0),adro,kaddr+16,faddr+3216)
    else: # shape[0]>64 
        ULEN  = math.ceil(shape[0]/UNIT) # number of logic address for one kernel/fm data
        KVCNT = math.ceil(ULEN*ws1[3]/VSEG)  # number of vec64 for one kernel data row
        KRLEN = math.ceil(ws1[0]*ULEN*ws1[3]/VSEG) # number of vec64 for (one_kernel_row * output_channel) 
        KALEN = math.ceil(ULEN*ws1[3]) # logic address length of one_kernel_row
        KLAST = VSEG if ULEN*ws1[3]%VSEG==0 else ULEN*ws1[3]%VSEG # logic address for last segment of kernel row
        RVCNT = math.ceil(ws0[2]*ULEN/VSEG) # number of vec64 for one feature map row
        for kr in range(ws1[3]):
            for kh in range(ws1[0]):
                khcnt=kh
                for kv in range(KVCNT):
                    kvcnt=kv
                    adnum=KLAST if kv==KVCNT-1 else VSEG # HW maskv - address numbers in one vector for mask format #1 
                    #lastb= 8 if ws1[1]%UNIT==0 else ws1[1]%UNIT  # HW maskb - highest address bytes of mask format #1 
                    #lensg= (0 if adnum<0 else (adnum-1)*unit) + lastb # actual number bytes in one address 
                    kaddr=kr*KRLEN*VSEG + khcnt*KALEN + kvcnt*VSEG + offset
                    #print(kr,kh,kv,"----",kaddr,adnum,maskb)
                    krow0 = int(kaddr/VSEG)
                    krow1 = int((kaddr+adnum-1)/VSEG)
                    start = kaddr%VSEG 
                    if krow0==krow1:
                        kernel = mem1[krow0][start*unit:(start+adnum)*unit]                      
                    else:
                        kernel = np.zeros((adnum*unit,),dtype=np.int32)
                        kernel[0:(8-start)*unit] = mem1[krow0][start*unit:8*unit]
                        kernel[(8-start)*unit:adnum*unit] = mem1[krow1][0:(adnum+start-8)*unit]
                    for fr in range(0,ws0[1]-ws1[2]+1,stride):
                        frcnt=fr+kr
                        faddr_offset = frcnt*RVCNT*VSEG + kvcnt*VSEG  
                        for fx in range(0,ws0[2]-ws1[3]+1,stride):
                            fxcnt=fx
                            # When ULEN<=8, memory read in one cycle (may use previous cycle data)
                            # If ULEN>8, recommend to have ULEN align to 8, such as 16,24,32 to enable read memory in only one cycle
                            faddr=faddr_offset+fxcnt*ULEN # Recommend to have ULEN align to 1/2/4/8 to read memory in one cycle
                            init=0 if kr==0 and kv==0  else output[kh][fr][fx]
                            frow0=int(faddr/VSEG)
                            frow1=int((faddr+adnum-1)/VSEG)
                            fstar=faddr%VSEG
                            if frow0==frow1:
                                fdata=mem0[frow0][fstar*unit:(fstar+adnum)*unit]
                            else:
                                fdata=np.zeros((adnum*unit,),dtype=np.uint8)
                                fdata[0:(8-fstar)*unit] = mem0[frow0][fstar*unit:8*unit]
                                fdata[(8-fstar)*unit:adnum*unit] = mem0[frow1][0:(adnum+fstar-8)*unit]
                            mult_2d=kernel*fdata                         
                            output[kh][fr][fx] = mult_2d.sum()+init
                            if(kr<2 and fr<3): print("VLEN,PSUM:",adnum,(kr!=0 or kv!=0),"R,H,V,Y,X:",kr,kh,kv,fr,fx)
    return output 

# hardware conv2d simulation : fmap channel=1/8, kernel 5x5
def run_conv2d_sim_1(W1,W2,offset=5*8):
    # Load weights into local memory
    M0 = cim_load_weight(W1,[])
    M0 = cim_load_weight(W2,M0)
    print("Total len of weights:",len(M0))
    #image = np.array(image_list[0]).reshape((1,28,28)) 
    test = np.arange(1024,dtype=np.uint8).reshape((1,32,32)) % 11
    M1 = cim_load_fmap(test,W1.shape)
    out = cim_conv2d(M1,test.shape,M0,W1.shape)
    ref = conv2d_layer(test,W1)
    if np.array_equal(ref,out): 
        print("---> Run_conv2d_sim_1 passed")
    else:
        print("\n",out[0])
        print("\n",ref[0])
    
    test = np.arange(2592,dtype=np.uint8).reshape((8,18,18)) % 11
    M2 = cim_load_fmap(test,W2.shape)
    out = cim_conv2d(M2,test.shape,M0,W2.shape,offset=offset)
    ref = conv2d_layer(test,W2)
    if np.array_equal(ref,out): 
        print("---> Run_conv2d_sim_1 passed")
    else:
        print("\n",out[0])
        print("\n",ref[0])     

# hardware conv2d simulation : fmap channel=3/9 
def run_conv2d_sim_2(W1,W2,offset=20*8):
    # Load weights into local memory
    M0 = cim_load_weight(W1,[])
    M0 = cim_load_weight(W2,M0)
    print("Total len of weights:",len(M0))
    test = np.arange(1024,dtype=np.uint8).reshape((1,32,32)) % 11
    fm1 = np.concatenate((test,test,test),axis=0)
    M1 = cim_load_fmap(fm1,W1.shape)
    out = cim_conv2d(M1,fm1.shape,M0,W1.shape)
    ref = conv2d_layer(fm1,W1)
    if np.array_equal(ref,out): 
        print("---> Run_conv2d_sim_2 passed")
    else:
        print("\n",out[0])
        print("\n",ref[0])

    M0 = cim_load_weight(W2,[])
    test = np.arange(2592,dtype=np.uint8).reshape((8,18,18)) % 11
    ch = test[1].reshape((1,18,18))
    fm2 = np.concatenate((test,ch),axis=0)
    M2 = cim_load_fmap(fm2,W2.shape)   
    #out = cim_conv2d(M2,fm2.shape,M0,W2.shape,offset=offset)
    out = cim_conv2d(M2,fm2.shape,M0,W2.shape)
    ref = conv2d_layer(fm2,W2)
    print(out)
    if np.array_equal(ref,out): 
        print("---> Run_conv2d_sim_2 passed")
    else:
        print("\n",out[0])
        print("\n",ref[0])  
    #print_mem_hex(M0)
    #print("\n")
    #print_mem_hex(M2)


# Hardware conv2d simulatin: fmap channel>64, W~(4,90,10,10), Fm~(90,18,18)
def run_conv2d_sim_3(W):
    M0 = cim_load_weight(W,[])
    print("Total len of weights:",len(M0))
    #test = np.arange(2916,dtype=np.uint8).reshape((9,18,18)) % 11
    test = np.arange(2592,dtype=np.uint8).reshape((8,18,18)) % 11
    ch = test[1].reshape((1,18,18))
    fm2 = np.concatenate((test,ch),axis=0)
    fm = np.concatenate((fm2,fm2),axis=0)
    #fm = np.concatenate((test,test),axis=0)
    for i in range(8):
        #fm = np.concatenate((fm,test),axis=0)
        fm = np.concatenate((fm,fm2),axis=0)

    M1 = cim_load_fmap(fm,W.shape) 
    out = cim_conv2d(M1,fm.shape,M0,W.shape,offset=0)
    ref = conv2d_layer(fm,W)    
    if np.array_equal(ref,out): 
        print("---> Run_conv2d_sim_3 passed")
    else:
        print("\n",out[0])
        print("\n",ref[0])
    
# test cim conv2d features 
def cim_conv2d_test(W1,W2):
    
    print("\nMNIST data set CNN comparison:")
    #run_conv2d_sim_1(W1,W2)
    
    ch = W1[1].reshape(1,1,5,5)
    Wx = np.concatenate((W1,ch),axis=0)
    Wx1 = np.concatenate((Wx,Wx,Wx),axis=1)
    ch = W2[0:2].reshape(16,1,5,5)
    Wx2 = np.concatenate((W2,ch),axis=1)
    print("\nCNN channel extend to 3 and 9:")
    #run_conv2d_sim_2(Wx1,Wx2)

    ch = np.zeros((9,3,5,5),dtype=np.int32)
    Wx3 = np.concatenate((Wx1,ch),axis=3)
    ch = np.zeros((9,3,5,10),dtype=np.int32)
    Wx3 = np.concatenate((Wx3,ch),axis=2)
    ch = np.zeros((16,9,5,5),dtype=np.int32)
    Wx4 = np.concatenate((Wx2,ch),axis=3)
    ch = np.zeros((16,9,5,10),dtype=np.int32)
    Wx4 = np.concatenate((Wx4,ch),axis=2)
    print("\nKernel size extend to 10x10:")
    run_conv2d_sim_2(Wx3,Wx4,offset=70*8)

    #for i in range(9): print(Wx4[0][i])
    ch = Wx4[0:4]
    Wx5 = np.concatenate((ch,ch),axis=1)
    for i in range(8):
        Wx5 = np.concatenate((Wx5,ch),axis=1)
    print("\nFM shape (90,18,18), Kernel shape (4,90,10,10):")
    #run_conv2d_sim_3(Wx5)
    
def conv2d_layer_hw_sim(fmap,W,stride=1,pad=0,pair=True):
    fmap = validate_fmap(fmap,W[0],pad,pair)
    M0 = cim_load_weight(W,[])
    M1 = cim_load_fmap(fmap,W.shape)
    out = cim_conv2d(M1,fmap.shape,M0,W.shape)
    return out

def model_infer_hw_sim(image,model):
    (W1,B1,W2,B2,F3,B3,WS1,WS2,WS3) = model
    
    # Conv layer 1 with relu and max pool 
    # Conv core shape is (8,1,5,5) with stride=1 and pad=2, 
    # max_pool filter is (2,2), stride=1, pad=1, pair=false
    out = conv2d_layer_hw_sim(image,W1,pad=2) + B1
    y_scale_1 = 3.679605722427368
    y_zero_point_1 = 0
    out = quantize_linear(out,y_scale_1,y_zero_point_1,WS1)
    out1 = max_pool_layer(out,(2,2),stride=2,pad=1,pair=False)
    #print_image_data(out1[0])

    # Conv layer 2 with relu and m pool 
    # Conv core shape is (16,8,5,5) with stride=1 and pad=2, 
    # max_pool filter is (3,3), stride=3, pad=0, pair=true  
    out = conv2d_layer_hw_sim(out1,W2,pad=2) + B2
    y_scale_2 = 9.886133193969727
    y_zero_point_2 = 0
    out = quantize_linear(out,y_scale_2,y_zero_point_2,WS2) 
    out2 = max_pool_layer(out,(3,3),stride=3,pad=0)

    # Flatten/Softmax layer - flatten by MatMul of factor in shape of (256,10)
    b3_scale = 0.00104560155887156
    b3_zero_point = 121
    out3 = out2.reshape((256)).dot(F3)*WS3*y_scale_1*y_scale_2 + (B3-b3_zero_point)*b3_scale

    # Predict result
    result = list(out3).index(out3.max())
    return result

'''
Below subroutines demostrate matrix mult A·B 
If use GNN mode, each 64/16 rows of B matrix is compressed into 64/16 bits of mask to calcuate vector adder 
'''
def trans_matrix_to_vector(UNIT,VSEG,d):
    a, b = d.shape
    VECTOR = UNIT*VSEG
    padding = (UNIT - (b % UNIT)) % UNIT
    padded_zero = np.zeros((a, padding),dtype=np.int32)
    d_pad1 = np.concatenate((d, padded_zero),axis=1)
    pad_size = (VECTOR - (d_pad1.size % VECTOR)) % VECTOR
    temp1 = d_pad1.ravel()
    d_pad2 = np.pad(temp1, (0, pad_size), mode='constant', constant_values=0)
    mem = d_pad2.reshape(-1,VECTOR)
    return mem

def cim_matrix_load(d1, d2):
    assert d1.ndim==2 and d2.ndim==2
    assert d1.shape[1]==d2.shape[0]
    UNIT=8 # bytes in one logic address 
    VSEG=8 # number of logic address in one vec64 
    
    if d1.shape[1]<=4:
        pass
    else:
        mem0 = trans_matrix_to_vector(UNIT,VSEG,d1)
        dtran = np.transpose(d2,(1,0))
        mem1 = trans_matrix_to_vector(UNIT,VSEG,dtran)

    return mem0,mem1
    
    
# def cim_matrix_load_b(d1,d2):
#     assert d1.ndim==2 and d2.ndim==2
#     assert d1.shape[1]==d2.shape[0]
#     shape = d1.shape
    
#     UNIT=8 # bytes in one logic address 
#     VSEG=8 # number of logic address in one vec64 
#     dtran = np.transpose(d2,(1,0))
#     if shape[1]<=4:
#         GLEN = int(UNIT/shape[1]) # number of group data in one logic address
#         CLEN = 1 # number of address for one column of data
#         CSEG = int(UNIT/GLEN) # number of bytes needed for memory store one column of data
#         VCNT = 1 # number of vec64 need for one column of data
#         VALL = math.ceil(CLEN*shape[0]/VSEG/GLEN) # total number of vec64 for whole matrix
#         RCNT = math.ceil(shape[0]/GLEN) # number of row after grouped
#         mem0 = np.zeros((VALL,64),dtype=np.int32)
#         mem1 = np.zeros((VALL,64),dtype=np.int32)
#         for r in range(RCNT):
#             for g in range(GLEN):
#                 rg = r*GLEN+g
#                 if rg>shape[0]-1: continue
#                 adr = r
#                 row = int(adr/VSEG)
#                 ofs = adr%VSEG
#                 seg = shape[1]
#                 mem0[row][ofs*UNIT+g*CSEG:ofs*UNIT+g*CSEG+seg] = d1[rg]
#                 mem1[row][ofs*UNIT+g*CSEG:ofs*UNIT+g*CSEG+seg] = dtran[rg]   
#     else:
#         CLEN = math.ceil(shape[1]/UNIT) # number of address for one column of data
#         VCNT = math.ceil(CLEN/VSEG) # number of vec64 need for one column of data
#         VALL = math.ceil(CLEN*shape[0]/VSEG) # total number of vec64 for whole matrix
#         VLD2 = math.ceil(CLEN*d2.shape[1]/VSEG) # total number of vec64 for d2 
#         mem0 = np.zeros((VALL,64),dtype=np.int32)
#         mem1 = np.zeros((VLD2,64),dtype=np.int32)
#         for r in range(shape[0]):
#             for v in range(VCNT):
#                 adr = r*CLEN + v*VSEG
#                 row = int(adr/VSEG)
#                 ofs = adr%VSEG
#                 if shape[1] <= UNIT:
#                     seg = shape[1]
#                     mem0[row][ofs*UNIT:ofs*UNIT+seg] = d1[r]
#                 elif shape[1] <=UNIT*VSEG:
#                     seg = shape[1]
#                     row1 = int((adr+CLEN-1)/VSEG)
#                     if row==row1:
#                         mem0[row][ofs*UNIT:ofs*UNIT+seg] = d1[r]
#                     else:
#                         mem0[row][ofs*UNIT:VSEG*UNIT] = d1[r][0:(VSEG-ofs)*UNIT]
#                         mem0[row1][0:seg-(VSEG-ofs)*UNIT] = d1[r][(VSEG-ofs)*UNIT:seg]
#                 else:
#                     lst = shape[1]%(VSEG*UNIT) if shape[1]%(VSEG*UNIT)>0 else VSEG*UNIT
#                     seg = lst if v==VCNT-1 else VSEG*UNIT
#                     ads = int(seg/UNIT)-1 if int(seg/UNIT)>1 else 0
#                     row1 = int((adr+ads)/VSEG)
#                     #print(r,v,adr,ofs,row,row1,seg)
#                     if row==row1:
#                         temp1 = mem0[row][ofs*UNIT:ofs*UNIT+seg]
#                         temp2 = d1[r][v*64:v*64+seg]
#                         mem0[row][ofs*UNIT:ofs*UNIT+seg] = d1[r][v*64:v*64+seg]
#                     else:
#                         mem0[row][ofs*UNIT:VSEG*UNIT] = d1[r][v*64:v*64+(VSEG-ofs)*UNIT]
#                         mem0[row1][0:seg-(VSEG-ofs)*UNIT] = d1[r][v*64+(VSEG-ofs)*UNIT:v*64+seg]
                        
#         for r in range(d2.shape[1]):
#             for v in range(VCNT):
#                 adr = r*CLEN + v*VSEG
#                 row = int(adr/VSEG)
#                 ofs = adr%VSEG
#                 if shape[1] <= UNIT:
#                     seg = shape[1]
#                     mem1[row][ofs*UNIT:ofs*UNIT+seg] = dtran[r]
#                 elif shape[1] <=UNIT*VSEG:
#                     seg = shape[1]
#                     row1 = int((adr+CLEN-1)/VSEG)
#                     if row==row1:
#                         mem1[row][ofs*UNIT:ofs*UNIT+seg] = dtran[r]
#                     else:
#                         mem1[row][ofs*UNIT:VSEG*UNIT] = dtran[r][0:(VSEG-ofs)*UNIT]
#                         mem1[row1][0:seg-(VSEG-ofs)*UNIT] = dtran[r][(VSEG-ofs)*UNIT:seg]
#                 else:
#                     lst = shape[1]%(VSEG*UNIT) if shape[1]%(VSEG*UNIT)>0 else VSEG*UNIT
#                     seg = lst if v==VCNT-1 else VSEG*UNIT
#                     ads = int(seg/UNIT)-1 if int(seg/UNIT)>1 else 0
#                     row1 = int((adr+ads)/VSEG)
#                     #print(r,v,adr,ofs,row,row1,seg)
#                     if row==row1:
#                         if row<VLD2 and r<d2.shape[1]: mem1[row][ofs*UNIT:ofs*UNIT+seg] = dtran[r][v*64:v*64+seg]
#                     else:
#                         if row<VLD2 and r<d2.shape[1]: mem1[row][ofs*UNIT:VSEG*UNIT] = dtran[r][v*64:v*64+(VSEG-ofs)*UNIT]
#                         if row1<VLD2 and r<d2.shape[1]: mem1[row1][0:seg-(VSEG-ofs)*UNIT] = dtran[r][v*64+(VSEG-ofs)*UNIT:v*64+seg]                     
#     return mem0,mem1

def cim_matrix_mult_sim(M0,M1,S0,S1):
    UNIT=8 # bytes in one logic address 
    VSEG=8 # number of logic address in one vec64 
    if S0[1]<=4:
        GLEN = int(UNIT/S0[1]) # number of group data in one logic address
        CLEN = 1 # number of address for one column of data
        CSEG = int(UNIT/GLEN) # number of bytes needed for memory store one column of data
        VCNT = 1 # number of vec64 need for one column of data
        VALL = math.ceil(CLEN*S0[0]/VSEG/GLEN) # total number of vec64 for whole matrix
        RCNT = math.ceil(S0[0]/GLEN) # number of row after grouped
        vmem = np.zeros((S0[0],S1[1]),dtype=np.int32) 
        for r in range(S0[0]):
            for c in range(S1[1]):
                # M0 memory access 
                adr = int(r/GLEN)
                grp = r%GLEN
                row = int(adr/VSEG)
                ofs = adr%VSEG
                seg = S0[1]
                vec0 = M0[row][ofs*UNIT+grp*CSEG:ofs*UNIT+grp*CSEG+seg]
                # M1 memory access 
                adr = int(c/GLEN)
                grp = c%GLEN
                row = int(adr/VSEG)
                ofs = adr%VSEG                
                vec1 = M1[row][ofs*UNIT+grp*CSEG:ofs*UNIT+grp*CSEG+seg]
                #print(r,c,grp,row,ofs,vec0,vec1)
                vmem[r][c] = (vec0*vec1).sum()
    else:
        CLEN = math.ceil(S0[1]/UNIT) # number of address for one column of data
        VCNT = math.ceil(CLEN/VSEG) # number of vec64 need for one column of data
        VALL = math.ceil(CLEN*S0[0]/VSEG) # total number of vec64 for whole matrix 
        vmem = np.zeros((S0[0],S1[1]),dtype=np.int32) 
        lst = CLEN%VSEG if CLEN%VSEG>0 else VSEG  # number of address for last segment
        # print("Constant R,V,X,Last:",S0[0],VCNT,S1[1],lst)
        # print("Constant RS,VS,XS:",CLEN,VSEG,CLEN)
        for r in range(S0[0]):
            for v in range(VCNT):               
                seg = lst*UNIT if v==VCNT-1 else VSEG*UNIT # number of address for current segment
                adr0 = r*CLEN+v*VSEG
                row0 = int(adr0/VSEG)
                ads = int(seg/UNIT)-1 if int(seg/UNIT)>1 else 0
                row1 = int((adr0+CLEN-1)/VSEG) if VCNT==1 else int((adr0+ads)/VSEG)
                ofs = adr0%VSEG
                if row0==row1:
                    vec0 = M0[row0][ofs*UNIT:ofs*UNIT+seg]
                else:
                    vec0 = np.zeros((seg,),dtype=np.int32)
                    vec0[0:(VSEG-ofs)*UNIT] = M0[row0][ofs*UNIT:VSEG*UNIT]
                    vec0[(VSEG-ofs)*UNIT:seg] = M0[row1][0:(ofs-VSEG)*UNIT+seg]
                
                # print("\nWeigh------------")
                # array_to_hex_string(vec0.tolist())
                for c in range(S1[1]):
                    init = 0 if v==0 else vmem[r][c] 
                    adr1 = c*CLEN+v*VSEG
                    row0 = int(adr1/VSEG)
                    row1 = int((adr1+CLEN-1)/VSEG) if VCNT==1 else int((adr1+ads)/VSEG)
                    ofs = adr1%VSEG
                    if row0==row1:
                        vec1 = M1[row0][ofs*UNIT:ofs*UNIT+seg]
                    else:
                        vec1 = np.zeros((seg,),dtype=np.int32)
                        vec1[0:(VSEG-ofs)*UNIT] = M1[row0][ofs*UNIT:VSEG*UNIT]
                        vec1[(VSEG-ofs)*UNIT:seg] = M1[row1][0:(ofs-VSEG)*UNIT+seg]
                    vmem[r][c] = (vec0*vec1).sum() + init
                    # array_to_hex_string(vec1.tolist())
                    # print("Len,Init,OADR,ADRW,ADRV:",seg,v!=0,r*S1[1]+c,adr0+16,adr1+136,(vec0*vec1).sum())
 
    temp = {
        "R":S0[0],
        "V":VCNT,
        "X":S1[1],
        "Last":lst,
        "RS":CLEN,
        "VS":VSEG,
        "XS":CLEN,
    }
    return vmem,temp

def cim_matrix_mult_test():
    d1 = np.arange(9,dtype=np.int32).reshape((3,3)) % 11
    d2 = np.arange(9,dtype=np.int32).reshape((3,3)) % 13 - 6
    
    # d1 = np.arange(150,dtype=np.int32).reshape((50,3))
    # d2 = np.arange(150,dtype=np.int32).reshape((3,50))
    # d1 = np.arange(150,dtype=np.int32).reshape((50,3)) % 11
    # d2 = np.arange(150,dtype=np.int32).reshape((3,50)) % 13 - 6
    
    '''
    ### error
    # d1 = np.arange(15,dtype=np.int32).reshape((3,5))
    # d2 = np.arange(20,dtype=np.int32).reshape((5,4))
    # d1 = np.arange(72,dtype=np.int32).reshape((8,9)) % 11
    # d2 = np.arange(63,dtype=np.int32).reshape((9,7)) % 13 - 6  
    # d1 = np.arange(72,dtype=np.int32).reshape((8,9)) % 11
    # d2 = np.arange(99,dtype=np.int32).reshape((9,11)) % 13 - 6  
    # d1 = np.arange(54,dtype=np.int32).reshape((9,6)) % 11
    # d2 = np.arange(66,dtype=np.int32).reshape((6,11)) % 13 - 6  
    # d1 = np.arange(91,dtype=np.int32).reshape((7,13)) % 11
    # d2 = np.arange(143,dtype=np.int32).reshape((13,11)) % 13 - 6  
    '''
    d1 = np.arange(45,dtype=np.int32).reshape((5,9)) % 11
    d2 = np.arange(45,dtype=np.int32).reshape((9,5)) % 13 - 6  

    d1 = np.arange(770,dtype=np.int32).reshape((11,70)) % 11
    d2 = np.arange(770,dtype=np.int32).reshape((70,11)) % 13 - 6  

    # d1 = np.arange(2295,dtype=np.int32).reshape((17,135)) % 11
    # d2 = np.arange(2295,dtype=np.int32).reshape((135,17)) % 13 - 6  

    # d1 = np.arange(60,dtype=np.int32).reshape((6,10))
    # d2 = np.arange(60,dtype=np.int32).reshape((10,6))
    # d1 = np.arange(60,dtype=np.int32).reshape((6,10)) % 11
    # d2 = np.arange(60,dtype=np.int32).reshape((10,6)) % 13 - 6

    # d1 = np.arange(200,dtype=np.int32).reshape((10,20)) % 11
    # d2 = np.arange(200,dtype=np.int32).reshape((20,10)) % 13 - 6
    
    # d1 = np.arange(900,dtype=np.int32).reshape((10,90))
    # d2 = np.arange(810,dtype=np.int32).reshape((90,9)) 
    
    
    # d1 = np.arange(410,dtype=np.int32).reshape((5,82))
    # d2 = np.arange(410,dtype=np.int32).reshape((82,5))
    # d1 = np.arange(5600,dtype=np.int32).reshape((70,80)) % 80
    # d2 = np.arange(5600,dtype=np.int32).reshape((80,70)) % 13-6
    
    # d1 = np.arange(900,dtype=np.int32).reshape((10,90)) % 11
    # d2 = np.arange(810,dtype=np.int32).reshape((90,9)) % 13 - 6
    aa = 65
    bb = 50
    cc = 65
    d1 = np.arange(aa * bb, dtype=np.int32).reshape((aa, bb)) 
    d2 = np.arange(bb * cc, dtype=np.int32).reshape((bb, cc)) 
    '''
    d1 = np.arange(1950,dtype=np.int32).reshape((10,195)) % 11
    d2 = np.arange(1755,dtype=np.int32).reshape((195,9)) % 13 - 6
    '''
    M0,M1 = cim_matrix_load(d1,d2)
    M0_b,M1_b = cim_matrix_load_b(d1,d2)
    print_mem_hex(M0)
    print_mem_hex(M1)
    out,_ = cim_matrix_mult_sim(M0,M1,d1.shape,d2.shape)
    print(out)
    ref = d1.dot(d2)
    if np.array_equal(ref,out): print("---> Cim_matrix_mult_test passed")
    else: print("---> Cim_matrix_mult_test failed\n",out,"\n",ref)

def print_mem_hex(M):
    print("Length x 128bits:",len(M)*4)
    for m in M: 
        array_to_hex_string(m)
                

def array_to_hex_string(arr):
    row= math.ceil(len(arr)/16)
    for r in range(row):
        line = ""
        for s in range(16):
            if(r*16+s>=len(arr)): 
                line=line+"00"
            else:
                line=line+ ("00"+hex(arr[r*16+s] & 0xFF)[2:])[-2:]
        print(line)

def cim_gnn_add_test():
    d1 = np.arange(1950,dtype=np.int32).reshape((10,195)) % 11
    d2 = np.array([
        [0,0,0,0,0,0,0,1,  0,0,0,0,0,0,1,0,  0,0,0,0,0,0,1,1,  0,0,0,0,0,1,0,0,
         0,0,0,0,0,1,0,1,  0,0,0,0,0,1,1,0,  0,0,0,0,0,1,1,1,  0,0,0,0,1,0,0,0,
         0,0,0,0,0,0,0,1,  0,0,0,0,0,0,1,0,  0,0,0,0,0,0,1,1,  0,0,0,0,0,1,0,0,
         0,0,0,0,0,1,0,1,  0,0,0,0,0,1,1,0,  0,0,0,0,0,1,1,1,  0,0,0,0,1,0,0,0, 
         0,0,0,0,0,0,0,1,  0,0,0,0,0,0,1,0,  0,0,0,0,0,0,1,1,  0,0,0,0,0,1,0,0,
         0,0,0,0,0,1,0,1,  0,0,0,0,0,1,1,0,  0,0,0,0,0,1,1,1,  0,0,0,0,1,0,0,0, 0,0,1],
        [0,0,0,0,1,0,0,1,  0,0,0,0,1,0,1,0,  0,0,0,0,1,0,1,1,  0,0,0,0,1,1,0,0,
         0,0,0,0,1,1,0,1,  0,0,0,0,1,1,1,0,  0,0,0,0,1,1,1,1,  0,0,0,1,0,0,0,0, 
         0,0,0,0,1,0,0,1,  0,0,0,0,1,0,1,0,  0,0,0,0,1,0,1,1,  0,0,0,0,1,1,0,0,
         0,0,0,0,1,1,0,1,  0,0,0,0,1,1,1,0,  0,0,0,0,1,1,1,1,  0,0,0,1,0,0,0,0, 
         0,0,0,0,1,0,0,1,  0,0,0,0,1,0,1,0,  0,0,0,0,1,0,1,1,  0,0,0,0,1,1,0,0,
         0,0,0,0,1,1,0,1,  0,0,0,0,1,1,1,0,  0,0,0,0,1,1,1,1,  0,0,0,1,0,0,0,0, 0,1,0],         
        [0,0,0,1,0,0,0,1,  0,0,0,1,0,0,1,0,  0,0,0,1,0,0,1,1,  0,0,0,1,0,1,0,0,
         0,0,0,1,0,1,0,1,  0,0,0,1,0,1,1,0,  0,0,0,1,0,1,1,1,  0,0,0,1,1,0,0,0,  
         0,0,0,1,0,0,0,1,  0,0,0,1,0,0,1,0,  0,0,0,1,0,0,1,1,  0,0,0,1,0,1,0,0,
         0,0,0,1,0,1,0,1,  0,0,0,1,0,1,1,0,  0,0,0,1,0,1,1,1,  0,0,0,1,1,0,0,0, 
         0,0,0,1,0,0,0,1,  0,0,0,1,0,0,1,0,  0,0,0,1,0,0,1,1,  0,0,0,1,0,1,0,0,
         0,0,0,1,0,1,0,1,  0,0,0,1,0,1,1,0,  0,0,0,1,0,1,1,1,  0,0,0,1,1,0,0,0, 1,0,0]
        ])
    d2 = np.transpose(d2,(1,0)) 
    d3 = np.array([
        [1,2,3,4,5,6,7,8, 1,2,3,4,5,6,7,8, 1,2,3,4,5,6,7,8, 1,2,3,4,5,6,7,8, 1],
        [9,10,11,12,13,14,15,16, 9,10,11,12,13,14,15,16, 9,10,11,12,13,14,15,16, 9,10,11,12,13,14,15,16, 2],
        [17,18,19,20,21,22,23,24, 17,18,19,20,21,22,23,24, 17,18,19,20,21,22,23,24, 17,18,19,20,21,22,23,24, 4]
        ])
    ref = d1.dot(d2)
    print(ref)
    
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    image_list, targets = load_mnist_data()
    W1 = load_weight("weight_int8/conv1_5x5_int8.npy")
    B1 = load_weight("weight_int8/add_conv1_int32.npy")
    B1 = B1.reshape((8,1,1))
    WS1 = load_weight("weight_int8/w1_scale.npy")
    WS1 = WS1.reshape((8,1,1))

    W2 = load_weight("weight_int8/conv2_5x5_int8.npy")
    B2 = load_weight("weight_int8/add_conv2_int32.npy")
    B2 = B2.reshape((16,1,1))
    WS2 = load_weight("weight_int8/w2_scale.npy")
    WS2 = WS2.reshape((16,1,1))

    F3 = load_weight("weight_int8/reshape_256x10_int8.npy")
    B = load_weight("weight_int8/add_matmul_uint8.npy")
    B3 = B.reshape((10,)) 
    WS3 = load_weight("weight_int8/reshape_scale.npy")
    
    ''' 
    Attention: To prevent overflow of numpy.dot() math-mult, F3 (int8) should extend to int32.
    However, no need to extend in actual circuit because HW mult is uint8*int8 and Adder is int32 
    W1/W2/B3 are Similary 
    '''
    W1 = W1 * np.ones(W1.shape,dtype=np.int32)
    W2 = W2 * np.ones(W2.shape,dtype=np.int32)
    F3 = F3 * np.ones(F3.shape,dtype=np.int32)
    B3 = B3 * np.ones(B3.shape,dtype=np.int32)
    
    cim_matrix_mult_test()
    # cim_gnn_add_test()
    exit()
    # Test CIM conv2d functions
    cim_conv2d_test(W1,W2)
    exit()
    # Run test sets and statistic correct rate
    failure = 0
    cnt = 100
    print("\n============== MNIST Infer Test ==============")
    for i in range(cnt):
        # Conv inputs should be in shape of (Channel, Y/Height, X/Width)
        image = np.array(image_list[i]).reshape((1,28,28)) 
        ret = model_infer(image,(W1,B1,W2,B2,F3,B3,WS1,WS2,WS3))
        #ret = model_infer_hw_sim(image,(W1,B1,W2,B2,F3,B3,WS1,WS2,WS3))
        if ret!=targets[i]: failure +=1
        status = "Correct" if  ret==targets[i] else "Wrong <---"
        print("{:>4d} ".format(i)+f"Expect {targets[i]}, output {ret}: {status}")
    print(f"Top 1 classify correct rate {(cnt-failure)/cnt*100}%",)
    
    