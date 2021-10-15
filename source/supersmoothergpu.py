import os
import numpy.ctypeslib as npct
from ctypes import *
import csv
import numpy as np
from contextlib import contextmanager
import sys

def getColumn(filename, column):
    results = csv.reader(open(filename), delimiter=",")
    # next(results, None)  # skip the headers
    return [result[column] for result in results]


# This function converts an input numpy array into a different
# data type and ensure that it is contigious.  
def convert_type(in_array, new_dtype):

    ret_array = in_array
    
    if not isinstance(in_array, np.ndarray):
        ret_array = np.array(in_array, dtype=new_dtype)
    
    elif in_array.dtype != new_dtype:
        ret_array = np.array(ret_array, dtype=new_dtype)

    if ret_array.flags['C_CONTIGUOUS'] == False:
        ret_array = np.ascontiguousarray(ret_array)

    return ret_array


#from stackoverflow 
# https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
def redirect_stdout():
    print ("Verbose mode is false. Redirecting C shared library stdout to /dev/null")
    sys.stdout.flush() # <--- important when redirecting to files
    newstdout = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    sys.stdout = os.fdopen(newstdout, 'w')

def computeIndexRangesForEachObject(objId):
    start_index_arr=[]
    end_index_arr=[]
    unique_obj_ids_arr=[]
    lastId=objId[0]
    unique_obj_ids_arr.append(objId[0])

    start_index_arr.append(0)
    for x in range(0, len(objId)):
        if(objId[x]!=lastId):
            end_index_arr.append(x-1)
            start_index_arr.append(x)
            lastId=objId[x]
            #update the list of unique object ids
            unique_obj_ids_arr.append(objId[x])

    #last one needs to be populated
    end_index_arr.append(len(objId)-1)

    start_index_arr=np.asarray(start_index_arr, dtype=int)
    end_index_arr=np.asarray(end_index_arr, dtype=int)
    unique_obj_ids_arr=np.asarray(unique_obj_ids_arr)

    return start_index_arr, end_index_arr, unique_obj_ids_arr

def enumerateObjects(start_index_arr, end_index_arr):
    enumObjectId=[]
    for x in range (start_index_arr.size):
        numElems=end_index_arr[x]-start_index_arr[x]+1
        # print("Num elems: %d" %(numElems))
        enumObjectId.extend(numElems*[x])

    enumObjectId=np.asarray(enumObjectId, dtype=int)    
    # print("Total number of lines after enumeration: %d" %(enumObjectId.size))    
    # print("Total number of unique objects after enumeration: %d" %(np.unique(enumObjectId).size))    
    return enumObjectId


#Use the formulation in Richards et al. 2011
def computeNumFreqAuto(objId, timeX, fmin, fmax):

    start_index_arr, end_index_arr, _ = computeIndexRangesForEachObject(objId)     

    timeXLocal=np.asfarray(timeX)

    observing_window_arr=[]
    for x in range (0, start_index_arr.size):
        idxStart=start_index_arr[x]
        idxEnd=end_index_arr[x]
        observing_window_arr.append(timeXLocal[idxEnd]-timeXLocal[idxStart])

    observing_window_arr=np.asarray(observing_window_arr, dtype=float)

    maximumObservingWindow=np.max(observing_window_arr)

    

    deltaf=0.1/maximumObservingWindow

    num_freqs=(fmax-fmin)/deltaf
    num_freqs=int(num_freqs)
    print("*********************")
    print("Automatically generating the number of frequencies based on maximum observing window:")
    print("Max. Observing Window: %f, Delta f: %f" %(maximumObservingWindow, deltaf))
    print("Number of frequencies: ", num_freqs)
    print("*********************")
    return num_freqs




def supersmoother(objId, timeX, magY, magDY, minFreq, maxFreq, MODE, alphain=9.0, freqToTest="auto", dtype="float", verbose="false"):

    ###############################
    #Check for valid parameters and set verbose mode and generate frequencies for auto mode 

    #prevent C output from printing to screen
    if (verbose=="false"):
        redirect_stdout()

    #if the user doesn't specify the number of frequencies
    if (freqToTest=="auto"):
        freqToTest=computeNumFreqAuto(objId, timeX, minFreq, maxFreq)

    #check alpha parameter to be within [0, 10]    
    if (alphain <0 or alphain >10):
        print("Error: Alpha must be in the range [0, 10]")
        exit(0)    

    #check that the mode is 1, 2, 3, or 4
    if (MODE!=1 and MODE!=2 and MODE!=3 and MODE!=4):    
        print("Error: MODE must be 1, 2, 3, or 4")
        print("Modes: 1- GPU Original SuperSmoother")
        print("Modes: 2- GPU Single Pass SuperSmoother")
        print("Modes: 3- CPU Original SuperSmoother")
        print("Modes: 4- CPU Single Pass SuperSmoother")
        exit(0)

    ###############################
    

    #enumerate objId so that we can process objects with non-numeric Ids
    #original objects are stored in ret_uniqueObjectIdsOrdered
    start_index_arr, end_index_arr, ret_uniqueObjectIdsOrdered = computeIndexRangesForEachObject(objId)     
    objId = enumerateObjects(start_index_arr, end_index_arr)

    # Create variables that define C interface
    array_1d_double = npct.ndpointer(dtype=c_double, ndim=1, flags='CONTIGUOUS')
    array_1d_float = npct.ndpointer(dtype=c_float, ndim=1, flags='CONTIGUOUS')
    array_1d_unsigned = npct.ndpointer(dtype=c_uint, ndim=1, flags='CONTIGUOUS')

    #load the shared library (either the float or double version)
    lib_path = os.getcwd()
    if (dtype=="float"):
        libsupersmootherfloat = npct.load_library('libpysupsmufloat.so', lib_path)
    elif (dtype=="double"):     
        libsupersmootherdouble = npct.load_library('libpysupsmudouble.so', lib_path)

    #total number of rows in file
    sizeData=len(objId)
    print("[Python] Number of rows in file: %d" %(sizeData))

    #convert input from lists to numpy arrays
    objId=np.asarray(objId, dtype=int)
    timeX=np.asfarray(timeX)
    magY=np.asfarray(magY)
    magDY=np.asfarray(magDY)

    #convert to CTYPES
    if (dtype=="float"):
        c_objId=convert_type(objId, c_uint)
        c_timeX=convert_type(timeX, c_float)
        c_magY=convert_type(magY, c_float)
        c_magDY=convert_type(magDY, c_float)
    elif (dtype=="double"):     
        c_objId=convert_type(objId, c_uint)
        c_timeX=convert_type(timeX, c_double)
        c_magY=convert_type(magY, c_double)
        c_magDY=convert_type(magDY, c_double)

    df=(maxFreq-minFreq)/freqToTest*1.0

    # Allocate arrays for results 
    uniqueObjects=np.size(np.unique(objId))
    print("[Python] Unique objects: %d" % (uniqueObjects))

    if (dtype=="float"):
        ret_pgram = np.zeros(uniqueObjects*freqToTest, dtype=c_float)
        pgramDataGiB=((ret_pgram.size*4.0)/(1024*1024*1024))    
    elif (dtype=="double"): 
        ret_pgram = np.zeros(uniqueObjects*freqToTest, dtype=c_double)      
        pgramDataGiB=((ret_pgram.size*8.0)/(1024*1024*1024))    

    # extern "C" void SuperSmootherPy(unsigned int * objectId, DTYPE * timeX, DTYPE * magY, DTYPE * magDY, 
    # unsigned int sizeData, double minFreq, double maxFreq, unsigned int freqToTest, double alphain, int MODE)
    
    if (dtype=="float"):
        #define the argument types
        libsupersmootherfloat.SuperSmootherPy.argtypes = [array_1d_unsigned, array_1d_float, array_1d_float, 
        array_1d_float, c_uint, c_double, c_double, c_uint, c_double, c_int, array_1d_float]
        #call the library
        libsupersmootherfloat.SuperSmootherPy(c_objId, c_timeX, c_magY, c_magDY, c_uint(sizeData), c_double(minFreq), c_double(maxFreq), c_uint(freqToTest), c_double(alphain), c_int(MODE), ret_pgram)
    if (dtype=="double"):    
        #define the argument types
        libsupersmootherdouble.SuperSmootherPy.argtypes = [array_1d_unsigned, array_1d_double, array_1d_double, 
        array_1d_double, c_uint, c_double, c_double, c_uint, c_double, c_int, array_1d_double]
        #call the library
        libsupersmootherdouble.SuperSmootherPy(c_objId, c_timeX, c_magY, c_magDY, c_uint(sizeData), c_double(minFreq), c_double(maxFreq), c_uint(freqToTest), c_double(alphain), c_int(MODE), ret_pgram)

    
    


    print("[Python] Size of pgram in elems: %d (%f GiB)" %(ret_pgram.size, pgramDataGiB))

    
    #for convenience, reshape the pgrams as a 2-D array
    ret_pgram=ret_pgram.reshape([uniqueObjects, freqToTest])

    ret_periods=np.zeros(uniqueObjects)
    for x in range(0, uniqueObjects):
    	ret_periods[x]=1.0/(minFreq+(df*np.argmax(ret_pgram[x])))

    if(uniqueObjects>1):    
        print("[Python] Sum of all periods: %f" %(np.sum(ret_periods)))
    else:
        print("[Python] Period for object: %f" %ret_periods[0])

    return ret_uniqueObjectIdsOrdered, ret_periods, ret_pgram    





if __name__ == "__main__":
    

    fname="../data/SDSS_stripe82/SDSS_stripe82_band_z.txt"

    #for single object (first object from stripe82)
    # fname="../data/SDSS_stripe82/SDSS_stripe82_band_z_obj_2794912.txt"
    
    
    objIdArr=getColumn(fname, 0)
    timeXArr=getColumn(fname, 1)
    magYArr=getColumn(fname, 2)
    magDYArr=getColumn(fname, 3)


    
    min_f=0.1
    max_f=10.0
    N_f=330000
    alpha=9.0
    ssmode=1
    dtype="double"
    verbose="false"

    

    #use default parameters
    # objIds, periods, pgrams = supersmoother(objIdArr, timeXArr, magYArr, magDYArr, min_f, max_f, ssmode)    
    
    #with all parameters assigned    
    objIds, periods, pgrams = supersmoother(objIdArr, timeXArr, magYArr, magDYArr, min_f, max_f, ssmode, alpha, N_f, dtype, verbose)    

    print("[Python] Obj 0: objId: %s period: %f" %(objIds[0],periods[0]))

