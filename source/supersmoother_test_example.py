import csv
import numpy as np

#load the supersmoother python library
import supersmoothergpu as ss


def getColumn(filename, column):
    results = csv.reader(open(filename), delimiter=",")
    # next(results, None)  # skip the headers
    return [result[column] for result in results]

if __name__ == "__main__":
    
    fname="SDSS_stripe82_band_z.txt"
    
    objIdArr=getColumn(fname, 0)
    timeXArr=getColumn(fname, 1)
    magYArr=getColumn(fname, 2)
    magDYArr=getColumn(fname, 3)

    #parameters
    min_f=0.1
    max_f=10.0
    N_f=330000
    alpha=9.0
    ssmode=1
    dtype="float"
    verbose="false"

    

    #use default parameters
    objIds, periods, pgrams = ss.supersmoother(objIdArr, timeXArr, magYArr, magDYArr, min_f, max_f, ssmode)    
    
    #with all parameters assigned    
    # objIds, periods, pgrams = ss.supersmoother(objIdArr, timeXArr, magYArr, magDYArr, min_f, max_f, ssmode, alpha, N_f, dtype, verbose)    

    print("[Python] Obj 0: objId: %s period: %f" %(objIds[0],periods[0]))

