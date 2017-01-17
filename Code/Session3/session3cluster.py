#!/bin/env python
import sys
sys.path.append('.')

import session3
import re

if __name__ == '__main__':
    levels_pyramid = 0
    randomSplits = True
    usePCA=0
    
    try:
        if len(sys.argv) < 3:
            raise ValueError, "Needed number of slots and detector type."
        num_slots = int(sys.argv[1])
        descriptor_type = sys.argv[2]
        index = 3
        while index < len(sys.argv):
            if sys.argv[index].lower() == '--levelsp':
                lps=sys.argv[index + 1]
                if lps=='0':
                    levels_pyramid=0
                else:
                    p = re.compile('\[(\d.+?\d)\]')
                    rs=p.findall(lps)
                    if len(rs)==0:
                        raise ValueError, "The number of levels must be [[x,x],[y,y]]."
                    levels_pyramid=[]
                    for r in rs:
                        levels=r.split(',')
                        levels_pyramid.append([int(levels[0]),int(levels[1])])
                index += 2
            elif sys.argv[index].lower() == '--norandoms':
                randomSplits = False
                index += 1
            elif sys.argv[index].lower() == '--pca':
                usePCA=int(sys.argv[index + 1])
                index += 2
            else:
                raise ValueError, "Unknown parameter: %s" % (sys.argv[index])
    except:
        print "Unhanded exception while parsing the user parameters:", sys.exc_info()[1]
        print ""
        print "Syntax:"
        print "$ python launchcluster.py <number_of_slots> <descriptor_type> (--levelsp <pyramid_levels> | --usehik | --norandoms)*"
        print ""
        print "where:"
        print "        <number_of_slots>    number of processes used in multiprocessing."
        print "        <descriptor_type>    descriptor type (DENSE, SIFT, SURF, ORB, HARRIS)."
        print " --levelsp <pyramid_levels>  number of levels of the spatial pyramid, if used (default=0)."
        print "           --norandoms       disable random validation splits for cross validation."
        print "         --pca <pca>         pca input, if used."
        print ""
        raise
    
    print "Using %s detector, randomSplits=%s, levels_pyramid=%s, usePCA=%s" % (descriptor_type,randomSplits,levels_pyramid,usePCA)
    session3.launchsession3(num_slots,descriptor_type,randomSplits,levels_pyramid,usePCA)