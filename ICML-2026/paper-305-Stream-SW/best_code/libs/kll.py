#!/usr/bin/python
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Written by Edo Liberty. 
'''

import sys
from random import random
from math import ceil

class  KLL:
    def __init__(self, k, c = 2.0/3.0, lazy=True, alternate=True):
        if k<=0: raise ValueError("k must be a positive integer.")
        if c <= 0.5 or c > 1.0: raise ValueError("c must larger than 0.5 and at most 1.0.")
        self.k = k
        self.c = c
        self.lazy = lazy
        self.alternate = alternate
        self.compactors = []
        self.H = 0
        self.size = 0 
        self.maxSize = 0
        self.grow()
        
    def grow(self):
        self.compactors.append(Compactor(self.alternate))
        self.H = len(self.compactors)
        self.maxSize = sum(self.capacity(height) for height in range(self.H))

    def capacity(self, height):
        depth = self.H - height - 1
        return int(ceil(self.c**depth*self.k)) + 1
    
    def update(self, item):
        self.compactors[0].append(item)
        self.size += 1
        if self.size >= self.maxSize:
            self.compress()
            assert(self.size < self.maxSize)
    def update_batch(self,items):
        for item in items:
            self.update(item)
    def compress(self):
        for h in range(len(self.compactors)):
            if len(self.compactors[h]) >= self.capacity(h):
                if h+1 >= self.H: self.grow()
                self.compactors[h+1].extend(self.compactors[h].compact())
                self.size = sum(len(c) for c in self.compactors)
                if(self.lazy):
                    break

    def merge(self, other):
        # Grow until self has at least as many compactors as other
        while self.H < other.H: self.grow()
        # Append the items in same height compactors 
        for h in range(other.H): self.compactors[h].extend(other.compactors[h])
        self.size = sum(len(c) for c in self.compactors)
        # Keep compressing until the size constraint is met
        while self.size >= self.maxSize:
            self.compress()
        assert(self.size < self.maxSize) 
        
    def rank(self, value):
        r = 0
        for (h, c) in enumerate(self.compactors):
             for item in c:
                 if item <= value:
                     r += 2**h
        return r
    def items(self):
        all_items= []
        for (h, items) in enumerate(self.compactors):
            all_items.extend(item for item in items)
        return all_items
    def items_check(self):
        for h in range(self.H):
            print(self.capacity(h))
        all_items= []
        for (h, items) in enumerate(self.compactors):
            all_items.append(items)
        return all_items
    def cdf(self):
        itemsAndWeights = []
        for (h, items) in enumerate(self.compactors):
             itemsAndWeights.extend( (item, 2**h) for item in items )
        totWeight = sum( weight for (item, weight) in itemsAndWeights)
        itemsAndWeights.sort()
        cumWeight = 0
        cdf = []
        for (item, weight) in itemsAndWeights:
            cumWeight += weight
            cdf.append( (item, float(cumWeight)/float(totWeight) ) )
        return cdf
    def empirical_distribution(self):
        supports = []
        weights =[]
        for (h, items) in enumerate(self.compactors):
            supports.extend(item for item in items)
            weights.extend(2 ** h for item in items)
        return supports, weights


    def ranks(self):
        ranksList = []
        itemsAndWeights = []
        for (h, items) in enumerate(self.compactors):
             itemsAndWeights.extend( (item, 2**h) for item in items )
        itemsAndWeights.sort()
        cumWeight = 0
        for (item, weight) in itemsAndWeights:
            cumWeight += weight
            ranksList.append( (item, cumWeight) )
        return ranksList

class Compactor(list):
    def __init__(self, alternate=True):
        self.numCompaction = 0
        self.offset = 0
        self.alternate = alternate

    def compact(self):
        if (self.numCompaction%2==1 and self.alternate):
            self.offset = 1 - self.offset
        else:
            self.offset = int(random() < 0.5)

        self.sort()

        lastItem = None
        if (len(self)%2==1):
            lastItem = self.pop(-1)

        for i in range(self.offset,len(self),2):
            yield self[i]

        self.clear()
        if lastItem is not None:
            self.append(lastItem)

        self.numCompaction += 1

                          
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=128, 
                        help='''controls the number of elements in the sketch which is 
                        at most 3k+log2(n). n is the length of the stream.''')
    parser.add_argument('-t', type=str, choices=["string","int","float"], default='string',
                        help='defines the type of stream items, default="string".')
    args = parser.parse_args()
    
    k = args.k if args.k > 0 else 128
    conversions = {'int':int,'string':str,'float':float}
         
    kll = KLL(k)
    for line in sys.stdin:
        item = conversions[args.t](line.strip('\n\r'))
        kll.update(item)
        
    for (item, quantile) in kll.cdf():
        print('%f\t%s'%(quantile,str(item)))
        
