# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:13:31 2019

@author: alexn
"""
import pandas as pd
import numpy as np
import sys

"""
review.json file:
    685900 Lines
    
    columns = ['business_id', 'cool', 'date', 'funny', 'review_id', 'stars', 'text',
       'useful', 'user_id']
"""

def transform_data(filename, output,  iterations=10):
    data = pd.read_json(filename)
    convert_data = data[['stars', 'text']].dropna().sort_index()

    size =  convert_data.shape[0] / iterations    
    for i in range(iterations):
        convert_data[int(i*size):int((i+1)*size)].to_json(output + str(i) + '.json')
        

def chunk_data(filename, output):
    data = pd.read_json(filename, lines=True, chunksize = 100000)
    i = 0
    for chunks in data:
        chunks.to_json(output + str(i) + '.json')
        i += 1

def main():
    #transform_data('review-0.json', 'review-0-')
    data = pd.read_json('data/review-0-0.json')
    
    #full_data = pd.concat([data1, data0])
    data1 = data[(data['stars'] > 3) | (data['stars'] < 3)]
    data2 = data[(data['stars'] == 1) | (data['stars'] == 5)]
    data3 = data[(data['stars'] == 2) | (data['stars'] == 4)]
    data4 = data[(data['stars'] == 1) | (data['stars'] == 5) | (data['stars'] == 3)]

    data1[0:500].to_json("data/review500-3.json")
    data2[0:500].to_json("data/review500-1-5.json")
    data3[0:500].to_json("data/review500-2-4.json")
    data4[0:500].to_json("data/review500-1-3-5.json")
    data[0:500].to_json('data/review500-all.json')
    
if __name__ == "__main__":
    main()
    

