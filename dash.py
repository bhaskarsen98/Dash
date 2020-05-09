#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 19:15:38 2019

@author: bhaskar
"""
import nltk

#for the english stopwords present in the nltk
from nltk.corpus import stopwords

#for the cosine_distance() function present in nltk.cluster.util 
import numpy as np
from nltk.cluster.util import cosine_distance

#imports networkx 2.4.0
#backcompatibilty of this code lies till networkx 1.1.0
import networkx as nx

#for plotting undirected graph using cosine matrix
#import matplotlib.pyplot as plt
#%%
#reads the orignal atricle from a txt file

def read_article(file_name):
    file = open(file_name, "r")
    filedata = file.readlines()
    
    #makes a list of strings that are sentences present in the article
    article = filedata[0].split(". ")
    sentences = []
    for sentence in article:
        print(sentence)
        #depending on your version of conda ,any one of the below lines would work fine
        #seprates the words in the string article[sentence]
        #1
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
        #2
        #sentences.append(re.sub('[^a-zA-Z]',' ', sentence).split(' '))
        
    #leaves the last     
    sentences.pop() 
    
    return sentences


#%%
#returns cell value for cosine matrix
#iterates over the sentences[] list and calculates the cosine distance between them
#cosine_distance(u, v):
#"""
#Returns 1 minus the cosine of the angle between vectors v and u. This is
#equal to 1 - (u.v / |u||v|).
#"""
#return 1 - (numpy.dot(u, v) / (sqrt(numpy.dot(u, u)) * sqrt(numpy.dot(v, v))))

def sentence_similarity(sent1, sent2, stopwords=None):
    
    #use empty list if stopwords aren't present for given language 
    if stopwords is None:
        stopwords = []
    
    #convert the words to lowercase for nltk.stopwords[]
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    #remove redundant words    
    all_words = list(set(sent1 + sent2))
    
    #initialize vector for sent1 and sent2
    #a numpy array can also be used by making further alterations in the code 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        
        #remove stopwords
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        
        #remove stopwords
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
        
    #cosine distance calculated according to above formula
    return 1 - cosine_distance(vector1, vector2)


#%%
#building cosine similarity matrix 

def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    #iterates over every element of list sentences[] to calculte cosine distance  
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            
            #ignore if both are same sentences
            if idx1 == idx2:
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


#%%

def generate_summary(file_name, top_n):
    nltk.download("stopwords")
    stop_words = stopwords.words('english')
    summarize_text = []

    ## Step 1 - Read text anc split it
    sentences=  read_article(file_name)
    top_n=top_n
    
    ## Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    ## Step 3 - Rank sentences in similarity martix
    #creates an undirected graph using the square matrix obtained form build_similarity_matrix()
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix,parallel_edges=True, create_using=None)
    #draw the unlabled graph
    #nx.draw(sentence_similarity_graph)
    #plt.draw()
    
    #converts the undirected graph to a directed graph by adding two dircted graph for each undirected edge
    #writes weights for the sentences
    scores = nx.pagerank(sentence_similarity_graph)

    ## Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    print("Indexes of top ranked_sentence order are ", ranked_sentence)    
    
    #adds top_n sentences to the summarized text
    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    ## Step 5 - output the summarize text
    print("Summarize Text: \n", ". ".join(summarize_text))


#%%
    
# let's begin
generate_summary( "tt.txt",5)