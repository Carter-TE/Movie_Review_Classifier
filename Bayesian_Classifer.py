"""Assignment 4C

Aurthors:   Carter Edmond
"""
import re
import os
import numpy as np
from numpy.core.fromnumeric import sort
import pandas as pd
import math
import random

class BayesianClassifier():
    def __init__(self, pos_path='pos', neg_path='neg' ):
        self.pos_counts = dict()
        self.neg_counts = dict()
        self.pos_ev = list() 
        self.neg_ev = list()
        self.rm_ev = list()
        self.attr_usefulness = dict()
        self.attr = dict()
        self.top_attr = dict()
        self.pos_folder_path = pos_path
        self.neg_folder_path = neg_path
        self.pos_train_data, self.pos_test_data = self.split_train_test(self.pos_folder_path)     # data type: dataframe - single column of file names
        self.neg_train_data, self.neg_test_data = self.split_train_test(self.neg_folder_path)

        self.pos_counts = self.count_words(self.pos_train_data, self.pos_counts, self.pos_folder_path)
        self.neg_counts = self.count_words(self.neg_train_data, self.neg_counts, self.neg_folder_path)
        self.initialize()
    
    def initialize(self):
        pos_word_count = sum(self.pos_counts.values())
        neg_word_count = sum(self.neg_counts.values())
        total_words = neg_word_count + pos_word_count

        # Calculates mutual independence of each attribute
        for word in self.pos_counts:
            #
            pos_occs = self.pos_counts[word]
            try:
                neg_occs = self.neg_counts[word]
            except KeyError:
                continue
            pos_mi = self.mutual_information(pos_occs, total_words, (pos_occs+neg_occs), pos_word_count)
            neg_mi = self.mutual_information(neg_occs, total_words, (pos_occs+neg_occs), neg_word_count)

            usefullness = abs(pos_mi - neg_mi)
            self.attr[word] = self.calc_raw_probability(word)

            self.attr_usefulness[word] = usefullness
                    
        self.update_top_attr()

            
            
    def train(self, n, x):
        """Trains classifer by randomly removing attributes add adding back in the ones with 
        the highest usefullness value. Updates pos_ev, neg_ev, and rm_ev fields
        
        :param: n: Number of attributes to remove
        :param: x: Number of attributes to add back
        """

        positive = {}
        negative = {}
        rm_attr = {}
        rm_word_usefullness = {}

        print("Removing attributes")
        for i in range(n):
            if len(self.attr) == 0:
                break
            word, prob = random.choice(list(self.attr.items()))
            rm_attr[word]  = self.attr.pop(word)

            if prob[0] in self.neg_ev or prob[1] in self.pos_ev:
                print("Attribute probability overlap")
            positive[word] = prob[1]
            negative[word] = prob[0]
            rm_word_usefullness[word] = self.attr_usefulness[word]

        self.pos_ev = [*list(positive.items())]
        self.pos_ev.sort(reverse=True, key=lambda s:s[1]) # sorted list of top positve evidence by raw probability
        self.neg_ev = [*list(negative.items())]
        self.neg_ev.sort(reverse=True, key=lambda s:s[1]) # sorted list of top negative evidence by raw probability
        
        self.rm_ev = [*list(rm_word_usefullness.items())] # sorted list of top evidence by highest MI
        self.rm_ev.sort(reverse=True, key=lambda s:s[1])

        print("Adding back attributes")
        count = 0
        for j in range(x):
            word = self.rm_ev[0][0]
            if self.rm_ev[0][1] > .50:
                count += 1
                self.attr[word] = self.calc_raw_probability(word) # Adds word and raw probability back to list of attr
                
                # Updates attr_usefulness to later update top attrs
                item = self.rm_ev.pop(0)
                self.attr_usefulness[item[0]] = item[1]
                
        self.update_top_attr()

        print(count,"/",x," total added back")

    def split_train_test (self, folder_path):
        """Split the data into training and testing data
        """
        files = os.listdir(os.path.abspath(folder_path))
        file_names = list(filter(lambda file: file.endswith('.txt'), files))
        
        df = pd.DataFrame(file_names)
        # approximate 9:1 ratio between training and test split
        mask = np.random.rand(len(df)) <=.9
        training_data = df[mask] #90% extracted
        testing_data = df[~mask] #remaining 10% extracted

        print(f"No. of training examples : {training_data.shape[0]}")
        print(f"No. of testing examples: {testing_data.shape[0]}")

        return training_data, testing_data

    def mutual_information(self, a, b, c, d):
        """Computes Mutual Information of a given word based on its occurences

        :param a: Occurrences in given class
        :param b: Total words
        :param c: Total occurrences 
        :param d: Total words in given class

        :return Mutual information value
        """
        first_part = ((a*b)/(c*d))
        return (math.log(first_part,2))

    def count_words(self, df, word_dict, dir_path):
        """Counts the total occurences of all words in a folder of text files. 
        Files chosen are based on df; which filters through the dir_path.
        
        :param df: Dataframe containing the names of the files to check
        :param word_dict: Dictionary containing words and accumulated occurrences
        :param dir_path: Path to folder containing text files
        :return dict: Returns dictionary with updated word occurence values
        """

        # Combines all text of files into single string
        text = str()
        for i in range(len(os.listdir(pos_file_path))):
            try:
                with open(os.path.join(dir_path, df.at[i, 0]), 'r') as f:
                    text += f.read()
            except KeyError:
                continue
        
        # Parses text removing punctuation
        chars = re.compile(r"[^a-zA-Z0-9-\s]")
        text = chars.sub("",text)
        
        # Counts the word occurrneces line by line 
        words = text.split()
        for word in words:
            if len(word) < 6:
                continue
            if word in word_dict:
                word_dict[word] += 1
            else: 
                word_dict[word] = 1
        
        return word_dict
    
    def update_top_attr(self):
        sorted_usefulness = list(self.attr_usefulness.items())
        sorted_usefulness.sort(reverse=True, key=lambda s:s[1])
        n = len(sorted_usefulness)   
        for i in range(int(n*.1)):
            item = sorted_usefulness.pop(0)
            self.attr_usefulness.pop(item[0])
            try: 
                self.top_attr[item[0]] = self.attr.pop(item[0])
            except KeyError:
                i-=1

    def calc_raw_probability(self, word):
        """Calculates P(word|Class)

        :return: tuple: (- probability, + probability)
        """

        a = 0
        pos_prob = (self.pos_counts[word] + a)/ (sum(self.pos_counts.values()) + a) # Requires laplace smoothing
        neg_prob = (self.neg_counts[word] + a)/ (sum(self.neg_counts.values()) + a)
        return (neg_prob, pos_prob)
    
    def get_attributes(self):
        return dict(self.attr)

    def get_top_attributes(self):
        return dict(self.top_attr)
    
    def get_test_file(self):
           pos_file = os.path.join(self.pos_folder_path, self.pos_test_data.iat[0, 0])
           neg_file = os.path.join(self.neg_folder_path, self.neg_test_data.iat[0, 0])
           return(neg_file, pos_file)

class MovieReviewClassifier():
    def __init__(self, pos_folder='pos', neg_folder='neg'):
        self.bc = BayesianClassifier(pos_folder, neg_folder)
        self.attributes = self.bc.get_attributes()
        self.top_attributes = self.bc.get_top_attributes()
        self.test_files = self.bc.get_test_file()
        
    def learn(self, rm=250, add_back=125):
        current_attr = dict(self.attributes)
        cycles = 1
        self.bc.train(rm, add_back)

        print("Toatal Training Cycles: ", cycles)
        self.attributes = self.bc.get_attributes()
        self.top_attributes = self.bc.get_top_attributes()
        #self.attributes += self.bc.get_probabilities()

    def compute_probability(self, text=None):
        """Computes the probability of a class given all the attributes"""
        pos_prob = .5
        neg_prob = .5
        attr_count = 0
        for attr in self.top_attributes:            
            if  attr not in text: # text is not None and
                continue
            
            n = 1/(pos_prob+neg_prob)
            pos_prob = n*pos_prob
            neg_prob = n*neg_prob

            pos_prob = (pos_prob * self.top_attributes[attr][1]) + 1
            neg_prob = (neg_prob * self.top_attributes[attr][0]) + 1 
            
            
            attr_count += 1

        

        top_attr_prob = (neg_prob, pos_prob)
        #print("Based on ",attr_count,"top attrs: ",top_attr_prob)

        for attr in self.attributes:            
            if  attr not in text: # text is not None and
                continue
            #if pos_prob + neg_prob < 1e-200:
                # Normalize
            n = 1/(pos_prob+neg_prob)
            pos_prob = n*pos_prob
            neg_prob = n*neg_prob

            pos_prob = (pos_prob * self.attributes[attr][1]) #+ 1
            neg_prob = (neg_prob * self.attributes[attr][0]) #+ 1 
            
           
            attr_count += 1
        
        
        #print("Attributes included: ",attr_count,"/",len(self.attributes)+len(self.top_attributes))    
        return neg_prob, pos_prob, top_attr_prob
    
    def test(self):
        neg_correct = 0
        pos_correct = 0
        pos_tests = self.bc.pos_test_data
        neg_tests = self.bc.neg_test_data
        tot_tests = 0
    
        for i in range(len(os.listdir(pos_file_path))):
            try:
                file = os.path.join(pos_file_path, pos_tests.at[i, 0])
            except KeyError:
                continue
            prediction = classifier.classify(file)
            tot_tests += 1
            if prediction[0] == 1:
                pos_correct += 1

        for i in range(len(os.listdir(neg_file_path))):
            try:
                file = os.path.join(neg_file_path, neg_tests.at[i, 0])
            except KeyError:
                continue
            prediction = classifier.classify(file)
            tot_tests += 1
            if prediction[0] == 0:
                neg_correct += 1


        print("Total Tests: ",tot_tests)
        print("Pos Correct: ",pos_correct,"/", pos_tests.shape[0])
        print("Neg Correct: ",neg_correct,"/", neg_tests.shape[0])
        print("Tot Correct: ",pos_correct + neg_correct,"/", tot_tests)
        return pos_correct + neg_correct, tot_tests
            
    def classify(self, file):
        text = str()
        try:
            with open(file, 'r') as f:
                text = f.read()
        except KeyError:
            print("Invalide file name/path")
            return

        neg_prob, pos_prob, top_attr_probs = self.compute_probability(text)

        confidence = (neg_prob,pos_prob)
        prediction = max(confidence)

        pos_confidence = round(confidence[1]*100)
        neg_confidence = round(confidence[0]*100)
        if(prediction == confidence[1]):
            # print("Positive Review; Confidence: ",pos_confidence,"%")
            return 1, pos_confidence
        else:
            # print("Negative Review; Confidence: ",neg_confidence,"%")
            return 0, neg_confidence


# Main to test functionality of tokenizer counter
if __name__ == '__main__':
    pos_file_path = 'pos' # input("Input positive file path: ")
    neg_file_path = 'neg' # input("Input negative file path: ")

    classifier = MovieReviewClassifier(pos_file_path,neg_file_path)
    test_files = classifier.bc.get_test_file()
    classifier.test()
    classifier.learn(5000,3000)
    correct, total = classifier.test()
    accuracy = int((correct/total)*100)
    i = 0
    while accuracy < 85:
        classifier.learn(1000,500)
        correct, total = classifier.test()
        accuracy = int((correct/total)*100)
        i+=1

        if i == 10: 
            break

   
    
    
    


    
    



