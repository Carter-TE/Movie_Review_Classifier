import json
import math
import os
import csv

from numpy.core.numeric import outer

class MovieReviewClassifier():
    def __init__(self, classifier_path='attributes.json'):
        with open(classifier_path) as f:
            self.attributes = json.load(f)

    def classify(self, file):
        text = str()
        try:
            with open(file, 'r') as f:
                text = f.read()
        except KeyError:
            print("Invalide file name/path")
            return

        neg_prob, pos_prob = self.compute_probability(text)

        liklihood = (neg_prob,pos_prob)
        prediction = max(liklihood)

        if(prediction == liklihood[1]):
            return 1
        else:
            return 0

    def compute_probability(self, text=None):
        """Computes the probability of a class given all the attributes"""
        attr_count = 0

        present_pos_probs = [math.log(self.attributes[attr][1]) for attr in self.attributes if attr in text]
        present_neg_probs = [math.log(self.attributes[attr][0]) for attr in self.attributes if attr in text]

        attr_count = len(present_pos_probs)
        pos_prob = 0
        neg_prob = 0

        for i in range(attr_count):
            pos_prob = pos_prob + present_pos_probs[i]
            neg_prob = neg_prob +  present_neg_probs[i]

        return neg_prob, pos_prob

if __name__ == '__main__':
    reviews_path = input("Enter path to folder with movie reviews: ")
    classifier_path = input("Enter path to Bayesian classifier: ")

    classifier = MovieReviewClassifier(classifier_path)

    with open('output.csv', 'w'):
        pass

    for file_name in os.listdir(reviews_path):
        file = os.path.join(reviews_path, file_name)
        prediction = classifier.classify(file)
        line = [file_name, prediction]

        with open('output.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(line)
    
    correct = 0
    total = 0 
    with open('output.csv') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if row == []:
                continue
            if row[1] == '0':
                correct += 1
            total += 1
    

    print("Accuracy: ",correct,"/",total)