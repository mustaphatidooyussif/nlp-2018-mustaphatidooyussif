import argparse
import random
import numpy as np
import pandas as pd
from string import punctuation
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import TfidfVectorizer

class CommandlineArgument(object):
    """Class for command line argument parsing
    This classs takes two commandline arguments and returns the argument object."""
    def __init__(self):
        self.args = self.parse_arguments()
        self.questions_file = self.args.questions_file

    def getTask(self):
        return self.args.task #returns the task

    def parse_arguments(self): #command line arguments parser
        """Commandline arguments parser: 
        Takes to command line arguements and returns the argument parser.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('task', help='task name: topic or answer')
        parser.add_argument('questions_file', help='Guestions file e.g question_file.txt')
        args = parser.parse_args()

        return args

class TopicModelling(CommandlineArgument):
    
    def __init__(self):
        super().__init__()
        self.task = "topic"

    def getTask(self):
        return self.task

    def predit_topic(self):
        print("Please modify me to contain the code for the topic modeling")

class QuestionAnswering(CommandlineArgument):
    """Question answering model

    This class predicts the answers to some given sets of questions. """
    def __init__(self):
        super().__init__()
        self.args = self.parse_arguments()
        # self.questions_file = self.args.questions_file
        self.train_questions =  self.read_dataset('Questions.txt')
        self.test_questions = self.read_dataset(self.questions_file)
        self.answers = self.read_dataset('Answers.txt')

    def cos_sim(self, a, b):
        """Takes 2 vectors a, b and returns the cosine similarity according 
        to the definition of the dot product
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    def read_dataset(self, filename):
        """Takes string filename and returns a list of the read file content."""
        f = open(filename, 'rb')
        data = []
        for datum in f:
            # datum = datum.strip()
            datum =  datum.decode("utf-8") 
            data.append(self.replace_chars(datum))

        return data

    def split_to_train_test(self, test_ratio = 0.20):
        """This function splits the dataset into training and 
        testing datasets."""
        # Get the train questions and the test questions
        questions = self.train_questions
        answers = self.test_questions

        train_question = []
        test_question = []
        train_answers = []
        test_answers = []

        num_of_train_questions = int(len(questions) * (1-test_ratio)) 
        
        #split the documents into testing and training dataset.
        while len(train_question) < num_of_train_questions:  
            index = random.randrange(20)  
            train_question.append(questions.pop(index))  #randomly add documents to training document set
            train_answers.append(answers.pop(index)) 
            
        return train_question, train_answers, test_question, test_answers

    # Removing all stop words and punctuations from a given sentence
    # and changing the sentence to lowercase sentence
    def replace_chars(self, sentence):
        stopwords = set(list(stop_words.ENGLISH_STOP_WORDS) + ['\t', '\n'])
        punctuations = list(punctuation)
        for char in punctuations:        #Replace all punctuation marks with an empty string
            sentence = sentence.replace(str(char), '')
        words_filtered = [word for word in sentence.lower().split() if word not in stopwords and len(word) > 2]
        return ' '.join(words_filtered)

    def answer_questions_using_cosine_sim(self, questions = None):
        """This returns the answer to a given question."""
        vectorizer= TfidfVectorizer()
        
        if questions is None:
            test_question = self.test_questions

        #fit train questions
        #transform trainset and change to array
        X = vectorizer.fit(self.train_questions) 
        array = X.transform(self.train_questions).toarray()

        ##transform testset and change to numpy array
        test_array = X.transform(test_question).toarray() 

        #find the cosine similarity between the questions and the test question. 
        answers = []
        for i in range(len(test_question)):
            max_value = 0
            for j in range(len(self.train_questions)):
                if max_value < self.cos_sim(array[j], test_array[i]):
                    max_value = self.cos_sim(array[j], test_array[i])
                    answer_index = j        #get the index of the current most similar question
            answers.append(self.answers[answer_index])  #get the answer of the most similar question.
        return answers

    def write_results(self, filename, answer_list):
        """Takes string file name, and list of strings and writes the 
        content of the list into a file with the specified file name."""
        with open(filename, 'w') as f:
            for answer in answer_list:
                f.write(str(answer) + "\n")

def main():
    arguments = CommandlineArgument() 

    if arguments.getTask().lower() == "topic":
        topic = TopicModelling()
        print("Please call the top modeling call here...")
    elif arguments.getTask().lower() == "qa":
        qa= QuestionAnswering()
        answer = qa.answer_questions_using_cosine_sim()
        qa.write_results("qa_results.txt", answer)
    else:
        raise Exception("Program does not support the requested task")

if __name__ == "__main__":
    main()