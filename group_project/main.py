import argparse
import random
import nltk 
import numpy as np
import pandas as pd
from string import punctuation
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import scale 


class CommandlineArgument(object):
    """Class for command line argument parsing
    This classs takes two commandline arguments and returns the argument object."""
    def __init__(self):
        self.args = self.parse_arguments()
        self.questions_file = self.args.questions_file

    def getTask(self):
        return self.args.task #returns the task

    def getFile(self):
        return self.questions_file

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
        self.clf = LogisticRegression() #This variable is for our classifier 
        self.X_train = None #This variable is for our train data sentences 
        self.X_test = None #This variable is for our test data sentences 
        self.y_train = None #This variable is for our train data predictions 
        self.y_test = None #This varible is for our test data  predictions 
        self.vector = None 


    #This function reads in the data from a file and splits it into training and test data    
    def read(self,file, file2):
        df = pd.read_csv(file, sep='\t', names= ["questions"])
        df2 = pd.read_csv(file2, sep='\t', names =['topics'])
        df['topics'] = df2.topics
        df['topics_id'] = df2['topics'].factorize()[0]
        stopword_set = set(stopwords.words('english'))
        self.vector = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
        data_predict = df.topics
        data_txt = self.vector.fit_transform(df.questions).toarray()
        data_txt.shape
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data_txt,data_predict,test_size=0.33,random_state=0)
        
    #This function trains the classifier with the training data
    def train(self):
        clf = self.clf.fit(self.X_train, self.y_train)
        
    #This function predicts the test file and writes the results to a file named results.txt   
    def predict(self,file):
        testList = []
        file = open(file, 'r')
        for line in file.readlines():
            testList.append(line)
        test_list_vector = self.vector.transform(testList)
        results = self.clf.predict(test_list_vector)
        result_list = list(results)
        file_result = open("topic_results.txt","w")
        for result in result_list:
            file_result.write(str(result)+"\n")
        file_result.close()
        
    #This function checks the accuracy of the classifier     
    def accuracy(self):
        accuracy = self.clf.score(self.X_test,self.y_test)
        print (accuracy)
        
        
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
        log = TopicModelling()
        log.read('Questions.txt','Topics.txt')
        log.train()
        # log.accuracy()
        log.predict(arguments.getFile())
        print("Please call the top modeling call here...")
    elif arguments.getTask().lower() == "qa":
        qa= QuestionAnswering()
        answer = qa.answer_questions_using_cosine_sim()
        qa.write_results("qa_results.txt", answer)
    else:
        raise Exception("Program does not support the requested task")

if __name__ == "__main__":
    main()