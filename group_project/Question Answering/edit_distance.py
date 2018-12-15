import pandas as pd
from parsedata import generate_pairs
from nltk.metrics import edit_distance
import argparse



def edit_dis(list):
    """Retrieves the similar question in the test data with a users question"""

    # retrive dataframe
    df = generate_pairs()

    # convert to list for modeling
    x = df.Question.tolist()
    y = df.Answer.tolist()

    predicted = []

    for u_sent in list:

        x_index = -1  # initialize index for tracking index of similar question
        ini_val = 1000  # initialize edit distance for similar question
        
        for i in range(len(x)):
            # calculate edit distance
            val_dis = edit_distance(u_sent[0].split(), x[i].split())
            if(val_dis < ini_val):
                ini_val = val_dis
                x_index = i
        predicted.append(y[x_index])
    
    return predicted


def convert_to_list(txt):
    """takes a text file and returns a list"""
    data = pd.read_csv(txt, delimiter="\n", header=None)
    
    return data.values.tolist()


def write_to_file(file_path, my_list):
    """write to files"""
    with open(file_path, 'w') as f:
        for item in my_list:
            f.write(item + "\n")



if __name__ == "__main__":
    # accept commandline argument
    parser = argparse.ArgumentParser()
    parser.add_argument('task', help='task name: qa => question answering')
    parser.add_argument('file', help='Questions file e.g question_file.txt')

    args = vars(parser.parse_args())

    # extract arguments passed from command-line
    task = args.get('task', None)
    test_file = args.get('file', None)

    if task == "qa" and test_file:
        u_list = convert_to_list(test_file)
        predicted = edit_dis(u_list)
        write_to_file("predicted.txt", predicted)
