import pandas as pd
from pathlib import Path

def readData(txt):
    data = pd.read_csv(txt, delimiter="\n", header=None)
    # remove sentence numbering
    data[0] = data[0].apply(lambda x: str(x).lower())

    return data[0]


def generate_pairs():
    # create main dataframe
    col_names = ["Topic", "Question", "Answer"]
    df = pd.DataFrame(columns=col_names)

    # create pandas dataframes for files
    topics = readData("dataset/processed_data/Topics.txt")
    questions = readData("dataset/processed_data/Questions.txt")
    answers = readData("dataset/processed_data/Answers.txt")

    # create large data frame
    df["Topic"]=topics.apply(lambda x: str(x).lower())
    df["Question"]=questions.apply(lambda x: str(x).lower())
    df["Answer"]=answers.apply(lambda x: str(x).lower())

    return df




if __name__ == "__main__":
    df=generate_pairs()
    # write_to(df)
    # print(df.head())
