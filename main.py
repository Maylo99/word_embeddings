# This is a sample Python script.
from WordEmbeddings import WordEmbeddings
from semantic_search import SemanticSearch
from openai_init import OpenaiInit
from classifier.classifier import Classifier

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    user_input = input("Enter 'y' or 'n': ")
    openai_init= OpenaiInit()
    if user_input.lower() == 'y':
        word_embeddings = WordEmbeddings(openai_init.openai_key)
        word_embeddings.embedding()
    elif user_input.lower() == 'n':
        print("The if statement will not be executed.")
    else:
        print("Invalid input. Please enter 'y' or 'n'.")

    semantic_search = SemanticSearch(openai_init.openai_key)
    semantic_search.get_sorted_categories()

    #classifier= Classifier("classifier/dataset.csv")
    #classifier.classify()





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
