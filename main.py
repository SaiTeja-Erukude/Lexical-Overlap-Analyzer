import spacy
import pandas as pd

def readFile(path):    
    try:
        with open(path, 'r') as file:
            return file.read()    

    except Exception as e:
        print(f"An error occurred: {e}")
        return ''


def processText(text):    
    # Load SpaCy English model
    nlp = spacy.load("en_core_web_sm")

    text = text.strip().lower()
    doc = nlp(text)

    freq = {}
    for token in doc:
        # if a token is an alpha and not a stop word
        if token.is_alpha and not token.is_stop and token.lemma_ != '-PRON-':
            if token.lemma_ in freq:
                freq[token.lemma_] += 1
            else:
                freq[token.lemma_] = 1

    return freq


if __name__ == '__main__':

    doc1_path = 'doc1.txt'
    doc2_path = 'doc2.txt'   
    
    print('Reading the documents ...')
    doc1 = readFile(doc1_path)
    doc2 = readFile(doc2_path)

    print('Processing the text ...')
    doc1_dict = processText(doc1)
    doc2_dict = processText(doc2)

    # Load the English word frequency dataset
    print('Loading English word frequency data ...')
    df = pd.read_csv('ngram_freq.csv')
    eng_dict = df.set_index('word')['count'].to_dict()


    print('Analyzing common words ...')
    default_val = 0
    res_dict = {}
    
    # optimize later : iterate the smallest dict
    
    for word in doc1_dict:
        f1 = doc1_dict.get(word, default_val)       # freq of word in doc1
        f2 = doc2_dict.get(word, default_val)       # freq of word in doc2
        fe = eng_dict.get(word, default_val)        # freq of word in english lang
        
        if f2 > 0:
            weight = (f1 - fe) * (f2 - fe)
            res_dict[word] = weight

    print('Result Dict: ', res_dict)

    # sort the result dict in descending order
    res_keys = sorted(res_dict, key=res_dict.get, reverse=True)

    # print 20 most common words from two documents
    print('\n20 most common words:')
    for i in range(20):
        print(res_keys[i], ': ', doc1_dict.get(res_keys[i], 0), doc2_dict.get(res_keys[i], 0))