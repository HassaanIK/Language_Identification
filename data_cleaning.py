from data_analysis import df
from nltk.tokenize import word_tokenize
import re
import pandas as pd
import nltk

#Removing Duplicates
# df = df.drop_duplicates(subset='Text')
# df = df.reset_index(drop=True)

nltk.download('punkt')
# Initialize the set of non-alphanumeric characters to remove
nonalphanumeric = ['\'', '.', ',', '\"', ':', ';', '!', '@', '#', '$', '%', '^', '&',
                   '*', '(', ')', '-', '_', '+', '=', '[', ']', '{', '}', '\\', '?', 
                   '/', '>', '<', '|', ' ']

def clean_text(text):
    """
    Function to clean and preprocess text data.
    """
    # Tokenize the text using spaCy
    tokens = word_tokenize(text)
    
    # Remove non-alphanumeric characters
    words = [word.lower() for word in tokens if word not in nonalphanumeric]
    
    # Join the lemmatized words back into a single string
    cleaned_text = " ".join(words)
    
    return cleaned_text

def remove_english(text):
    """
    function that takes text as input and returns text without english words
    """
    pat = "[a-zA-Z]+"
    text = re.sub(pat, "", text)
    return text


#applying clean_text function to all rows in 'Text' column 
# df['clean_text'] = df['Text'].apply(clean_text)



# #Removing English from Chinese text
# df_Chinese = df[df['language']=='Chinese']  # Chinese data in dataset

# clean_text = df.loc[df.language=='Chinese']['clean_text']
# clean_text = clean_text.apply(remove_english)  # removing English words
# df_Chinese.loc[:,'clean_text'] = clean_text

# # Concatenate the original DataFrame with the cleaned Chinese text DataFrame
# df = pd.concat([df, df_Chinese], axis=0, ignore_index=True)

# # Drop rows with 'Chinese' language from the original DataFrame
# df = df[~df['language'].isin(['Chinese'])].reset_index(drop=True)


# # shuffling dataframe and resetting index
# df = df.sample(frac=1).reset_index(drop=True)