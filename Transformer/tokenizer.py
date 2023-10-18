import nltk
from nltk.tokenize import word_tokenize

#nltk.download('punkt')

with open("Transformer/pairs.tsv",encoding="utf-8") as t:
    text = t.read()
#text = text.split("\t\t")

print(text[:100])