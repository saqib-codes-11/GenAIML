from rake_nltk import Rake

import nltk
nltk.download('stopwords')

r = Rake()

# Extraction given the text.
sentence = "The court case between amber and johnny is coming to a close."
r.extract_keywords_from_text(sentence)

keywords = r.degree
kwAsString = ""

for key in keywords.items():
    kwAsString = kwAsString + " " + str(key[0])

print(kwAsString)
