import nltk
from farasa.pos import FarasaPOSTagger
from farasa.segmenter import FarasaSegmenter
from farasa.ner import FarasaNamedEntityRecognizer

def importingData():
    cleanSens = []
    labels = []
    test_sentences = []
    with open("PreProccessing/sentences.txt", 'r', encoding='utf-8') as file:
        for line in file:
            # Process each line as needed
            cleanSens.append(line.strip())
    with open("PreProccessing/labels.txt", 'r', encoding='utf-8') as file:
        for line in file:
            # Process each line as needed
            labels.append(int(line.strip()))


    with open("PreProccessing/test_sentences.txt", 'r', encoding='utf-8') as file:
        for line in file:
            # Process each line as needed
            test_sentences.append(line.strip())
    return cleanSens , labels  , test_sentences

nltk.download('punkt')

# Example Arabic sentence
arabic_sentence , labels ,test_sentences = importingData()
# POS tagging using Farasa
farasa_segmenter = FarasaSegmenter(interactive=True)
farasa_pos_tagger = FarasaPOSTagger(interactive=True)

# Tokenize and POS tagging
tokens = farasa_segmenter.segment(arabic_sentence)
pos_tags = farasa_pos_tagger.tag(tokens)

print("Original Sentence:", arabic_sentence)
print("Tokens:", tokens)
print("POS Tags:", pos_tags)

# Spell checking using Farasa
from farasa.stemmer import FarasaStemmer

farasa_stemmer = FarasaStemmer(interactive=True)
corrected_tokens = farasa_stemmer.stem(tokens)

corrected_sentence = ' '.join(corrected_tokens)
print("Corrected Sentence:", corrected_sentence)
