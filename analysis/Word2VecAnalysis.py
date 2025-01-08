from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
'''de rulat doar o data'''
#nltk.download('punkt_tab')
#nltk.download('averaged_perceptron_tagger_eng')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import numpy as np
import pandas as pd
from typing import List
from Utils.Enums import RomanianParty
from Utils.UtilityFunctions import preprocess_text


def get_wordnet_pos(word):
    """
    mapeaza etichetele generate de nltk pentru lematizare
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def analyze_document_similarity(doc_vectors: np.ndarray, party_names: List[str]) -> pd.DataFrame:
    """
    Calculeaza similiratitatea cosinus intre vectorii documentelor si returneaza matricea.
    Similaritatea este calculata folosind produsul scalar normalizat al vectorilor
    """
    num_docs = len(doc_vectors)
    similarity_matrix = np.zeros((num_docs, num_docs))

    for i in range(num_docs):
        for j in range(num_docs):
            # Calculam norma vectorilor
            norm_i = np.linalg.norm(doc_vectors[i])
            norm_j = np.linalg.norm(doc_vectors[j])

            # Calculam similaritatea cosinus direct
            if norm_i > 0 and norm_j > 0:  # evitam impartirea la 0
                similarity = np.dot(doc_vectors[i], doc_vectors[j]) / (norm_i * norm_j)
                # Folosim abs() pentru a trata potentialele erori numerice mici
                # care ar putea duce la valori foarte apropiate de 0 dar negative
                similarity_matrix[i][j] = abs(similarity)
            else:
                similarity_matrix[i][j] = 0.0

    return pd.DataFrame(similarity_matrix, columns=party_names, index=party_names)
def create_doc2vec_model(manifesto_corpus: List[str], vector_size=100):
    """
    face embeddings pe corpus
    """
    lemmatizer = WordNetLemmatizer()

    # documentele sunt deja preprocesate, le tokenizez si lematizez mai bine
    tagged_documents = [
        TaggedDocument(
            words=[lemmatizer.lemmatize(word, get_wordnet_pos(word))
                   for word in word_tokenize(preprocess_text(doc))],
            tags=[str(i)]
        )
        for i, doc in enumerate(manifesto_corpus)
    ]

    # Initialize Doc2Vec model with improved parameters
    model = Doc2Vec(
        vector_size=vector_size,
        window=10,  # cate cuvinte din jurul tintei vor fi luate in considerare
        min_count=5,  # trebuie sa apara cuvantul de min 5 ori ca sa fie luat in considerare
        workers=4,
        epochs=200,  # si cu 100 e acelasi rezultat
        dm=1,  # modelul pv-dm
        dbow_words=1,  # antreneaza simultan vectorii
        negative=5  # se folosesc sampling uri negative pt a imbunatati eficienta
    )

    # construirea si antrenarea modelului
    model.build_vocab(tagged_documents)
    model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.epochs)

    # preluarea vectorilor cu lemmatizare imbunatatita
    lemmatizer = WordNetLemmatizer()
    doc_vectors = np.array([
        model.infer_vector([lemmatizer.lemmatize(word, get_wordnet_pos(word))
                            for word in word_tokenize(preprocess_text(doc))])
        for doc in manifesto_corpus
    ])

    return model, doc_vectors


def find_party_specific_vocabulary_doc2vec(model: Doc2Vec,
                                           manifesto_corpus: List[str],
                                           party_names: List[str],
                                           top_n: int = 10):
    """
    Gaseste cuvintele distinctive pentru fiecare partid
    """
    lemmatizer = WordNetLemmatizer()
    documents = [[lemmatizer.lemmatize(word, get_wordnet_pos(word))
                  for word in word_tokenize(preprocess_text(doc))]
                 for doc in manifesto_corpus]  # tokenizarea si lemmatizarea

    # calcul frecventa globala pentru fiecarui cuvant
    global_freq = {}
    for doc in documents:
        for word in set(doc):
            global_freq[word] = global_freq.get(word, 0) + 1

    print("\nMost distinctive words for each party (Doc2Vec analysis):")
    for party_idx, (doc, party_name) in enumerate(zip(documents, party_names)):
        print(f"\n{party_name}:")

        doc_vector = model.infer_vector(doc)  # vectorul documentului

        # Calcul scor cuvant (similaritate+specificitate)
        word_scores = {}
        doc_words = set(doc)

        for word in doc_words:
            if word in model.wv:
                # calcul similaritate semantica
                word_vector = model.wv[word]
                similarity = np.dot(doc_vector, word_vector) / (
                        np.linalg.norm(doc_vector) * np.linalg.norm(word_vector) + 1e-9
                )

                # calcul specificitate (kinda tf-idf)
                doc_freq = global_freq.get(word, 0)
                specificity = 1.0 / (1.0 + np.log1p(doc_freq))

                word_scores[word] = similarity * specificity

        # cele mai distinctive cuvinte
        top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        for word, score in top_words:
            print(f"  {word}: {score:.4f}")
            similar_words = model.wv.most_similar(word, topn=3)
            print(f"    Similar words: {', '.join(word for word, _ in similar_words)}")


def find_word_in_manifestos_doc2vec(word: str,
                                    model: Doc2Vec,
                                    manifesto_data: List[any],
                                    manifesto_corpus: List[str]):
    """
    Cautare semantica a cuvantului si a cuvintelor semnificativ similare
    """
    lemmatizer = WordNetLemmatizer()
    word = lemmatizer.lemmatize(word, get_wordnet_pos(word))

    if word not in model.wv:
        print(f"\nWord '{word}' not found in vocabulary.")
        return

    print(f"\nSearching for '{word}' and semantically similar words in manifestos:")

    # se ia vectorul tinta si cuvintele similare cu scor de similaritate
    word_vector = model.wv[word]
    similar_words = [(word, 1.0)] + model.wv.most_similar(word, topn=3)

    for i, manifesto in enumerate(manifesto_data):
        processed_words = [lemmatizer.lemmatize(w, get_wordnet_pos(w))
                           for w in word_tokenize(preprocess_text(manifesto_corpus[i]))]
        doc_vector = model.infer_vector(processed_words)

        # calcul similiaritate semantica cu documentul
        doc_similarity = np.dot(word_vector, doc_vector) / (
                np.linalg.norm(word_vector) * np.linalg.norm(doc_vector) + 1e-9
        )

        # contorizeaza aparitiile cuvantului tinta si a celorlalte cuvinte similare, in document
        matches = []

        for similar_word, sim_score in similar_words:
            count = processed_words.count(similar_word)
            if count > 0:
                matches.append(f"'{similar_word}' (similarity: {sim_score:.2f}): {count} times")

        party_name = RomanianParty.get_name(manifesto.metadata.party_id)
        print(f"\n{party_name}'s manifesto (document similarity: {doc_similarity:.4f}):")
        if matches:
            for match in matches:
                print(f"  {match}")