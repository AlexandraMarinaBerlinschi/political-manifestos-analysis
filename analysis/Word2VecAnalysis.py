from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
#nltk.download('punkt_tab') #de rulat doar o data
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from typing import List
from Utils.Enums import RomanianParty
from Utils.UtilityFunctions import preprocess_text

def analyze_document_similarity(doc_vectors: np.ndarray, party_names: List[str]) -> pd.DataFrame:
    """
    Calculeaza similiratitatea cosinus intre vectorii documentelor si returneaza matricea
    """
    similarity_matrix = np.zeros((len(doc_vectors), len(doc_vectors)))

    for i in range(len(doc_vectors)):
        for j in range(len(doc_vectors)):
            similarity = np.dot(doc_vectors[i], doc_vectors[j]) / ( #calculul similiaritatii cosinus (te rog fii formula buna)
                np.linalg.norm(doc_vectors[i]) * np.linalg.norm(doc_vectors[j]) + 1e-9
            )
            similarity_matrix[i][j] = similarity

    return pd.DataFrame(similarity_matrix, columns=party_names, index=party_names)

def create_doc2vec_model(manifesto_corpus: List[str], vector_size=100):
    """
    face embeddings pe corpus
    """
    # documentele sunt deja preprocesate, le tokenizez
    tagged_documents = [
        TaggedDocument(words=word_tokenize(preprocess_text(doc)), tags=[str(i)])
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

    # preluarea vectorilor
    doc_vectors = np.array([
        model.infer_vector(word_tokenize(preprocess_text(doc)))
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
    documents = [word_tokenize(preprocess_text(doc)) for doc in manifesto_corpus] #tokenizarea

    # calcul frecventa globala pentru fiecarui cuvant
    global_freq = {}
    for doc in documents:
        for word in set(doc):
            global_freq[word] = global_freq.get(word, 0) + 1

    print("\nMost distinctive words for each party (Doc2Vec analysis):")
    for party_idx, (doc, party_name) in enumerate(zip(documents, party_names)):
        print(f"\n{party_name}:")

        doc_vector = model.infer_vector(doc) #vectorul documentului

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

        #cele mai distinctive cuvinte
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
    if word not in model.wv:
        print(f"\nWord '{word}' not found in vocabulary.")
        return

    print(f"\nSearching for '{word}' and semantically similar words in manifestos:")

    #se ia vectorul tinta si cuvintele similare cu scor de similaritate
    word_vector = model.wv[word]
    similar_words = [(word, 1.0)] + model.wv.most_similar(word, topn=3)

    for i, manifesto in enumerate(manifesto_data):
        doc_vector = model.infer_vector(word_tokenize(preprocess_text(manifesto_corpus[i])))

        # calcul similiaritate semantica cu documentul
        doc_similarity = np.dot(word_vector, doc_vector) / (
            np.linalg.norm(word_vector) * np.linalg.norm(doc_vector) + 1e-9
        )

        # contorizeaza aparitiile cuvantului tinta si a celorlalte cuvinte similare, in document
        words = word_tokenize(preprocess_text(manifesto_corpus[i]))
        matches = []

        for similar_word, sim_score in similar_words:
            count = words.count(similar_word)
            if count > 0:
                matches.append(f"'{similar_word}' (similarity: {sim_score:.2f}): {count} times")

        party_name = RomanianParty.get_name(manifesto.metadata.party_id)
        print(f"\n{party_name}'s manifesto (document similarity: {doc_similarity:.4f}):")
        if matches:
            for match in matches:
                print(f"  {match}")
