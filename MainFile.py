import re
from string import ascii_lowercase
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import MWETokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from ScrapeData import Scraper
from FileOperations import FileOperations
import GlobalVariables as GV
from DataPreprocessing import Preprocessing
from Doc2VecModel import Doc2VecModel


def GetGlossaryTerms():
    # Create a variable which holds the URL to be scraped
    baseurl = 'https://www.ncbi.nlm.nih.gov/pubmedhealth/topics/health/'

    # Define the pattern to be searched for in the glossary terms
    pattern1 = re.compile(r'(.*) \(see (.*)\)')
    pattern2 = re.compile(r'(.*) \((.*)\)')

    # Create an object of Scraper class
    scraper = Scraper()

    # Create a variable to hold the list of glossary terms and list of synonymous terms
    glossarylist = []
    synonymlist = []

    # Loop through all the characters in lowercase ascii(a-z)
    for char in tqdm(ascii_lowercase):
        url = baseurl + char + '/'

        # Get the html page
        soup = scraper.getlink(url)

        # Find all the list items - Glosarry terms
        resultlisttags = scraper.findall(soup, "ul", **{'class':'resultList'})
        # resultlisttag = scraper.find(soup, "ul", class_='resultList')

        for resultlisttag in resultlisttags:
            glossarylistaglist = scraper.findall(resultlisttag, "li")

            # Loop to iterate over the list of all the terms found in the page
            for glossarylistag in glossarylistaglist:
                litext = glossarylistag.text.strip()
                glossaryterm = scraper.find(glossarylistag, "a").text.strip()

                if litext != glossaryterm or '(' in litext:
                    # Processing required
                    # Get rid of colon
                    litext = litext.split( ':' )[0]

                    # Find the pattern1
                    matches = re.match(pattern1, litext)

                    # If match is not found then check for pattern2
                    if matches is None:
                        matches = re.match(pattern2, litext)

                    # Get the two terms extracted by the pattern
                    term1, term2 = matches[1], matches[2]

                    # save in synonym list
                    synonymlist.append((term1.lower().split(), term2.lower().split()))

                    # Append the glossary term to the list
                    glossarylist.append(term1)

                else:
                    # Append the glossary term to the list
                    glossarylist.append(glossaryterm.split( ':' )[0])

    # Additional step to save the glossary in the format which can be used for tokenization easily
    glossarylist = (set(glossarylist))
    glossarylist = [(gloss.lower().split()) for gloss in glossarylist]

    # Return the glossary list and the synonyms list
    return glossarylist, synonymlist

def InitializeGlossary():

    # Create FileOperation object
    fo = FileOperations()

    # Initialize the two list to None
    glossarylist, synonymlist = [None]*2

    if fo.exists(GV.healthGlossaryFilePath):
        # Load the file from disk
        glossarylist, synonymlist = fo.LoadFile(GV.healthGlossaryFilePath) , fo.LoadFile(GV.synonymsFilePath)

    else:
        # Get all the glossary terms
        glossarylist, synonymlist = GetGlossaryTerms()

        # Save the glossary terms

        fo.SaveFile(GV.healthGlossaryFilePath, glossarylist, mode='wb')

        # Save the synonyms
        fo.SaveFile(GV.synonymsFilePath, synonymlist, mode='wb')

    del fo

    return glossarylist, synonymlist


def SaveGlossary(glossarylist, synonymlist):
    fo = FileOperations()

    if fo.exists(GV.glossaryFilePath):
        return
    else:
        glossarylist, synonymlist = fo.LoadFile(GV.healthGlossaryFilePath), fo.LoadFile(GV.synonymsFilePath)
        synonymterm2 = set(tuple(term2) for term1, term2 in synonymlist)
        synonymterm2 = list((list(term) for term in synonymterm2))
        glossarylist += list(synonymterm2)
        fo.SaveFile(GV.glossaryFilePath, glossarylist, mode='wb')
    del fo


def PreprocessData():
    # Create an object initialized to None
    pubmedarticlelists = None

    # Create FileOperations object
    fo = FileOperations()

    # parse the xml file
    p = Preprocessing()

    # If parsed file is present then load the file else parse the file
    if fo.exists(GV.parsedDataFile):
        pubmedarticlelists = p.LoadFile(GV.parsedDataFile)

    else:
        # Call the Parse method
        pubmedarticlelists, unsavedpmids = p.parse(GV.inputXmlFile)

        print(len(pubmedarticlelists))
        print(len(unsavedpmids))

        # Save the parsed data to a file
        fo.SaveFile(GV.parsedDataFile, pubmedarticlelists, mode='wb')
        fo.SaveFile(GV.unsavedPmidFile, unsavedpmids, mode='w')

        pubmedarticlelists = p.LoadFile(GV.parsedDataFile)

    del fo

    return pubmedarticlelists


def GetDocuments(articlelists, title=True, abstract=True, meshwords=False):
    p = Preprocessing()
    docs = p.GetDocuments(articlelists, title=title, abstract=abstract, meshwords=meshwords)
    del p
    return docs


def TokenizeDocs(docs, glossarylist, filename=GV.tokenizedDocumentD2VFile):
    tokenizeddocs = []
    combineddocuments = []
    fo = FileOperations()
    # tokenizer = RegexpTokenizer(r'\w+')
    if fo.exists(filename):
        # Load the file
        combineddocuments = fo.LoadFile(filename)
        pass

    else:
        tokenizer = MWETokenizer(glossarylist)
        regtokenizer = RegexpTokenizer(r'\w+')
        for doc in tqdm(docs):
            sentences = sent_tokenize(doc)

            tmp = []
            for sentence in sentences:
                tokens = tokenizer.tokenize(regtokenizer.tokenize(sentence.lower()))
                token_lowercase = [x.lower() for x in tokens]
                tmp.append(token_lowercase)
            tokenizeddocs.append(tmp)

        for doc in tqdm(tokenizeddocs):
            tokdoc = []
            [tokdoc.extend(sent) for sent in doc]
            combineddocuments.append(tokdoc)

        # Save the file
        fo.SaveFile(filename, combineddocuments, mode='wb')

    del fo

    return combineddocuments


def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def TokenizeDocsNew(docs, glossarylist, filename=GV.tokenizedDocumentD2VFile):
    tokenizeddocs = []
    combineddocuments = []
    fo = FileOperations()
    # tokenizer = RegexpTokenizer(r'\w+')
    if fo.exists(filename):
        # Load the file
        combineddocuments = fo.LoadFile(filename)
        pass

    else:
        tokenizer = MWETokenizer(glossarylist)
        regtokenizer = RegexpTokenizer(r'\w+')
        lmtzr = WordNetLemmatizer()
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        stop_words = stopwords.words('english')
        for doc in tqdm(docs):
            sentences = sent_tokenize(doc)

            tmp = []
            for sentence in sentences:
                # For each sentence in the sentences

                # Tokenize the sentence based on Regex and then using MWETokenizer
                tokens = tokenizer.tokenize(regtokenizer.tokenize(sentence.lower()))

                # Lower the case of all the tokens
                token_lowercase = [x.lower() for x in tokens]

                # Lemmatize the sentence. Find the POS tags and then lemmatize
                tokens_lowecase_tagged = nltk.pos_tag(token_lowercase)
                lammetized_sentence = [lmtzr.lemmatize(wrd, pos=get_wordnet_pos(tag)) for wrd, tag in tokens_lowecase_tagged]

                # Stem the sentence
                stemmed_sentence = [stemmer.stem(wrd) for wrd in lammetized_sentence]

                # Remove the stop words
                processed_sentence = [word for word in stemmed_sentence if word not in stop_words]

                tmp.append(processed_sentence)
            tokenizeddocs.append(tmp)

        for doc in tqdm(tokenizeddocs):
            tokdoc = []
            [tokdoc.extend(sent) for sent in doc]
            combineddocuments.append(tokdoc)

        # Save the file
        fo.SaveFile(filename, combineddocuments, mode='wb')

    del fo

    return combineddocuments


def Doc2Vec(docs, ids, glossarylist, pubmedarticlelists):

    # Tokenize all the docs
    tokenizeddocs = TokenizeDocs(docs, glossarylist, GV.tokenizedDocumentD2VFile)

    # Create Doc2Vec Model. Changing parameters will change the model name
    doc2vecmodel = Doc2VecModel(seed=1, num_features = 200, min_word_count = 2, context_size = 3)
    taggeddocuments = doc2vecmodel.CreateTaggedDocuments(tokenizeddocs, ids)
    model = doc2vecmodel.Model(taggeddocuments, ids)

    # Get model filename
    modelfile = doc2vecmodel.GetModelFileName()

    #Load the model
    model = doc2vecmodel.LoadModel(modelfile)

    # Save Similar Documents
    doc2vecmodel.SaveSimilarDocuments(pubmedarticlelists, GV.similarDocumentListFile)

    #Play
    similardocdict = FileOperations().LoadFile(GV.similarDocumentListFile)
    print(similardocdict['29794785']['Title'])
    print('---------------------------------------')
    for id, title, score in similardocdict['29794785']['Similar']:
        print(id, ' : ', title)

    doc2vecmodel.Visualize('29794785')


def tfidf(docs, ids, glossarylist):

    a = 20
    import gensim
    import numpy as np
    tokenizeddocs = TokenizeDocsNew(docs, glossarylist, filename=GV.tokenizedDocumentTfidfFile)


    # from nltk.tokenize import word_tokenize
    # tokenizeddocs_nltk = [[w.lower() for w in word_tokenize(text)] for text in tqdm(docs)]

    dictionary = gensim.corpora.Dictionary(tokenizeddocs)

    corpus = [dictionary.doc2bow(doc) for doc in tqdm(tokenizeddocs)]

    tf_idf = gensim.models.TfidfModel(corpus)
    print(tf_idf)
    corpus_tfidf = tf_idf[corpus]

    # similarity = gensim.similarities.Similarity('./tfidf/', tf_idf[corpus], num_features=len(dictionary))
    # print(similarity)

    index = gensim.similarities.MatrixSimilarity(tf_idf[corpus])
    print("We compute similarities from the TF-IDF corpus : %s" % type(index))
    index.save(GV.similarityIndexFile)

    # Load the saved index
    index = gensim.similarities.MatrixSimilarity.load(GV.similarityIndexFile)
    # sims = index[corpus_tfidf]

    similarityMatrix = np.array([index[corpus_tfidf[i]]
                                 for i in tqdm(range(len(corpus)))])

    np.save(GV.similarityMatrixTFfidf, similarityMatrix)  # .npy extension is added if not given
    similarityMatrix = np.load(GV.similarityMatrixTFfidf)

    # Check the model
    sims = index[corpus_tfidf[0]]

    idx = (-sims).argsort()[:20]

    vals = sims[idx]

    npids = np.array(ids)
    npdocs = np.array(docs)

    simids = npids[idx]
    simdocs = npdocs[idx]

    for i in range(len(idx)):
        print(simids[i], ' : ', simdocs[i])

    a = 20


def LSIModel(docs, ids, glossarylist):
    # from LSIModel import LSIModel
    from nltk.corpus import stopwords
    import gensim
    import numpy as np
    from gensim.models import LsiModel
    from gensim.similarities import MatrixSimilarity

    tokenizeddocs = TokenizeDocsNew(docs, glossarylist, filename=GV.tokenizedDocumentLSIFile)
    stop_words = stopwords.words('english')
    tokenizeddocsf = [word for word in tqdm(tokenizeddocs) if word not in stop_words]

    dictionary = gensim.corpora.Dictionary(tokenizeddocsf)
    corpus = [dictionary.doc2bow(doc) for doc in tqdm(tokenizeddocsf)]
    tf_idf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tf_idf[corpus]
    lsi = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=500)
    lsi_index = MatrixSimilarity(lsi[corpus_tfidf])
    similarityMatrix = np.array([lsi_index[lsi[corpus_tfidf[i]]]
                                    for i in tqdm(range(len(corpus)))])

    np.save(GV.similarityMatrixLSI, similarityMatrix)  # .npy extension is added if not given
    similarityMatrix = np.load(GV.similarityMatrixLSI)

    # Save the id list
    pmidlist = np.array(ids)
    np.save(GV.similarityMatrixLSI_ID, pmidlist)

    # Query
    queryIndex = 0
    pmid = pmidlist[0]
    sims = similarityMatrix[0]

    idx = (-sims).argsort()[:20]

    vals = sims[idx]

    npids = np.array(ids)
    npdocs = np.array(docs)

    simids = npids[idx]
    simdocs = npdocs[idx]

    for i in range(len(idx)):
        print(simids[i], ' : ', simdocs[i])

    a = 20


def main():

    # Initialize the Glossary
    glossarylist, synonymlist = InitializeGlossary()

    # Save the glossary
    SaveGlossary(glossarylist, synonymlist)

    # Preprocess the data in the xml file
    pubmedarticlelists = PreprocessData()

    # Get the documents list: Title, Abstract and Mesh words or any combination of these
    docs, ids = GetDocuments(pubmedarticlelists, title=True, abstract=True, meshwords=False)

    # Use docs and their ids

    # Create Model: Doc2Vec and Tf-Idf

    # Doc2Vec model
    Doc2Vec(docs, ids, glossarylist, pubmedarticlelists)

    # Tf-Idf Model
    # tfidf(docs, ids, glossarylist)

    # LSI Model
    # LSIModel(docs, ids, glossarylist)

if __name__ =='__main__':
    main()
