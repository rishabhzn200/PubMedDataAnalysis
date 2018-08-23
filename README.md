# PubMedDataAnalysis

This Project was developed to:

1. Extract the articles from the Pubmed website.
2. Build Doc2Vec, Latent Semantic Indexing and TFIDF.
3. Generate the similarity matrix for each of these models.
4. Similarity matrices are then combined and are used to find the similar articles.

Libraries Used: <br />
<em>1. Requests and BeautifulSoup : </em>To Scrape the data from "https://www.ncbi.nlm.nih.gov/pubmed/".<br />
<em>2. lxml and regex : </em>To Preprocess XML data.<br/>
<em>3. NLTK :</em> (Tokenization, Stemming, Lemmatization).<br/> 
<em>4. Gensim :</em> For models (Doc2Vec, TFIDF and LSA).<br/>
