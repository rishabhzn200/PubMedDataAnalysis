# PubMedDataAnalysis

This Project was developed to:

1. Extract the articles from the Pubmed website.
2. Compute the Similarity matrix using Doc2Vec model, Latent Semantic Indexing and TFIDF.
3. Combine the similarity scores generated above to find the similar articles.

Libraries Used: <br />
<em>1. Requests and BeautifulSoup : </em>To Scrape the data from "https://www.ncbi.nlm.nih.gov/pubmed/".<br />
<em>2. lxml and regex : </em>To Preprocess XML data.<br/>
<em>3. NLTK :</em> (Tokenization, Stemming, Lemmatization).<br/> 
<em>4. Gensim :</em> For Doc2Vec, TFIDF and LSA.<br/>
