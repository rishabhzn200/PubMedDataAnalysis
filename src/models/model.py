import gensim
from tqdm import tqdm
import multiprocessing
import constants as GV
from file_utils import FileHandler


class Doc2VecModel:
    def __init__(self, seed=1, num_features=20, min_word_count=2, context_size=6):
        # Initialize Model
        self.doc2vec_model = None

        # Initialize the parameters for Doc2Vec Model
        self.seed = seed

        # Dimensionality of the resulting word vectors
        self.num_features = num_features

        # Minimum word count threshold
        self.min_word_count = min_word_count

        # Number of threads to run in parallel
        self.num_workers = multiprocessing.cpu_count()

        # Context window length
        self.context_size = context_size

        # Create the output model file path
        self.outputmodelfilename = (
            GV.basedirectoy
            + "pubmedd2vmodel_F_"
            + str(self.num_features)
            + "_C_"
            + str(self.context_size)
            + ".model"
        )

        pass

    def create_tagged_documents(self, tokenizeddocs, ids):
        taggeddocuments = None
        fo = FileHandler()

        if fo.exists(GV.taggedDocumentFile):
            taggeddocuments = fo.load_file(GV.taggedDocumentFile)
        else:
            taggeddocuments = [
                gensim.models.doc2vec.TaggedDocument(s, [ids[i]])
                for i, s in tqdm(enumerate(tokenizeddocs))
            ]
            fo.save_file(GV.taggedDocumentFile, taggeddocuments, mode="wb")

        del fo

        return taggeddocuments

    def get_model(self, docs, ids):
        # Link: https://www.programcreek.com/python/example/103013/gensim.models.doc2vec.TaggedDocument
        # Link: https://rare-technologies.com/doc2vec-tutorial/

        self.doc2vec_model = gensim.models.doc2vec.Doc2Vec(
            seed=self.seed,
            workers=self.num_workers,
            size=self.num_features,
            min_count=self.min_word_count,
            window=self.context_size,
        )

        # Build the vocab
        self.doc2vec_model.build_vocab(docs)

        # Start training the model
        self.doc2vec_model.train(
            docs,
            total_examples=self.doc2vec_model.corpus_count,
            epochs=self.doc2vec_model.iter,
        )
        print("Training finished")

        # Save the model
        self.doc2vec_model.save(self.outputmodelfilename)
        print("Model saved")
        return self.doc2vec_model

    def get_model_fileName(self):
        return self.outputmodelfilename

    def load_model(self, filename=None):
        if filename is None:
            self.doc2vec_model = gensim.models.doc2vec.Doc2Vec.load(
                self.outputmodelfilename
            )
        else:
            self.doc2vec_model = gensim.models.doc2vec.Doc2Vec.load(filename)

        return self.doc2vec_model

    def save_similar_documents(self, pubmedarticlelists, similardocfilename):
        pdocs = self.doc2vec_model.docvecs.doctag_syn0  # [:pts]

        # Get all the pmids
        pmids = self.doc2vec_model.docvecs.offset2doctag  # [:pts]

        # Create the similar documents dictionary for each pmid
        similardocdict = {}
        import pickle

        for idx, pmid in tqdm(enumerate(pmids)):
            # output the top 20 similair documents
            similardocdict[pmid] = self.doc2vec_model.docvecs.most_similar(
                pmid, topn=23752
            )
            similardocdict[pmid].insert(0, (pmid, "1.0"))

            # TODO New code
            if idx % 1000 == 0 or idx == 23753:
                with open(
                    "./saveddata/simdictpmid.pkl", mode="a+b"
                ) as f:  # appending, not writing
                    pickle.dump(similardocdict, f)

                similardocdict = {}

            # TODO

        # { 'pmid1': {'Title':'Title', {Similar:[[id, 'title', score], [id, 'title', score], [id, 'title', score]]},
        #   'pmid2': {'Title':'Title', {Similar:[[id, 'title', score], [id, 'title', score], [id, 'title', score]]},
        #   ...
        # }

        similararticlesdict = {}
        for idx, pmid in tqdm(enumerate(pmids)):
            # Find current pmid title
            doctitle = pubmedarticlelists[pmid].ArticleTitle

            # Find similar documents pmids
            similardocpmids = similardocdict[pmid]

            similartitlescorelist = []

            # Iterate through all the pmids
            for id, score in similardocpmids:
                articletitle = pubmedarticlelists[id].ArticleTitle
                similartitlescorelist.append([id, articletitle, score])

            similararticlesdict[pmid] = {
                "Title": doctitle,
                "Similar": similartitlescorelist,
            }

        # Save the similar documents
        fo = FileHandler()
        fo.save_file(similardocfilename, similararticlesdict)
        # fo.SaveFile('./savedPickle/similardocuments.txt', similararticlesdict, mode='w', isobj=False)

    def visualize(self, pmid_choice):
        from sklearn.manifold import TSNE

        pdocs = self.doc2vec_model.docvecs.doctag_syn0
        pmids = self.doc2vec_model.docvecs.offset2doctag
        # pmid_choice = pmids[0]
        sims = self.doc2vec_model.docvecs.most_similar(pmid_choice, topn=20)
        print(sims)

        sim_indexes = []
        for id, dist in sims:
            sim_indexes.append(pmids.index(id))
            # if id not in pmids:
            #     print(id,' not found')

        print(max(sim_indexes))
        mainidindex = pmids.index(pmid_choice)

        tsne = TSNE(n_components=2, random_state=0)
        X_2d = tsne.fit_transform(pdocs)

        x = [p[0] for p in X_2d]
        y = [p[1] for p in X_2d]

        # Main Point
        x_main = [x[mainidindex]]
        y_main = [y[mainidindex]]

        # Similar Points
        x_sim = []
        y_sim = []

        for idx in sim_indexes:
            x_sim.append(x[idx])
            y_sim.append(y[idx])

        from matplotlib import pyplot as plt

        # colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
        plt.scatter(x, y, c="y", edgecolor="black", label="AllPoints")

        plt.scatter(x_sim, y_sim, c="g", edgecolor="black", label="Similar")

        plt.scatter(x_main, y_main, c="b", edgecolor="black", label="MainPoint")

        plt.legend()
        plt.savefig(
            "newtempfile_"
            + str(self.num_features)
            + "_"
            + str(self.context_size)
            + "_"
            + "_new.png"
        )
        plt.show()

        pass
