# import all the libraries
import lxml.etree as et
from XmlTagNames import TagName, PubmedArticle
from FileOperations import FileOperations
from tqdm import tqdm


# Define a class to pre-process the xml file
class Preprocessing:

    def __init__(self):
        pass

    def __getmeshwords(self, meshheadinglist):
        meshwords = [meshword.strip() for meshheding in meshheadinglist for meshnode in meshheding for meshword in meshnode.text.split(',')]
        return meshwords

    def __getarticledata(self, article):
        title, abstract, language = [None] * 3
        for node in article:
            if node.tag == TagName.ArticleTitle:
                title = node.text
            elif node.tag == TagName.Abstract:
                abstract = ''
                for abstextnode in node:

                    text = abstextnode.text
                    if text is not None:
                        abstract += ' ' + text.strip()

                    for innernodes in abstextnode:
                        text = innernodes.text
                        if text is not None:
                            abstract += ' ' + text.strip()
                        tail = innernodes.tail
                        if tail is not None:
                            abstract += ' ' + tail.strip()

                    tail = abstextnode.tail
                    if tail is not None:
                        abstract += ' ' + tail.strip()

                # try:
                #     abstract = ''.join([abstextnode.text for abstextnode in list(node)])
                # except:
                #     abstract = None
            elif node.tag == TagName.Language:
                language = node.text

        #print(abstract)
        return title, abstract, language

    # function to process medlinecitations
    def __processmedlinecitation(self, medlinecitations):
        pmid, dc, dr, title, abstracttext, language, meshwords = [None] * 7
        # print(list(medlinecitations))

        for node in medlinecitations:
            if node.tag == TagName.PMID:
                pmid = node.text
                # print(pmid)
                # if pmid in ['29747621', '29614582']:
                #     c = 20
                #     pass
                pass
            elif node.tag == TagName.DateCompleted:
                dc = '-'.join([n.text for n in list(node)])
                pass
            elif node.tag == TagName.DateRevised:
                dr = '-'.join([n.text for n in list(node)])
                pass
            elif node.tag == TagName.Article:
                title, abstracttext, language = self.__getarticledata(node)
                pass
            elif node.tag == TagName.MedlineJournalInfo:
                pass
            elif node.tag == TagName.ChemicalList:
                pass
            elif node.tag == TagName.CitationSubset:
                pass
            elif node.tag == TagName.MeshHeadingList:
                meshwords = self.__getmeshwords(node)
                pass

        return pmid, dc, dr, title, abstracttext, language, meshwords

        pass

    # function to process pubmeddata
    def __processpubmeddata(self, pubmeddata):
        # Do Nothing
        pass

    # Process the individual PubmedArticle
    def __processpubmedarticle(self, pubmedarticle):
        medlinecitations, pubmeddata = [None] * 2
        try:
            medlinecitations, pubmeddata = list(pubmedarticle)
            # print(medlinecitations, '\n', pubmeddata)
        except:
            print('Unpack Error')

        # Process MedlineCitation
        pmid, dc, dr, title, abstracttext, language, meshwords = self.__processmedlinecitation(medlinecitations)

        #TODO Currently not using the fields in PubmedData
        # Process pubmeddata
        self.__processpubmeddata(pubmeddata)

        return pmid, dc, dr, title, abstracttext, language, meshwords

    # Parse the xml file
    def parse(self, filename):
        pubmedarticlelist = []
        unsavedpmids = []

        #Create an object of TagName Class
        tagname = TagName()

        # tree = ET.parse(filename)
        context = et.iterparse(filename, events=('end',), tag=tagname.PubmedArticleSet)
        for event, PubmedArticleSet in context:
            # print(event, '\t', PubmedArticleSet)
            for index, pubmedarticle in tqdm(enumerate(PubmedArticleSet)):
                pmid, dc, dr, title, abstracttext, language, meshwords = self.__processpubmedarticle(pubmedarticle)
                pa = PubmedArticle(pmid, dc, dr, title, abstracttext, language, meshwords)

                if language in ['eng', None] and abstracttext not in ['', None]:
                    pubmedarticlelist.append(pa)
                else:
                    unsavedpmids.append(pmid)
                    print(pmid, ' is not saved. ', 'Language = ', language)
                    # print('\t', language)
                    # if language == 'eng': print('Abstract = ', abstracttext)
                # print(len(pubmedarticlelist))

        return pubmedarticlelist, unsavedpmids


    def LoadFile(self, filename):
        # Create FileOperations object
        fo = FileOperations()
        # Load the file
        articlelists = fo.LoadFile(filename)

        # Create an optional class obj and dict
        pubmedarticlelists = {
            article.PMID: PubmedArticle(article.PMID, article.DateCompleted, article.DateRevised, article.ArticleTitle,
                                        article.Abstract, article.Language, article.MeshWords) for i, article in
            enumerate(articlelists)}

        del fo
        return pubmedarticlelists
