# Defines all the tag names in the XML file


class TagName:

    PubmedArticleSet = 'PubmedArticleSet'
    PubmedArticle = 'PubmedArticle'
    PMID = 'PMID'
    DateCompleted = 'DateCompleted'
    DateRevised = 'DateRevised'
    Article = 'Article'
    MedlineJournalInfo = 'MedlineJournalInfo'
    ChemicalList = 'ChemicalList'
    CitationSubset = 'CitationSubset'
    MeshHeadingList = 'MeshHeadingList'
    DescriptorName = 'DescriptorName'
    QualifierName = 'QualifierName'
    ArticleTitle = 'ArticleTitle'
    Abstract = 'Abstract'
    Language = 'Language'


class PubmedArticle:
    def __init__(self, pmid, datecompleted, daterevised, title, abstracttext, language, meshwords):
        self.PMID = pmid
        self.DateCompleted = datecompleted
        self.DateRevised = daterevised
        self.ArticleTitle = title
        self.Abstract = abstracttext
        self.AuthorList = None
        self.Language = language
        self.MedlineJournalInfo = None
        self.ChemicalList = None
        self.MeshWords = meshwords

        # Not processing yet
        self.PubmedData = None

    def getdictionary(self):
        return self.__dict__
