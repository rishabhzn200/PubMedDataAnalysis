import re
from string import ascii_lowercase
from tqdm import tqdm
from ScrapeData import Scraper
from FileOperations import FileOperations
import GlobalVariables as GV
from DataPreprocessing import Preprocessing


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
        resultlisttag = scraper.find(soup, "ul", **{'class':'resultList'})
        # resultlisttag = scraper.find(soup, "ul", class_='resultList')
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

    # Additional step to save the glossary in the format which can be used for tokenization eaily
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

    del fo

    return pubmedarticlelists

def main():

    # Initialize the Glossary
    glossarylist, synonymlist = InitializeGlossary()

    # Save the glossary
    SaveGlossary(glossarylist, synonymlist)

    # Preprocess the data in the xml file
    pubmedarticlelists = PreprocessData()

    # Create Model

if __name__ =='__main__':
    main()
