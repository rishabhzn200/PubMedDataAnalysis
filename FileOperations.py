import pickle
import os

# Class to define all the File Operations
class FileOperations:
    def __init__(self):
        pass

    # Function to save the file using pickle
    def SaveFile(self, filepath, obj, mode='wb'):

        with open(filepath, mode) as picklefile:
            if mode == 'w':
                for data in obj:
                    picklefile.write(data)
                    picklefile.write('\n')

            elif type(obj) != 'type':
                pickle.dump(obj, picklefile, pickle.HIGHEST_PROTOCOL)
            else:
                for data in obj:
                    picklefile.write(data)
                    picklefile.write('\n')

    # Function to load the file using pickle
    def LoadFile(self, filepath):
        with open(filepath, 'rb') as input:
            filedata = pickle.load(input)
            return filedata

    # Function to check if the file exists
    def exists(self, filepath):
        return os.path.exists(filepath)
