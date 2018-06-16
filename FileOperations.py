import pickle
import os.path


class FileOperations:
    def __init__(self):
        pass

    def SaveFile(self, filepath, obj, mode='wb'):

        with open(filepath, mode) as picklefile:
            if type(obj) != 'type':
                pickle.dump(obj, picklefile, pickle.HIGHEST_PROTOCOL)
            else:
                for data in obj:
                    picklefile.write(id)
                    picklefile.write('\n')

    def LoadFile(self, filepath):
        with open(filepath, 'rb') as input:
            filedata = pickle.load(input)
            return filedata

    def exists(self, filepath):
        return os.path.exists(filepath)
