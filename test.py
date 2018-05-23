import sys
sys.path.insert(0, 'scripts')

import models   

def main():
    path = input('Enter file path of your model:')
    att_model = models.AttentionModel(path)
    att_model.quality()
    
if __name__ == '__main__':
    main()