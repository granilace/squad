import sys
sys.path.insert(0, 'scripts')

import models   

def main():
    epochs = input('Enter amount of epochs')
    att_model = models.AttentionModel()
    att_model.train(n_epochs=epochs)
    
if __name__ == '__main__':
    main()