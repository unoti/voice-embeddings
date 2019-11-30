import minibatch
import application

def train():
    #model = 
    print('Building speaker db')
    speaker_db = application.make_speaker_db()
    print('building batch')
    batch = minibatch.create_batch(speaker_db)
    print('batch.X shape',batch.X.shape)
    print('batch.X len=',len(batch.X))
    print('batch.x[0].shape',batch.X[0].shape)
    print('batch.x[1].shape',batch.X[1].shape)
    print('batch.Y len=',len(batch.Y))

if __name__ == '__main__':
    train()