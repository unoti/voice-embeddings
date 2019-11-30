import minibatch
import application

def train():
    print('creating speaker db')
    speaker_db = application.make_speaker_db()
    batch = minibatch.create_batch(speaker_db)
    print('batch=',batch)

if __name__ == '__main__':
    train()