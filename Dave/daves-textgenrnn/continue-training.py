from textgenrnn import textgenrnn

with open('DEMF-Movement Festival performers-v2.txt','r') as f_open:
    source = f_open.read()
newmodel = textgenrnn()
newmodel.load("textgenrnn_weights.hdf5")
for epoch in range(1, 10):
    print("Epoch", epoch);
    newmodel.train_from_file('DEMF-Movement Festival performers-v2.txt',
                            new_model=False,
                            rnn_size=128,
                            rnn_layers=2,
                            rnn_bidirectional=True,
                            dim_embeddings=100,
                            num_epochs=1,
                            word_level=False)