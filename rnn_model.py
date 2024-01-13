from thinc.layers import dropout
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, vocab_size=len(TEXT.vocab), embedding_dim=300, hidden_dim=256, output_dim=1, n_layers=2, 
                 bidirectional=False, dropout=0.5, pad_idx=TEXT.vocab.stoi[TEXT.pad_token]):

        super().__init__()
        
        #TO-DO
        #1. Initialize Embedding Layer
        self.emb = nn.Embedding(vocab_size,embedding_dim,pad_idx)
        self.embedding = nn.Embedding(vocab_size,embedding_dim,pad_idx)


        #2. Initialize RNN layer
        self.rnn = nn.RNN(input_size=embedding_dim, 
                          hidden_size=hidden_dim,
                          num_layers=n_layers,
                          dropout=dropout,
                          bidirectional=False)

        #3. Initialize a fully connected layer with Linear transformation
        self.fc = nn.Linear(hidden_dim, output_dim)

        #4. Initialize Dropout
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, text, text_lengths):
        #text = [sent_len, batch_size]

        #TO-DO
        #1. Apply embedding layer that matches each word to its vector and apply dropout. Dim [sent_len, batch_size, emb_dim]
        emb = self.emb(text)  

        # packed = nn.utils.rnn.pack_padded_sequence(emb, text_lengths, enforce_sorted=False)

        #2. Run the RNN along the sentences of length sent_len. 
        output , hidden = self.rnn(emb)


        #output = [sent len, batch size, hid dim * num directions]; 
        #hidden = [num layers * num directions, batch size, hid dim]
        

        #3. Get last forward (hidden[-1,:,:]) hidden layer and apply dropout
        hidden = torch.tanh(self.dropout(hidden[-1,:,:]))
        return self.fc(hidden)