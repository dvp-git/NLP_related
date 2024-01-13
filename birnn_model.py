import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size=len(TEXT.vocab), embedding_dim=300, hidden_dim=256, output_dim=1, n_layers=2, 
                 bidirectional=True, dropout=0.5, pad_idx=TEXT.vocab.stoi[TEXT.pad_token],directions=2):
        
        super().__init__()
        
        #TO-DO
        #1. Initialize Embedding Layer
        self.emb = nn.Embedding(vocab_size,embedding_dim,pad_idx)
        self.embedding = nn.Embedding(vocab_size,embedding_dim,pad_idx)

        #2. Initialize RNN layer
        self.brn = nn.GRU(input_size=embedding_dim,
                  hidden_size=hidden_dim,
                  num_layers=n_layers,
                  bidirectional=bidirectional,
                  dropout=dropout)

        #3. Initialize a fully connected layer with Linear transformation
        self.fc = nn.Linear(hidden_dim*2,output_dim)

        #4. Initialize Dropout
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_dim
        self.directions = directions
        
    def forward(self, text, text_lengths):
        #text = [sent_len, batch_size]

        #TO-DO
        #1. Apply embedding layer that matches each word to its vector and apply dropout. Dim [sent_len, batch_size, emb_dim]
        emb = self.emb(text)

        emb = self.dropout(emb)

        # packed = nn.utils.rnn.pack_padded_sequence(emb, text_lengths, batch_first=False, enforce_sorted=False)

        #2. Run the RNN along the sentences of length sent_len. 

        #output = [sent len, batch size, hid dim * num directions];
        #hidden = [num layers * num directions, batch size, hid dim]
        
        _ , hidden = self.brn(emb)


        # Defining batch
        batch = text.shape[1]
        #3. Concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers and apply dropout
        hidden = hidden.view(-1, self.directions, batch, self.hidden_size)
        hidden = hidden.sum(0)    # Summing the 2 hidden states
        hidden = torch.tanh(self.dropout(torch.cat([hidden[0,:,:], hidden[1,:,:]],-1))) # batch and 2*hidden size
        return self.fc(hidden)