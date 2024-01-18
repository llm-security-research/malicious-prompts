import torch.nn as nn

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(SiameseNetwork, self).__init__()
    
        self.embedding = nn.Sequential(
            nn.Linear(384, 512),
            nn.ReLU(inplace=True), 
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, embedding_dim),
        )

    def forward_one(self, x):
        return self.embedding(x)

    def forward(self, input1, input2, input3):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        output3 = self.forward_one(input3)

        return output1, output2, output3