from torch import einsum, nn
from torch.nn.functional import relu
import torch

class MultipleRankingLossBiEncoder(nn.Module):
    """
    Triplet Margin Loss function.
    """

    def __init__(self, device, temperature=1):
        super(MultipleRankingLossBiEncoder, self).__init__()
        self.CELoss = nn.CrossEntropyLoss()
        self.temperature = temperature
        
        self.device = device
    
    def forward(
        self,
        anchors,
        positives
    ):
        pw_similarity = torch.mm(anchors / self.temperature, positives.T)
        labels = torch.tensor([x for x in range(anchors.shape[0])], device=self.device)
        
        pw_loss = self.CELoss(pw_similarity, labels)
        
        return pw_loss, (pw_similarity.argmax(dim=1, keepdim=True).squeeze() == labels)

    def val_forward(self, anchors, positives):
        return self.forward(anchors, positives)
