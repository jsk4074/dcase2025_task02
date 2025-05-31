import torch
import torch.nn as nn
import timm

class encoder_model(nn.Module):
    def __init__(self, 
                 model_name='resnet18', 
                 pretrained=True, 
                 input_size=64, 
                 in_chans=1, 
                 embedding_dim=1024):
        super().__init__()

        # Load timm backbone
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            in_chans=in_chans,
            num_classes=0,  
            global_pool='avg'  
        )

        # Find the output embedding size
        dummy = torch.randn(1, in_chans, input_size, input_size)
        with torch.no_grad():
            feat = self.backbone(dummy)
            self.backbone_out_dim = feat.shape[1]

        # Optional FC layers for embedding projection
        self.fc_encoder_embedding = nn.Sequential(
            nn.Linear(self.backbone_out_dim, embedding_dim * 4, bias=False), 
            nn.Dropout(0.2),
            nn.BatchNorm1d(embedding_dim * 4),
            nn.LeakyReLU(),

            nn.Linear(embedding_dim * 4, embedding_dim * 2, bias=False), 
            nn.Dropout(0.2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.LeakyReLU(),

            nn.Linear(embedding_dim * 2, embedding_dim, bias=False), 
            nn.Dropout(0.2), 
            nn.BatchNorm1d(embedding_dim), 
            nn.LeakyReLU()
        )

    def forward(self, x):
        feats = self.backbone(x)                   # [B, backbone_out_dim]
        embeddings = self.fc_encoder_embedding(feats)  # [B, embedding_dim]
        return embeddings
