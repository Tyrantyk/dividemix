import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.autograd import Variable



class swav(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.batch_size = args.batch_size
        self.proj_output_dim = args.proj_output_dim
        self.proj_hidden_dim = args.proj_hidden_dim
        self.num_prototypes = args.num_prototypes

        # backbone
        self.backbone = torchvision.models.resnet18(num_classes=10)
        self.backbone.fc = nn.Identity()
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=2, bias=False
        )
        self.backbone.maxpool = nn.Identity()
        self.features_dim = self.backbone.inplanes

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.proj_output_dim, self.proj_hidden_dim),
            nn.BatchNorm1d(self.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.proj_hidden_dim, self.proj_output_dim),
        )

        # prototypes
        self.prototypes = nn.utils.weight_norm(
            nn.Linear(self.proj_output_dim, self.num_prototypes, bias=False)
        )
        self.linear1 = nn.Linear(self.features_dim, self.proj_output_dim)
        self.classifier = nn.Linear(self.proj_output_dim, 10)

    def forward(self, x, grad=True):
        feats1 = self.backbone(x)
        feats1 = self.linear1(feats1)

        z1 = F.normalize(self.projector(feats1))

        p1 = self.prototypes(z1)
        if grad:
            logits = self.classifier(feats1)
        else:
            logits = self.classifier(feats1.detach())

        return feats1, z1, logits, p1