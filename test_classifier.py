#########################################
# Author:       Gentry Atkinson         #
# Organization: Texas State University  #
# Date:         03 Feb, 2023            #
#########################################

import torch
from torch import nn
import logging
import tqdm


class TestClassifier(nn.Module):

    def __init__(self, args) -> None:
        super().__init__()
        model = nn.Sequential(
            nn.LazyConv1d(64, 8),
            nn.Dropout1d(0.25),
            nn.ReLU(),
            nn.MaxPool1d(8),
            nn.LazyConv1d(64, 4),
            nn.Dropout1d(0.25),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.AvgPool2d(4),
            nn.LazyLinear(32),
            nn.LazyLinear(args.num_classes),
            nn.Softmax()
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    def train(self, args, dataloader, logger):
        for epoch in range(args.epochs):
            logging.info(f"Starting epoch {epoch}:")
            pbar = tqdm(dataloader)
            for i, (signals, labels) in enumerate(pbar):
                signals = signals.to(args.device).to(torch.float)
                labels = labels.to(args.device.to(torch.long))
                p = self.model(signals)
                loss = self.criterion(p, labels)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

    def test(self, args, dataloader, logger):
        pass

