#########################################
# Author:       Gentry Atkinson         #
# Organization: Texas State University  #
# Date:         03 Feb, 2023            #
#########################################

import torch
from torch import nn
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import math

class TestClassifier(nn.Module):

    def __init__(self, args) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.LazyConv1d(64, 8),
            nn.BatchNorm1d(64),
            nn.Dropout1d(0.25),
            nn.ReLU(),
            nn.MaxPool1d(8),
            nn.LazyConv1d(64, 4),
            nn.Dropout1d(0.25),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Flatten(),
            nn.LazyLinear(32),
            nn.LazyLinear(args.num_classes),
            nn.Softmax()
        ).to(args.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(params=self.model.parameters(), lr=args.lr)

    def train(self, args, dataloader, logger=None):
        for epoch in range(args.epochs):
            logging.info(f"Starting epoch {epoch}:")
            pbar = tqdm(dataloader)
            total_loss = 0
            for i, (signals, labels) in enumerate(pbar):
                signals = signals.to(args.device).to(torch.float)
                labels = labels.to(args.device).to(torch.long)
                labels = nn.functional.one_hot(labels, num_classes=args.num_classes).float()
                pred = self.model(signals)
                if args.diffusion_style == 'probabilistic_conditional':
                    #pred = torch.argmax(pred, dim=-1).long()
                    labels = torch.argmax(labels, dim=-1).float()              
                loss = self.criterion(pred, labels)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total_loss += loss.item()
            logging.info(f'Loss: {total_loss/len(dataloader)}')


    def test(self, args, dataloader, logger=None):
        all_preds = None
        all_true = None
        with torch.no_grad():
            for i, (X, y) in enumerate(dataloader):
                X = X.to(args.device)
                y = y.long().to(args.device)
                pred = self.model(X)
                pred = torch.argmax(pred, dim=-1)
                if args.diffusion_style == 'probabilistic_conditional':
                    y = torch.argmax(y, dim=-1)
                if all_preds == None:
                    all_preds = pred
                else:
                    all_preds = torch.concat((all_preds, pred))

                if all_true == None:
                    all_true = y
                else:
                    all_true = torch.concat((all_true, y))
        if args.device == 'cpu':
            return accuracy_score(all_true.detach().numpy(), all_preds.detach().numpy())
        else:
            return accuracy_score(all_true.cpu().detach().numpy(), all_preds.cpu().detach().numpy())

    def train_and_test_classifier(self, args, X, y, logger=None, X_new = None, y_new = None):
        dataset = None
        if X_new == None or y_new == None:
            dataset = torch.utils.data.TensorDataset(X, y)
            test_size = math.ceil(args.test_split* len(dataset))
            train_size = len(dataset) - test_size
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        else:
            test_len = math.ceil(X.shape[0]*args.test_split)
            X_test, X_train = torch.split(X, [test_len, len(X)-test_len])
            y_test, y_train = torch.split(y, [test_len, len(X)-test_len])
            train_dataset = torch.utils.data.TensorDataset(torch.concat((X_train, X_new)), torch.concat((y_train, y_new)))
            test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        self.train(args, dataloader, logger)
        dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        acc = self.test(args, dataloader, logger)
        return acc


if __name__ == '__main__':
    X = torch.randn((1000, 1, 128))
    y = torch.randint(low=0, high=2, size=(1000,3))
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.device = 'cuda'
    args.num_classes = 3
    args.lr = 0.001
    args.num_workers = 8
    args.batch_size = 32
    args.epochs = 5
    args.test_split = 0.2
    args.diffusion_style = 'probabilistic_conditional'

    c = TestClassifier(args)
    # dataset = torch.utils.data.TensorDataset(X, y)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    # c.train(args, dataloader)
    # p = c.test(args, dataloader)
    # print('predicted labels: ', p)

    acc = c.train_and_test_classifier(args, X, y)
    print('Acc: ', acc)

