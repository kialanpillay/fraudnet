import torch
import torch.nn as nn
import torch.utils.data

from evaluation import validate


def train(model, train_dataset, validation_dataset, args):
    criterion = nn.BCELoss()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    for epoch in range(args.num_epochs):
        print('Epoch {:2d}'.format(epoch + 1))
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if i % 500 == 0:
                print('Iteration {:<4d}  | Loss: {:5.6f}'.format(i, loss.item()))

        validate(model, validation_dataset, args)
