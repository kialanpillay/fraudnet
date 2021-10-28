import torch
import torch.nn as nn


def train(model, dataset, args):
    criterion = nn.BCELoss()

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False)

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
