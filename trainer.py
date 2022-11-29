import time
import os.path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def train(model: nn.Module,
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          epoch_num: int,
          optimizer: torch.optim.Optimizer,
          scheduler,
          scheduler_step,
          loss_func,
          path: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch = 0
    loss_list = []
    time_list = []
    accu_list = []

    if os.path.isfile(path + '.pth'):
        checkpoint = torch.load(path + '.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        loss_list = checkpoint['loss_list']
        time_list = checkpoint['time_list']
        accu_list = checkpoint['accu_list']

    zero_time = time.time()
    while epoch < epoch_num:  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        iters = len(train_loader)
        start_time = time.time()
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)

            # forward + backward + optimize
            output = model(inputs)
            # loss = criterion(output, labels)
            loss = loss_func(model, output, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None and scheduler_step == 'batch':
                scheduler.step(epoch + i / iters)

            # Time
            end_time = time.time()
            time_taken = end_time - start_time
            total_time_taken = end_time - zero_time

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[{}, {:5d}] loss: {:.3f}'.format(epoch, i + 1, running_loss / 2000), end=' ')
                print('Time: {:.3f} {:.3f}'.format(time_taken, total_time_taken))
                loss_list.append(running_loss / 2000)
                time_list.append(total_time_taken)
                running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        accu_list.append(test(model, val_loader) * 100)
        print('Accuracy: {:.3f}'.format(accu_list[-1]))
        if scheduler is not None and scheduler_step == 'epoch':
            scheduler.step()
        if accu_list[-1] >= np.max(accu_list):
            torch.save(model.state_dict(), path + '_best.pth')
        epoch += 1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss_list': loss_list,
            'time_list': time_list,
            'accu_list': accu_list}, path + '.pth')
    t1 = pd.DataFrame.from_dict({'loss': loss_list, 'time': time_list})
    t1.to_csv(path + '_loss.csv', index=False)
    t2 = pd.DataFrame.from_dict({'accu': accu_list})
    t2.to_csv(path + '_accu.csv', index=False)
    model.load_state_dict(torch.load(path + '_best.pth'))
    model_scripted = torch.jit.script(model)  # Export to TorchScript
    model_scripted.save(path + '.pt')


def test(model: nn.Module,
         test_loader: torch.utils.data.DataLoader):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
