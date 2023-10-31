import torch
from torch import nn


def train_cnn_mlp(net, my_dataset, idx, local_ep, device, lr, momentum, local_bs):
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    ldr_train = my_dataset.load_train_dataset(idx, local_bs)
    loss_func = nn.CrossEntropyLoss()

    epoch_loss = []
    for _ in range(local_ep):
        batch_loss = []
        for batch_idx, (data, label) in enumerate(ldr_train):
            data = data.detach().clone().type(torch.FloatTensor)
            if device != "cpu":
                data, labels = data.to(device), label.to(device)
            net.zero_grad()
            log_probs = net(data)
            loss = loss_func(log_probs, label)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
    return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

