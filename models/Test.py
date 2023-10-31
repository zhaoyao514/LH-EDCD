import torch
from torch import nn


def test_img(net_g, my_dataset, test_indices, local_test_bs, device):
    net_g.eval()
    # testing
    data_loader = my_dataset.load_test_dataset(test_indices, local_test_bs)
    loss_func = nn.CrossEntropyLoss()

    eval_acc = 0
    eval_loss = 0
    for idx, (data, label) in enumerate(data_loader):
        data = data.detach().clone().type(torch.FloatTensor)
        if device != torch.device('cpu'):
            data, label = data.to(device), label.to(device)
        log_probs = net_g(data)
        loss = loss_func(log_probs, label)
        eval_loss += loss.item() * label.size(0)
        # get the index of the max log-probability
        y_pred = torch.max(log_probs, 1)[1]
        eval_acc += y_pred.eq(label.data.view_as(y_pred)).long().cpu().sum().item()

    test_loss = eval_loss / len(data_loader.dataset)
    test_acc = 100.00 * eval_acc / len(data_loader.dataset)
    return test_acc, test_loss
