import torch.nn as nn
import torch


def get_loss(loss_module_name, loss_fn_args):
    loss_fn = eval(loss_module_name)
    return loss_fn(**loss_fn_args)


def accuracy_fn(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


class CrossEntropyLoss:
    def __init__(self):
        self.name_terms_to_return = [("CrossEntropyLoss", True), ("Accuracy", True)]
        # In list below, second argument of tuples are wheather to get mean or not
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()

    def __call__(self, model, data, labels, optimizer, device, is_training):
        data = data.to(device)
        labels = labels.to(device)
        labels_pred = model(data)
        probs = self.softmax(labels_pred)
        loss = self.criterion(labels_pred, labels)
        accuracy = accuracy_fn(output=labels_pred, target=labels)
        if not is_training:
            return (loss.item(), accuracy[0], probs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return (loss.item(), accuracy[0], probs)
