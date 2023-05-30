import torch.nn as nn
import torch


def get_loss(loss_module_name, is_ood, kwargs):
    loss_fn = eval(loss_module_name)
    return loss_fn(is_ood, **kwargs)


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
    def __init__(self, is_ood, **kwargs):
        # In list below, second argument of tuples are wheather to get mean or not
        super().__init__()
        self.name_terms_to_return = [("CrossEntropyLoss", True), ("Accuracy", True)]
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()
        if is_ood:
            self.name_terms_to_return = [("outlier detection performance", True)]
        else:
            self.name_terms_to_return = [("CrossEntropyLoss", True), ("Accuracy", True)]

    def in_distribution_performance(
        self, model, data, labels, optimizer, device, is_learning
    ):
        data = data.to(device)
        labels = labels.to(device)
        labels_pred = model(data)
        loss = self.criterion(labels_pred, labels)
        accuracy = accuracy_fn(output=labels_pred, target=labels)
        if not is_learning:
            return (loss.item(), accuracy[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return (loss.item(), accuracy[0])

    def ood_performance(self, model, data, labels, threshold, device):
        """This function returns number of data that are detected as an outlier.
        ALL the data taht are given in this function are outlier(out of distribution data)
        """

        data = data.to(device)
        labels = labels.to(device)
        labels_pred = model(data)
        probs = self.softmax(labels_pred)
        num_outliers = sum(probs.max(dim=1).values > threshold).item()
        outlier_detection_performance = 1 - num_outliers / data.shape[0]

        return (outlier_detection_performance,)
