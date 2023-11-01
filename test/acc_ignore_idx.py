from ignite.metrics import Accuracy
import torch


b, num_class = 3, 3
y = torch.randint(0, 3, (b, ))
y[0] = -100
y_pred = torch.randn((b, num_class))
print(y)
print(y_pred)


metric_fn = Accuracy()
metric_fn.reset()
metric_fn.update((y_pred, y))
acc = metric_fn.compute()

print(acc)