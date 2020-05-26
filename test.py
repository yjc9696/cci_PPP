import torch
device = torch.device('cuda:0')
print(torch.cuda.is_available())
print(device)
x = torch.rand(5, 3)

y = torch.rand(5, 3)

if torch.cuda.is_available():
    x = x.cuda()

    y = y.cuda()

    print(x + y)