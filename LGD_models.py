import torch
import torch.nn as nn

### first, let's create a CNN that replaces the prox operator in PGD.
class prox_net(nn.Module):
    def __init__(self, n_in_channels=2, n_out_channels=1, n_filters=32, kernel_size=3):
        super(prox_net, self).__init__()
        self.pad = (kernel_size-1) // 2
        self.conv1 = nn.Conv2d(n_in_channels, out_channels=n_filters,
                               kernel_size=kernel_size, stride=1,
                               padding=self.pad, bias=True)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size,
                               stride=1, padding=self.pad, bias=True)
        self.conv3 = nn.Conv2d(n_filters, out_channels=n_out_channels,
                               kernel_size=kernel_size, stride=1,
                               padding=self.pad, bias=True)

        self.act1 = nn.PReLU(num_parameters=1, init=0.25)
        self.act2 = nn.PReLU(num_parameters=1, init=0.25)

    def forward(self, x, u):
        ''' ################## YOUR CODE HERE #################### '''
        ### Note: here the two inputs denote the current iterate and the gradient
        input = torch.cat((x,u),dim=0)
        dx = self.act1(self.conv1(input))
        dx = self.act2(self.conv2(dx))
        dx = self.conv3(dx)

        return dx
    
class LGD_net(nn.Module):
    def __init__(self, step_size, fwd_op, adj_op, niter=5):
        super(LGD_net, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.niter = niter
        self.prox = nn.ModuleList([prox_net().to(device) for _ in range(self.niter)])
        self.step_size = nn.Parameter(step_size * torch.ones(self.niter).to(device))
        self.fwd_op = fwd_op
        self.adj_op = adj_op
    def forward(self, y, x_init):
        x = x_init
        ''' ################## YOUR CODE HERE #################### '''
        #### Note: the gradient at a given x is A^T(Ax-y).
        for iteration in range(self.niter):
            u = self.fwd_op(x) - y
            grad = self.adj_op(u)
            dx = self.prox[iteration](x,grad)
            x = x + self.step_size[iteration] * dx

        return x
