import torch
import torch.nn as nn

### first, let's create a CNN that replaces the prox operator in PGD.
class prox_net(nn.Module):
    """
    This class defines replaces proximal operator using a CNN. Specifically, the CNN
    has 3 convolutional layers with kernel size 3x3 and 32 filters. The activation
    function is PReLU with a single parameter. The input to the network is the current
    iterate and the gradient.
    """
    def __init__(self, n_in_channels=2, n_out_channels=1, n_filters=32, kernel_size=3):
        """
        Initializes the proximal operator network.

        Parameters
        ----------
        n_in_channels : int
            Number of input channels to the network. Default is 2.
        n_out_channels : int
            Number of output channels of the network. Default is 1.
        n_filters : int
            Number of filters in the convolutional layers. Default is 32.
        kernel_size : int
            Kernel size of the convolutional layers. Default is 3.
        
        Returns
        -------
        None
        """
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
        """
        This is the completed forward function of the proximal operator network, for
        part 2 of Question 3. The input to the network is the current iterate and the
        gradient.

        Parameters
        ----------
        x : torch.Tensor
            The current iterate.
        u : torch.Tensor
            The gradient.
        
        Returns
        -------
        dx : torch.Tensor
            The output of the network.
        """
        # Concatenate the current iterate and the gradient
        input = torch.cat((x,u),dim=0)
        dx = self.act1(self.conv1(input))
        dx = self.act2(self.conv2(dx))
        dx = self.conv3(dx)

        return dx
    
class LGD_net(nn.Module):
    """
    This class defines the LGD network, which is a learned gradient descent network.
    The network consists of a series of proximal operators, each followed by a learnable
    step size. The input to the network is the noisy image and the initial guess.
    """
    def __init__(self, step_size, fwd_op, adj_op, niter=5):
        """
        Initializes the LGD network.

        Parameters
        ----------
        step_size : float
            The initial guess of step size for the network.
        fwd_op : odl_torch.OperatorModule
            The forward operator (A).
        adj_op : odl_torch.OperatorModule
            The adjoint operator (A^T).
        niter : int
            Number of unrolled iterations for the network. Default is 5.

        Returns
        -------
        None
        """
        super(LGD_net, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.niter = niter
        self.prox = nn.ModuleList([prox_net().to(device) for _ in range(self.niter)])
        self.step_size = nn.Parameter(step_size * torch.ones(self.niter).to(device))
        self.fwd_op = fwd_op
        self.adj_op = adj_op
    def forward(self, y, x_init):
        """
        This is the forward function of the LGD network, for part 2 of Question 3.
        The input to the network is the noisy image and the initial guess.

        Parameters
        ----------
        y : torch.Tensor
            The noisy image.
        x_init : torch.Tensor
            The initial guess. (Usually the FBP reconstruction)
        
        Returns
        -------
        x : torch.Tensor
            The output of the network.
        """
        x = x_init

        for iteration in range(self.niter):
            # Ax - y
            u = self.fwd_op(x) - y

            # A^T(Ax - y)
            grad = self.adj_op(u)

            # Proximal operator
            dx = self.prox[iteration](x,grad)

            # Update the iterate
            x = x + self.step_size[iteration] * dx

        return x
