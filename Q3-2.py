# import libraries for CT and deep learning
import odl
import numpy as np
import matplotlib.pyplot as plt
import astra
astra.test()
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

import torch
import random
import odl
import odl.contrib.torch as odl_torch
from LGD_models import LGD_net

# Set random seeds for reproducibility
np.random.seed(29)
random.seed(29)
torch.manual_seed(29)

img_size = 256 # discretized with 256 samples per dimension
reco_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20],
                               shape=[img_size, img_size], dtype='float32')
# Make a parallel beam geometry with flat detector, using number of angles = num_angles
num_angles = 30
geometry = odl.tomo.parallel_beam_geometry(reco_space, num_angles=num_angles)

# Create the forward operator, adjoint operator, and the FBO operator in ODL
fwd_op_odl = odl.tomo.RayTransform(reco_space, geometry)
fbp_op_odl = odl.tomo.fbp_op(fwd_op_odl,filter_type='Ram-Lak',
                             frequency_scaling=0.6)
adj_op_odl = fwd_op_odl.adjoint


# Create phantom and noisy projection data in ODL
phantom_odl = odl.phantom.shepp_logan(reco_space, modified=True)
data_odl = fwd_op_odl(phantom_odl)
data_odl += odl.phantom.white_noise(fwd_op_odl.range, seed=29) * np.mean(data_odl) * 0.1
fbp_odl = fbp_op_odl(data_odl)

# convert the image and the sinogram to numpy arrays
phantom_np = phantom_odl.__array__()
fbp_np = fbp_odl.__array__()
data_np = data_odl.__array__()

data_range = np.max(phantom_np) - np.min(phantom_np)
psnr_fbp = compare_psnr(phantom_np,fbp_np,data_range=data_range)
ssim_fbp = compare_ssim(phantom_np,fbp_np,data_range=data_range)

# Gradient operator for the TV part
grad = odl.Gradient(reco_space)

# Stacking of the two operators
L = odl.BroadcastOperator(fwd_op_odl, grad)

# Data matching and regularization functionals
data_fit = odl.solvers.L2NormSquared(fwd_op_odl.range).translated(data_odl)
lam = 0.015
reg_func = lam * odl.solvers.L1Norm(grad.range)
g = odl.solvers.SeparableSum(data_fit, reg_func)

# We don't use the f functional, setting it to zero
f = odl.solvers.ZeroFunctional(L.domain)

# --- Select parameters and solve using ADMM --- #

# Estimated operator norm, add 10 percent for some safety margin
op_norm = 1.1 * odl.power_method_opnorm(L, maxiter=20)

niter = 200  # Number of iterations
sigma = 2.0  # Step size for g.proximal
tau = sigma / op_norm ** 2  # Step size for f.proximal

# Optionally pass a callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration(step=10) &
            odl.solvers.CallbackShow(step=10))

# Choose a starting point
x_admm_odl = L.domain.zero()

# Run the algorithm
odl.solvers.admm_linearized(x_admm_odl, f, g, L, tau, sigma,
                            niter, callback=None)
x_admm_np = x_admm_odl.__array__()


psnr_tv = compare_psnr(phantom_np, x_admm_np, data_range=data_range)
ssim_tv = compare_ssim(phantom_np, x_admm_np, data_range=data_range)

# Let's compute a reasonable initial value for the step-size 
# as step_size = 1/L, where L is the spectral norm of the forward operator.
op_norm = 1.1 * odl.power_method_opnorm(fwd_op_odl)
step_size = 1 / op_norm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fbp_op = odl_torch.OperatorModule(fbp_op_odl).to(device)
fwd_op = odl_torch.OperatorModule(fwd_op_odl).to(device)
adj_op = odl_torch.OperatorModule(adj_op_odl).to(device)


lgd_net = LGD_net(step_size=step_size, fwd_op=fwd_op, adj_op=adj_op).to(device) # realize the network and export it to GPU
num_learnable_params = sum(p.numel() for p in lgd_net.parameters() if p.requires_grad)
print('number of model parameters = {}'.format(num_learnable_params))
y = torch.from_numpy(data_np).to(device).unsqueeze(0) # noisy sinogram data as a torch tensor

# Use the FBP reconstruction as the initial value
x_init = fbp_op(y)

# target ground-truth as a torch tensor
ground_truth = torch.from_numpy(phantom_np).to(device).unsqueeze(0)

# define the loss and the optimizer
mse_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lgd_net.parameters(),lr = 1e-4)
num_epochs = 2000

##### training loop
for epoch in range(0, num_epochs):
    optimizer.zero_grad()
    '''################## YOUR CODE HERE #################### '''
    recon = lgd_net(y, x_init)
    loss = mse_loss(recon, ground_truth)
    loss.backward()
    optimizer.step()

    if(epoch%100 == 0):
        print('epoch = {}, loss = {}'.format(epoch, loss.item()))


lgd_recon_np = recon.detach().cpu().numpy().squeeze() # convert the LGD reconstruction to numpy format
# Let's display the reconstructed images by LGD and compare it with FBP and ADMM
plt.subplot(141)
plt.imshow(phantom_np.transpose(),cmap='bone')
plt.xticks([])
plt.yticks([])
plt.title('ground-truth')

plt.subplot(142)
plt.imshow(fbp_np.transpose(),cmap='bone')
plt.xticks([])
plt.yticks([])
plt.title('FBP')
plt.xlabel('PSNR: {:.2f} dB, SSIM: {:.2f}'.format(psnr_fbp, ssim_fbp))


plt.subplot(143)
plt.imshow(x_admm_np.transpose(),cmap='bone')
plt.xticks([])
plt.yticks([])
plt.title('TV')
plt.xlabel('PSNR: {:.2f} dB, SSIM: {:.2f}'.format(psnr_tv, ssim_tv))
plt.gcf().set_size_inches(9.0,6.0)


plt.subplot(144)
plt.imshow(lgd_recon_np.transpose(),cmap='bone')
plt.xticks([])
plt.yticks([])
plt.title('LGD')
psnr_lgd = compare_psnr(phantom_np,lgd_recon_np,data_range=data_range)
ssim_lgd = compare_ssim(phantom_np,lgd_recon_np,data_range=data_range)
plt.xlabel('PSNR: {:.2f} dB, SSIM: {:.2f}'.format(psnr_lgd, ssim_lgd))
plt.gcf().set_size_inches(12.0,6.0)
plt.savefig('figures/phantom_recon_comparison.png')
