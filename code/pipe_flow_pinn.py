!pip install pyDOE  
import torch  
import torch.nn as nn  
import numpy as np  
import matplotlib.pyplot as plt  
from pyDOE import lhs  
  پارامترهای دامنه و فیزیکی   
# 
R = 0.01  #  شعاع لوله  
L = 1.0   # طول لوله  
mu = 1e-3 #  ویسکوزیته  
rho = 1.0 #  چگالی  
D = 2 * R # قطر لوله  
u0 = 1.0  #  سرعت ورودی مشخص  
  برای سرعت و فشار PINN تعریف شبکه عصبی   
# 
class PINN(nn.Module):  
    def __init__(self):  
        super(PINN, self).__init__()  
        self.net = nn.Sequential(  
            nn.Linear(3, 64),  
            nn.Tanh(),  
            nn.Linear(64, 64),  
            nn.Tanh(),  
            nn.Linear(64, 4)  #  ها خروجی : u_r, u_theta, u_z, p  
        )  
  
    def forward(self, x):  
        return self.net(x)  
  تعداد نقاط   
# 
N_c = 10000  
 
r_c = R * np.sqrt(lhs(1, N_c))  
theta_c = 2 * np.pi * lhs(1, N_c)  
z_c = L * lhs(1, N_c)  
X_c = np.hstack((r_c, theta_c, z_c))  
X_c_tensor = torch.tensor(X_c, dtype=torch.float32)  
  z=0 در u_z تعریف تابع شرط مرزی ورودی برای   
# 
def inlet_condition(model, N=1000):  
    r_vals = torch.linspace(0, R, N).view(-1, 1)  
    z_in = torch.zeros_like(r_vals)  
    theta_in = torch.full_like(r_vals, np.pi / 2)  
    X_inlet = torch.cat([r_vals, theta_in, z_in], dim=1)  
    u_z_pred = model(X_inlet)[:, 2]  
    u_z_target = torch.full_like(u_z_pred, u0)  
    return torch.mean((u_z_pred - u_z_target) ** 2)  
  z=L شرط مرزی خروجی: فشار صفر در  
# 
def outlet_condition(model, N=1000):  
    r_vals = torch.linspace(0, R, N).view(-1, 1)  
    z_out = torch.full_like(r_vals, L)  
    theta_out = torch.full_like(r_vals, np.pi / 2)  
    X_outlet = torch.cat([r_vals, theta_out, z_out], dim=1)  
    p_pred = model(X_outlet)[:, 3]  
    return torch.mean(p_pred ** 2)  
  تعریف تابع ضرر شامل معادله پیوستگی و شرط مرزی ورودی و خروجی   
# 
  
def loss_function(model, X_c):  
    X_c.requires_grad = True  
    output = model(X_c)  
    u_r, u_theta, u_z, p = output[:, 0], output[:, 1], output[:, 2],  
output[:, 3]  
  
    r = X_c[:, 0]  
    theta = X_c[:, 1]  
    z = X_c[:, 2]  
  مشتقات  #    
  
    grads = torch.autograd.grad(u_r, X_c, torch.ones_like(u_r),  
create_graph=True)[0]  
    du_r_dr = grads[:, 0]  
    du_r_dtheta = grads[:, 1] / r  
    du_r_dz = grads[:, 2]  
ل 
  
    grads = torch.autograd.grad(u_theta, X_c,  
torch.ones_like(u_theta), create_graph=True)[0]  
    du_theta_dr = grads[:, 0]  
    du_theta_dtheta = grads[:, 1] / r  
    du_theta_dz = grads[:, 2]  
  
    grads = torch.autograd.grad(u_z, X_c, torch.ones_like(u_z),  
create_graph=True)[0]  
    du_z_dr = grads[:, 0]  
    du_z_dtheta = grads[:, 1] / r  
    du_z_dz = grads[:, 2]  
  
    grads = torch.autograd.grad(p, X_c, torch.ones_like(p),  
create_graph=True)[0]  
    dp_dr = grads[:, 0]  
    dp_dtheta = grads[:, 1] / r  
    dp_dz = grads[:, 2]  
  nabla · u = 0\ :معادله پیوستگی #    
  
    continuity = du_r_dr + u_r / r + du_theta_dtheta + du_z_dz  
  مجموع ضرر شامل پیوستگی، شرط مرزی ورودی و خروجی  #    
  
    loss = torch.mean(continuity**2) + inlet_condition(model) +  
outlet_condition(model)  
    return loss  
  آموزش مدل   
# 
model = PINN()  
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  
  
for epoch in range(10000):  
    optimizer.zero_grad()  
    loss = loss_function(model, X_c_tensor)  
    loss.backward()  
    optimizer.step()  
    if epoch % 500 == 0:  
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")  
  های سرعت و فشار ترسیم مؤلفه   
# 
r_plot = np.linspace(0, R, 100)  
z_plot = np.linspace(0, L, 100)  
R_grid, Z_grid = np.meshgrid(r_plot, z_plot, indexing='ij')  
Theta_fixed = np.full_like(R_grid, np.pi/2)  
  
 
X_plot = np.stack([R_grid.flatten(), Theta_fixed.flatten(),  
Z_grid.flatten()], axis=1)  
X_plot_tensor = torch.tensor(X_plot, dtype=torch.float32)  
  
with torch.no_grad():  
    output = model(X_plot_tensor)  
    u_r_pred = output[:, 0].numpy().reshape(R_grid.shape)  
    u_theta_pred = output[:, 1].numpy().reshape(R_grid.shape)  
    u_z_pred = output[:, 2].numpy().reshape(R_grid.shape)  
    p_pred = output[:, 3].numpy().reshape(R_grid.shape)  
  
plt.figure(figsize=(8, 5))  
plt.contourf(R_grid, Z_grid, u_r_pred, levels=50, cmap='jet')  
plt.colorbar(label='u_r')  
plt.xlabel('r')  
plt.ylabel('z')  
plt.title('Radial Velocity Component $u_r$')  
plt.show()  
  
plt.figure(figsize=(8, 5))  
plt.contourf(R_grid, Z_grid, u_theta_pred, levels=50, cmap='jet')  
plt.colorbar(label='u_θ')  
plt.xlabel('r')  
plt.ylabel('z')  
plt.title('Azimuthal Velocity Component $u_\theta$')  
plt.show()  
  
plt.figure(figsize=(8, 5))  
plt.contourf(R_grid, Z_grid, u_z_pred, levels=50, cmap='jet')  
plt.colorbar(label='u_z')  
plt.xlabel('r')  
plt.ylabel('z')  
plt.title('Axial Velocity Component $u_z$')  
plt.show()  
  
plt.figure(figsize=(8, 5))  
plt.contourf(R_grid, Z_grid, p_pred, levels=50, cmap='viridis')  
plt.colorbar(label='p')  
plt.xlabel('r')  
plt.ylabel('z')  
plt.title('Pressure Contour $p$')  
plt.show()  
  )r = 0( نمودار افت فشار در راستای لوله در شعاع مرکزی  
# 
z_center = np.linspace(0, L, 100)  
  PINN 19 کد  : ه ا   یص 
ل 
  
r_center = np.zeros_like(z_center)  
theta_center = np.full_like(z_center, np.pi/2)  
X_centerline = np.stack([r_center, theta_center, z_center], axis=1)  
X_centerline_tensor = torch.tensor(X_centerline, dtype=torch.float32)  
  
with torch.no_grad():  
    p_centerline = model(X_centerline_tensor)[:, 3].numpy()  
  
plt.figure(figsize=(8, 5))  
plt.plot(z_center, p_centerline, label='Pressure')  
plt.xlabel('z')  
plt.ylabel('Pressure')  
plt.title('Pressure Along Pipe Centerline')  
plt.grid(True)  
plt.legend()  
plt.show()  
  محاسبه ضریب اصطکاک بر اساس گرادیان فشار   
# 
pressure_gradient = np.gradient(p_centerline, z_center)  
friction_factor = D * np.abs(pressure_gradient) / (0.5 * rho *  
np.max(u_z_pred)**2)  
  
plt.figure(figsize=(8, 5))  
plt.plot(z_center, friction_factor, label='Friction Factor')  
plt.xlabel('z')  
plt.ylabel('f')  
plt.title('Friction Factor Along Pipe')  
plt.grid(True)  
plt.legend()  
plt.show()   میخوام این هم در فولدر چدا بزارم
