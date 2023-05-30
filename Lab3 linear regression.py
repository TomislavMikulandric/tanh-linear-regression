# Primjer je preuzet s: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
# i malo unaprijeđen :)
# Ideja primjera je predstavljanje sinusoide kubnom jednadžbom
import torch
import math
import matplotlib.pyplot as plt

#dtype za sve vrijednosti će nam biti float
dtype = torch.float
#sve ćemo staviti na gpu.
device = torch.device("cuda")

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
# Izračunamo sinuse od broja
y = torch.sin(x)

plt.plot(x.cpu(), y.cpu(), '-r', label='y=sin(x)')
plt.title('Graph of y=sin(x)')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.show()

# Ovaj kod priprema podatke na način (x, x^2, x^3)
p = torch.tensor([1, 2, 3], device=device)
xx = x.unsqueeze(-1).pow(p)

# Kreiramo jednostavni linearni sekvencijski model
# Ulaz je 3, a izlaz je 1
# Flatten izravnava izlaz na 1D tensor
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
).cuda()

loss_fn = torch.nn.MSELoss(reduction='sum').cuda()

learning_rate = 1e-6
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# Training petlja
for t in range(2001):
    # y = a + b x + c x^2 + d x^3
    y_pred = model(xx)

    loss = loss_fn(y_pred, y)
    if t % 100 == 0:
        print(t, loss.item())

    # Resetiramo gradijente
    optimizer.zero_grad()

    # Propagiranje greške unazad
    loss.backward()

    # Optimizator ažurira vrijednosti
    optimizer.step()

    # DParametri modela
    linear_layer = model[0]

    a = linear_layer.bias.item()
    b = linear_layer.weight[:, 0].item()
    c = linear_layer.weight[:, 1].item()
    d = linear_layer.weight[:, 2].item()

    #Prikazujemo graf učenja
    if t % 500 == 0:
        y_graph = a + b * x + c * x ** 2 + d * x ** 3
        plt.plot(x.cpu().detach(), y_graph.cpu().detach(), '-r', label='t = ' + str(t) + ' y=a + b * x + c * x ** 2 + d * x ** 3')
        plt.title('Graph of y = a + bx + cx^2 + dx^3')
        plt.xlabel('x', color='#1C2833')
        plt.ylabel('y', color='#1C2833')
        plt.legend(loc='upper left')
        plt.grid()
        plt.show()

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')