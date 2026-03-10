# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

<img width="1919" height="1012" alt="image" src="https://github.com/user-attachments/assets/bacc8538-5b0b-476a-b49c-63aaac61491f" />

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: KUKKADAPU CHARAN TEJ
### Register Number: 212224040167
```python

import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.Linear(1,10)
        self.n2 = nn.Linear(10,20)
        self.n3 = nn.Linear(20,1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self,x):
        x = self.relu(self.n1(x))
        x = self.relu(self.n2(x))
        x = self.n3(x)
        return x

# Initialize the Model, Loss Function, and Optimizer

model = NeuralNet()

criterion = nn.MSELoss()

optimizer = optim.RMSprop(model.parameters(), lr=0.001)
## Create Training Data

X_train = torch.linspace(-5,5,100).view(-1,1)
y_train = X_train**2

## Training Function
def train_model(model, X_train, y_train, criterion, optimizer, epochs=1000):
    model.history = {'loss': []}

    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model(X_train)

        loss = criterion(outputs, y_train)

        loss.backward()

        optimizer.step()

        model.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
## run training
train_model(model, X_train, y_train, criterion, optimizer)

## Plot Training Loss
import matplotlib.pyplot as plt

plt.plot(model.history['loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()


```
## Dataset Information

<img width="191" height="529" alt="image" src="https://github.com/user-attachments/assets/e7a9ab5c-680c-405e-9c1d-7d455a6e16b3" />

## OUTPUT

<img width="483" height="126" alt="image" src="https://github.com/user-attachments/assets/52bfbe89-6f0c-41c9-b7e5-060956da38b7" />


### Training Loss Vs Iteration Plot

<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/e2a2e045-b76d-4f62-9558-8b3b2675d1c3" />

### New Sample Data Prediction
```py 
X_n1_1 = torch.tensor([[9]], dtype=torch.float32)

prediction = model(
    torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)
).item()

print(f'Prediction: {prediction}')
```


<img width="729" height="244" alt="image" src="https://github.com/user-attachments/assets/64f0281a-1b90-4b2c-8f73-961eb805c9c0" />

## RESULT

Successfully executed the code to develop a neural network regression model.
