# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

<img width="954" height="633" alt="image" src="https://github.com/user-attachments/assets/1baf6a15-d9a5-4ee8-88c3-e280cdbf3dd9" />

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
class Neuralnet(nn.Module):
   def __init__(self):
        super().__init__()
        self.n1=nn.Linear(1,10)
        self.n2=nn.Linear(10,20)
        self.n3=nn.Linear(20,1)
        self.relu=nn.ReLU()
        self.history={'loss': []}
   def forward(self,x):
        x=self.relu(self.n1(x))
        x=self.relu(self.n2(x))
        x=self.n3(x)
        return x


# Initialize the Model, Loss Function, and Optimizer
nithi=NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(nithi.parameters(),lr=0.001)

def train_model(nithi, X_train, y_train, criterion, optimizer, epochs=1000):
    # initialize history before loop
    nithi.history = {'loss': []}

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = nithi(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # record loss
        nithi.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


```
## Dataset Information

<img width="191" height="529" alt="image" src="https://github.com/user-attachments/assets/e7a9ab5c-680c-405e-9c1d-7d455a6e16b3" />

## OUTPUT

<img width="480" height="125" alt="image" src="https://github.com/user-attachments/assets/f557f51e-0a10-469e-aa66-ec9d0b01f693" />


### Training Loss Vs Iteration Plot

<img width="1011" height="615" alt="image" src="https://github.com/user-attachments/assets/5ab07ea3-4cdc-4867-aa98-46af5c9ab5c4" />

### New Sample Data Prediction
```
X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = nithi(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
```
<img width="912" height="52" alt="image" src="https://github.com/user-attachments/assets/81d835e9-7edc-47d3-9dc6-42958eaa3fc8" />

## RESULT

Successfully executed the code to develop a neural network regression model.
