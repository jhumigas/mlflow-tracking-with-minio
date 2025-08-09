import marimo

__generated_with = "0.14.16"
app = marimo.App()


@app.cell
def _():
    from time import time
    from typing import List, Dict, Any, Tuple

    import mlflow
    import torch
    from torch.utils.data.dataloader import DataLoader
    from torch import nn
    from torch import optim
    from torchvision import datasets, transforms
    return (
        Any,
        DataLoader,
        Dict,
        List,
        Tuple,
        datasets,
        mlflow,
        nn,
        optim,
        time,
        torch,
        transforms,
    )


@app.cell
def _(Any, DataLoader, Tuple, datasets, time, transforms):
    def load_images(batch_size: int) -> Tuple[Any]:
        # Start of load time.
        start_time = time()

        # Define a transform to normalize the data
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

        # Download and load the training data
        train_dataset = datasets.MNIST('./data/mnistdata', download=True, train=True, transform=transform)
        test_dataset = datasets.MNIST('./data/mnistdata', download=True, train=False, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, test_loader, len(train_dataset), len(test_dataset), (time()-start_time)
    return (load_images,)


@app.cell
def _(List, nn):
    class MNISTModel(nn.Module):
        def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
            super().__init__()

            self.lin1 = nn.Linear(input_size, hidden_sizes[0])
            self.lin2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
            self.lin3 = nn.Linear(hidden_sizes[1], output_size)
            self.activation = nn.ReLU()
            self.output_activation = nn.LogSoftmax(dim=1)

        def forward(self, x):
            out = self.lin1(x)
            out = self.activation(out)
            out = self.lin2(out)
            out = self.activation(out)
            out = self.lin3(out)
            out = self.output_activation(out)
            return out
    return (MNISTModel,)


@app.cell
def _(Any, DataLoader, Dict, MNISTModel, mlflow, nn, optim, time):
    def train_model(model: MNISTModel, loader: DataLoader, params: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time()
        loss_func = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'])
        training_metrics = {}
        for epoch in range(params['epochs']):
            total_loss = 0
            for images, labels in loader:
                # Flatten MNIST images into a 784 long vector.
                images = images.view(images.shape[0], -1)

                # Training pass
                optimizer.zero_grad()

                output = model(images)
                loss = loss_func(output, labels)

                # This is where the model learns by backpropagating
                loss.backward()

                # And optimizes its weights here
                optimizer.step()

                total_loss += loss.item()
            else:
                mlflow.log_metric('training_loss', total_loss/len(loader), epoch+1)
                print("Epoch {} - Training loss: {}".format(epoch+1, total_loss/len(loader)))

        training_time_sec = (time()-start_time)
        training_metrics['training_time_sec'] = training_time_sec
        print("\nTraining Time (in seconds) =",training_time_sec)
        return training_metrics
    return (train_model,)


@app.cell
def _(Any, DataLoader, Dict, MNISTModel, torch):
    def evaluate_model(model: MNISTModel, loader: DataLoader) -> Dict[str, Any]:
        correct_count, total_count = 0, 0
        for images,labels in loader:
            for i in range(len(labels)):
                img = images[i].view(1, 784)
                # Turn off gradients to speed up this part
                with torch.no_grad():
                    logps = model(img)

                # Output of the network are log-probabilities, need to take exponential for probabilities
                ps = torch.exp(logps)
                probab = list(ps.numpy()[0])
                pred_label = probab.index(max(probab))
                true_label = labels.numpy()[i]
                if(true_label == pred_label):
                    correct_count += 1
                total_count += 1

        testing_metrics = {
            'incorrect_count': total_count-correct_count,
            'correct_count': correct_count,
            'accuracy': (correct_count/total_count)
        }
        print("Number Of Images Tested =", total_count)
        print("\nModel Accuracy =", (correct_count/total_count))
        return testing_metrics
    return (evaluate_model,)


@app.cell
def _(MNISTModel, load_images, mlflow, train_model):
    # Setup parameters
    params = {
        'batch_size': 64,
        'epochs': 35,
        'input_size': 784,
        'hidden_sizes': [128, 64],
        'lr': 0.035,
        'momentum': 0.5,
        'output_size': 10
        }

    # Setup mlflow to point to our server.
    run_name = f'Learning rate={params["lr"]}'
    mlflow.set_tracking_uri('http://localhost:5010/')
    mlflow.set_experiment('MNIST 3-layer network')
    mlflow.start_run(run_name=run_name)

    # Log parameters
    mlflow.log_params(params)

    # Load the data and log loading metrics.
    train_loader, test_loader, train_size, test_size, load_time_sec = load_images(params['batch_size'])
    mlflow.log_metric('train_size', train_size)
    mlflow.log_metric('test_size', test_size)
    mlflow.log_metric('load_time_sec', load_time_sec)

    # Train the model and log training metrics.
    model = MNISTModel(params['input_size'], params['hidden_sizes'], params['output_size'])
    training_metrics = train_model(model, train_loader, params)
    mlflow.log_metrics(training_metrics)





    return model, test_loader


@app.cell
def _(evaluate_model, mlflow, model, test_loader):
    # Test the model and log the accuracy as a metric.
    testing_metrics = evaluate_model(model, test_loader)
    mlflow.log_metrics(testing_metrics)

    # Log the raw data and the trained model as artifacts.
    mlflow.log_artifacts('./mnistdata', artifact_path='mnistdata')
    return


@app.cell
def _(mlflow, model, torch):
    from mlflow.models import infer_signature
    import numpy as np

    # Create sample input and predictions
    sample_input = sample_input = np.random.uniform(size=[1, 784]).astype(np.float32)

    # Get model output - convert tensor to numpy
    with torch.no_grad():
        output = model(torch.tensor(sample_input))
        sample_output = output.numpy()

    # Infer signature automatically
    signature = infer_signature(sample_input, sample_output)

    # Log model with signature
    model_info = mlflow.pytorch.log_model(model, name="mnistmodel", signature=signature)
    return


@app.cell
def _(mlflow):
    # End the run
    mlflow.end_run()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
