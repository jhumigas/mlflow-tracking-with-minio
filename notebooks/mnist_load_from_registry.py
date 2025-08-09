import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    from typing import List, Dict, Any, Tuple

    import mlflow
    import mlflow.pyfunc
    import numpy as np
    import torch
    from torch.utils.data.dataloader import DataLoader
    from torchvision import datasets, transforms
    return (
        Any,
        DataLoader,
        Dict,
        Tuple,
        datasets,
        mlflow,
        np,
        torch,
        transforms,
    )


@app.cell
def _(mlflow):
    model_name = 'mnist_classifier'
    model_version = 6

    mlflow.set_tracking_uri('http://localhost:5010/')
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    type(model)
    return (model,)


@app.cell
def _(Any, DataLoader, Tuple, datasets, transforms):
    def load_test_images(batch_size: int) -> Tuple[Any]:
        # Define a transform to normalize the data
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

        # Download and load the testing data
        test_dataset = datasets.MNIST('./data/mnistdata', download=True, train=False, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        return test_loader
    return (load_test_images,)


@app.cell
def _(Any, DataLoader, Dict, mlflow, np, torch):
    def evaluate_registered_model(model: mlflow.pyfunc.PyFuncModel, loader: DataLoader) -> Dict[str, Any]:
        correct_count, total_count = 0, 0
        for images, labels in loader:
            for i in range(len(labels)):
                img = images[i].view(1, 784)
                # Turn off gradients to speed up this part
                with torch.no_grad():
                    logps = model.predict(img.numpy())

                # Output of the network are log-probabilities, need to take exponential for probabilities
                ps = np.exp(logps)
                probab = list(ps[0])
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
    return (evaluate_registered_model,)


@app.cell
def _(evaluate_registered_model, load_test_images, model):
    test_loader = load_test_images(64)
    testing_metrics = evaluate_registered_model(model, test_loader)
    testing_metrics
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
