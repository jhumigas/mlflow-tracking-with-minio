import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import json
    # Example: a single flattened MNIST image (replace with your actual data)
    single_image_data = np.random.rand(784).astype(np.float32).tolist()
    # For a single image inference
    payload_single = {"inputs": [single_image_data]}
    # For batch inference (multiple images)
    # batch_image_data = [np.random.rand(784).astype(np.float32).tolist() for _ in range(5)]
    # payload_batch = {"inputs": batch_image_data}
    return json, np, payload_single


@app.cell
def _(json, payload_single):
    import requests
    scoring_uri = "http://localhost:5015/invocations" # Adjust port if different
    headers = {"Content-Type": "application/json"}
    response = requests.post(scoring_uri, data=json.dumps(payload_single), headers=headers)
    if response.status_code == 200:
        predictions = response.json()
        print("Predictions:", predictions)
    else:
        print(f"Error: {response.status_code} - {response.text}")
    return headers, requests, scoring_uri


@app.cell
def _():
    ## Testing with actual data
    from torch.utils.data.dataloader import DataLoader
    from torchvision import datasets, transforms
    from typing import List, Dict, Any, Tuple
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
def _(load_test_images):
    test_loader = load_test_images(1)
    return (test_loader,)


@app.cell
def _(headers, json, np, payload_single, requests, scoring_uri, test_loader):
    sample_image, sample_label = next(iter(test_loader))
    sample_input = {"inputs": [sample_image.view(1, 784).numpy()]}
    sample_response = requests.post(scoring_uri, data=json.dumps(payload_single), headers=headers)
    sample_prediction = sample_response.json()
    sample_label = np.argmax(sample_prediction["predictions"][0])

    return (sample_label,)


@app.cell
def _(mo, sample_label):
    mo.md(f"Predicted Label {sample_label} -- True Label {sample_label}")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
