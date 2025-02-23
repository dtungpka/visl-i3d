# ViSL

This project is designed to provide a flexible and scalable framework for training and evaluating multiple deep learning models, including an I3D model. The structure of the project allows for easy addition of new models and datasets, making it suitable for various research and development purposes.

## Project Structure

```
multi-models-project
├── src
│   ├── config
│   │   └── config.yaml
│   ├── datasets
│   │   ├── augmentation.py
│   │   ├── loader.py
│   │   └── __init__.py
│   ├── models
│   │   ├── i3d.py
│   │   ├── model_x.py
│   │   └── __init__.py
│   ├── main.py
│   └── utils.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/dtungpka/visl-i3d
cd visl-i3d
pip install -r requirements.txt
```

## Configuration

The configuration for model selection and hyperparameters can be found in `src/config/config.yaml`. Modify this file to choose different models and adjust settings as needed.

## Usage

To run the project, execute the `main.py` file:

```bash
python src/main.py
```

This will initialize the selected model and start the training or evaluation process based on the configuration specified.

## Models

Currently, the project includes the following models:

- **I3D Model**: Implemented in `src/models/i3d.py`, this model is designed for action recognition tasks.
- **Model X**: An additional model implemented in `src/models/model_x.py`, which can be used for experimentation and comparison.

## Datasets

The dataset loading and augmentation functionalities are implemented in the `src/datasets` folder. You can customize the data loading process in `loader.py` and apply various augmentation techniques in `augmentation.py`.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.