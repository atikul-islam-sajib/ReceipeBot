# RecipeBot using Langchain, VectorDB

RecipeBot is an advanced conversational assistant designed to help users find recipes based on the ingredients they have on hand. Utilizing a collection of PDF files containing diverse recipes, RecipeBot processes these documents, stores the information in a database, and efficiently retrieves relevant recipes in response to user queries.

## Features

- **Document Processing**: Extracts and processes recipe information from a wide range of PDF files.
- **Efficient Storage**: Utilizes a robust database to store and retrieve recipe data swiftly.
- **Conversational Interface**: Supports natural language queries to provide personalized recipe recommendations.
- **Data Version Control**: Integrates with DVC to manage data files and model versions effectively.

## Project Structure

```
.
├── Dockerfile
├── LICENSE
├── config.yml
├── data
│   ├── PDFs
│   │   └── [PDF files]
│   └── processed
│       ├── __init__.py
│       └── documents.pkl
├── db
│   ├── chroma.sqlite3
│   └── e78b2af2-d6d5-4ed7-9c2a-4cd253e93fc2
│       ├── data_level0.bin
│       ├── header.bin
│       ├── index_metadata.pickle
│       ├── length.bin
│       └── link_lists.bin
├── dvc.lock
├── dvc.yaml
├── model
│   └── __init__.py
├── mypy.ini
├── requirements.txt
├── research
│   ├── files
│   │   └── __init__.py
│   └── notebooks
│       └── ModelPrototype.ipynb
├── setup.py
└── src
    ├── __init__.py
    ├── helper.py
    ├── receipe_bot.py
    └── utils.py
```

## Installation

To set up RecipeBot on your local machine, follow these instructions:

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/atikul-islam-sajib/ReceipeBot.git
   cd ReceipeBot
   ```

2. **Set Up a Virtual Environment**:
   ```sh
   python3 -m venv langchain-env
   source langchain-env/bin/activate
   ```

3. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

4. **Install DVC (Data Version Control)**:
   ```sh
   pip install dvc
   ```

## Usage

### Processing PDFs and Training

1. **Add Your PDF Files**:
   Place your recipe PDF files in the `data/PDFs` directory.

2. **Run the DVC Pipeline**:
   Execute the following command to process the PDFs and train the model:
   ```sh
   dvc repro
   ```

### Interacting with RecipeBot

1. **Start the Train the model**:
   ```sh
   python src/receipe_bot.py --train
   ```

2. **Start the Chat Interface**:
   ```sh
   python src/receipe_bot.py --chat
   ```

3. **Enter Your Queries**:
   Interact with RecipeBot by typing queries such as:
   ```sh
   Query: I have meat and tomatoes. Can you suggest at least two recipes?
   ```

## Configuration

RecipeBot's configuration settings are stored in the `config.yml` file. This file includes parameters for the database, model, and other operational settings. Customize these parameters to fine-tune RecipeBot’s behavior according to your requirements.

## Contributing

We welcome contributions from the community! To contribute to RecipeBot, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Implement your changes and commit them to your branch.
4. Push your branch to your forked repository.
5. Open a pull request to the main repository with a detailed description of your changes.

## License

This project is licensed under the MIT License. For more details, please refer to the [LICENSE](LICENSE) file.