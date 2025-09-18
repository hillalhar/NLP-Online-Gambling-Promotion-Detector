# NLP Detector for Online Gambling Promotion     
     
This project focuses on developing a deep learning model to detect and classify Indonesian text comments that promote online gambling. Using Natural Language Processing (NLP) techniques, the system preprocesses text data and employs various Recurrent Neural Network (RNN) architectures to achieve high classification accuracy.     
     
## 📌 Project Overview
The proliferation of online gambling promotions in social media comments and forums is a growing concern. This project aims to automatically identify such promotional content from Indonesian text. A binary classification approach is adopted, where comments are labeled as either "Gambling Promotion" (1) or "Not a Gambling Promotion" (0). Three different deep learning models (Bidirectional LSTM, GRU, and SimpleRNN) are trained and evaluated to find the most effective architecture for this task.     
     
## 💾 Dataset
The dataset used for this project is `dataset_judol.csv`, which contains two columns:
- `comment`: The raw text of the comment in Indonesian.
- `label`: A binary label where `1` indicates a gambling promotion and `0` indicates a non-promotional comment.

The original dataset was imbalanced. To address this, **Random Undersampling** was applied to the majority class (non-promotional comments) to create a balanced dataset for training, ensuring the model does not become biased towards one class.
     
Dataset Source     
[Dataset Online gambling 1](https://www.kaggle.com/datasets/fahruu/komentar-judi-online)     
[Dataset Online gambling 2](https://www.kaggle.com/datasets/yaemico/judionline)

## ⚙️ Methodology

The project follows a standard NLP pipeline, from data preparation to model evaluation.

### 1. Data Balancing
The initial dataset contained an unequal distribution of classes. Random Undersampling was used to balance the dataset by reducing the number of samples in the majority class to match the number of samples in the minority class. This resulted in an equal number of samples for both classes (7,382 each).

### 2. Text Preprocessing
To clean and standardize the text data, the following preprocessing steps were performed:
- **Case Folding**: All text was converted to lowercase to ensure uniformity.
- **Noise Removal**: Unnecessary characters, such as punctuation, emojis, and special symbols, were removed using regular expressions.
- **Word Normalization**: A custom dictionary was used to convert Indonesian slang and abbreviations into their standard forms (e.g., `ga` → `tidak`, `depo` → `deposit`).
- **Stopword Removal**: Common Indonesian words that do not contribute significant meaning (e.g., `di`, `dan`, `yang`) were removed from the text.

### 3. Text Vectorization
After preprocessing, the cleaned text was converted into numerical vectors that the deep learning models can process:
- **Tokenization**: A Keras `Tokenizer` was used to map each unique word to an integer. The vocabulary was limited to the top 10,000 most frequent words.
- **Padding**: All text sequences were padded to a uniform length of 128. Shorter sequences were padded with zeros at the end (`post-padding`).

### 4. Model Development
Three different Bidirectional RNN architectures were implemented and compared:
1.  **Bidirectional LSTM (Long Short-Term Memory)**
2.  **Bidirectional GRU (Gated Recurrent Unit)**
3.  **Bidirectional SimpleRNN**

All models were trained using the `Adam` optimizer and `sparse_categorical_crossentropy` loss function. An `EarlyStopping` callback was used to monitor validation loss and prevent overfitting by stopping the training when performance ceased to improve.


