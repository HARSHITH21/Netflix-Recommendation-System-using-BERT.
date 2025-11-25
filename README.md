
# Netflix Recommendation System Using BERT and MLP

## Project Overview

This project is focused on building a personalized recommendation system for Netflix using **BERT** (Bidirectional Encoder Representations from Transformers) and **MLP** (Multilayer Perceptron). The system leverages natural language processing (NLP) techniques from BERT to extract content features from movie/TV show descriptions, and the MLP model is used to learn user preferences and make personalized recommendations.

### Authors:
- Harshith Gade (hg355)
- Sai Nikhil Dunuka (sd2279)
- Gudisinti Durgaprasad(gd299)

---

## Table of Contents:
1. [Introduction](#introduction)
2. [Dataset Description](#dataset-description)
3. [Project Objectives](#project-objectives)
4. [Model Architecture](#model-architecture)
5. [Code Overview](#code-overview)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [How to Run the Code](#how-to-run-the-code)

---

## Introduction

Netflix Inc. is one of the largest video streaming platforms in the world. To enhance user experience, Netflix recommends content based on user preferences and past interactions. This project aims to use advanced machine learning and NLP techniques to build a robust recommendation system using BERT for text-based feature extraction and MLP for predicting user preferences.

---

## Dataset Description

The dataset used contains a variety of metadata related to Netflix's movies and TV shows. The dataset includes attributes such as:
- **show_id**: Unique identifier for each movie or TV show.
- **type**: Categorization of the content (Movie or TV Show).
- **title**: Title of the movie or TV show.
- **director**: Director of the movie or TV show.
- **cast**: List of main actors.
- **country**: Country where the movie or show was produced.
- **release_year**: Year of release.
- **rating**: Rating assigned to the movie or TV show.
- **duration**: Duration of the movie (in minutes) or number of seasons (for TV shows).
- **description**: Brief summary of the movie or TV show.

*Dataset source*: [Kaggle Netflix Dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows)

---

## Project Objectives

1. **Feature Extraction**: Utilize BERT to extract relevant features from text data such as descriptions and reviews.
2. **User Preference Modeling**: Use MLP to learn user preferences from past interactions and generate personalized recommendations.
3. **Recommendation System**: Combine the features extracted by BERT with the user profile modeled by MLP to predict which content a user would prefer.

---

## Model Architecture

1. **BERT Feature Extraction**: BERT is used to process text data from the Netflix dataset, extracting semantic features that describe the content of each movie or TV show.
2. **MLP User Model**: An MLP is trained on user interaction data (e.g., ratings, viewing history) to learn complex relationships between user preferences and content features.
3. **Recommendation Generation**: The features from BERT are combined with the MLP-generated user profiles to predict which content a user would enjoy the most.

---

## Code Overview

- **Data Preprocessing**: 
  - Cleaning, tokenization, and encoding of categorical data.
  - Preprocessing steps include handling missing values and transforming text data for BERT input.
  
- **BERT Feature Extraction**:
  - Pre-trained BERT model is fine-tuned to extract feature vectors from the content descriptions.

- **MLP User Preference Model**:
  - An MLP model is built and trained on user data to learn user preferences based on the extracted features from BERT.

- **Recommendation System**:
  - The system combines the outputs of BERT and MLP to generate personalized recommendations for users.

---

## Results

The model achieves accurate recommendations by understanding both the content (through BERT) and user preferences (through MLP). Various experiments, such as comparison between MLP and clustering, demonstrate that MLP yields higher accuracy for recommendations.

---

## Conclusion

Using BERT and MLP in combination offers a powerful and scalable recommendation system for Netflix. By leveraging the textual understanding of BERT and the learning capability of MLP, this system provides personalized, relevant, and engaging recommendations for users.

---

## How to Run the Code

1. **Dependencies**: Install the following Python packages:
   - `transformers`
   - `torch`
   - `pandas`
   - `numpy`
   - `sklearn`

2. **Dataset**: Download the dataset from the [Kaggle Netflix Dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows).

3. **Run the Notebook**: 
   - Open `Netflix Recommendation System using BERT.ipynb` in Jupyter Notebook or Google Colab.
   - Follow the steps outlined in the notebook to preprocess the data, fine-tune BERT, train the MLP model, and generate recommendations.

---

### Link to Colab Notebook
[Netflix Recommendation System Code](https://colab.research.google.com/drive/1QJgWvOXCXQnIq_2YvkBtMy6SywtSbwci)

