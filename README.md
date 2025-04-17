# Job Description and Resume Matching System

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Bootstrap](https://img.shields.io/badge/bootstrap-%23563D7C.svg?style=for-the-badge&logo=bootstrap&logoColor=white)

A machine learning-powered system that automatically matches job descriptions with candidate resumes, streamlining the recruitment process by identifying the most suitable candidates.

## Features

- **Multi-file Support**: Process PDF, DOCX, and TXT resume formats
- **Dual Algorithm Approach**:
  - Cosine Similarity for direct text matching
  - Random Forest classifier for learned pattern recognition
- **Interactive Visualizations**:
  - Scatter plots of similarity scores
  - Algorithm comparison charts
- **User-Friendly Interface**:
  - Clean, responsive Bootstrap UI
  - Simple drag-and-drop file upload
- **Ranked Results**: Top 5 candidates displayed with matching scores

## How It Works

1. **Text Extraction**: System extracts text from resumes in various formats
2. **Vectorization**: Converts text to numerical features using TF-IDF
3. **Similarity Calculation**:
   - Computes cosine similarity between job description and resumes
   - Uses Random Forest to predict match probabilities
4. **Result Visualization**: Generates interactive charts comparing results

## Technologies Used

- **Python:** Backend development using Python programming language.
- **Flask:** Web framework for building the backend server and handling HTTP requests.
- **Bootstrap:** Frontend design and layout using Bootstrap for responsive and user-friendly UI.
- **Machine Learning Libraries:** Libraries such as scikit-learn for implementing machine learning algorithms for text similarity matching.
- **HTML/CSS:** Frontend markup and styling for web pages.



