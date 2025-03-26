# Fake Currency Detection

## Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Data Exploration & Preprocessing](#data-exploration--preprocessing)
- [Model Implementation & Evaluation](#model-implementation--evaluation)
- [Results and Insights](#results-and-insights)
- [Challenges](#challenges)
- [Future Improvements](#future-improvements)
- [Conclusion](#conclusion)
- [License](#license)

## Introduction
Counterfeit currency poses significant financial losses and security risks. As counterfeit techniques advance, traditional detection methods are insufficient. This project aims to develop an automated system using machine learning to accurately identify fake currency notes, thereby enhancing security and reducing losses.

## Problem Statement
Counterfeit currency in India leads to substantial financial losses and security threats. Manual detection is inefficient, and counterfeiters utilize advanced replication techniques. This project seeks to create an automated system that employs image processing and computer vision to verify the authenticity of Indian currency notes.

## Objectives
- **Automated Detection**: Identify fake currency notes using image processing techniques.
- **High Accuracy**: Ensure reliable results with minimal errors.
- **Fast Processing**: Provide quick results for real-time use.
- **User-Friendly Interface**: Design an intuitive interface for easy operation.

## Dataset
The dataset comprises images of both real and counterfeit currency notes, structured as follows:
- **₹500 Currency Notes**:
  - Images of real ₹500 notes.
  - Images of fake ₹500 notes.
  - Multiple images of each security feature.
  
- **₹2000 Currency Notes**:
  - Similar structure as the ₹500 dataset.

### Key Security Features for ₹500 Notes
1. ₹500 in Devanagari and English script (2 features).
2. Ashoka Pillar Emblem (1 feature).
3. RBI symbols in Hindi and English (2 features).
4. "500 Rupees" written in Hindi (1 feature).
5. RBI logo (1 feature).
6. Bleed lines on the left and right sides (2 features).
7. Number panel (1 feature).

## Data Exploration & Preprocessing
### Preprocessing Steps
1. **Image Acquisition**: Capturing images using a digital camera or scanner.
2. **Resizing**: Standardizing image dimensions for consistency.
3. **Grayscale Conversion**: Reducing computational complexity.
4. **Noise Reduction**: Applying Gaussian blur to remove noise.
5. **Feature Extraction**: Utilizing the ORB algorithm to identify key currency features.

## Model Implementation & Evaluation
### Machine Learning Algorithms Used
- **Feature Detection and Matching (ORB)**: Collects average and max SSIM scores for each feature.
- **Bleed Line Detection**: Analyzes bleed lines for authenticity.
- **Number Panel Authentication**: Validates the number of characters in the currency note.

### Performance Metrics
- **Testing Real Notes**: 
  - Total tested: 19 (9 ₹2000, 10 ₹500)
  - Correct results: 15
  - Accuracy: 79%
  
- **Testing Fake Notes**:
  - Total tested: 12 (6 each from ₹2000 and ₹500)
  - Correct results: 10
  - Accuracy: 83%

## Results and Insights
### Key Findings
- The system effectively detects fake currency based on predefined security features.
- High SSIM scores indicate strong matches between input features and genuine templates.
- ORB-based detection successfully identifies crucial security features.

### Visual Representations
- The system features a user-friendly graphical interface built using Streamlit, allowing users to upload currency images for analysis.
- Real-time classification outputs indicate whether currency is genuine or counterfeit.
  
### Sample Images
![Real Currency](https://github.com/Aditi31kapil/Fake-Currency-Detection/blob/main/Dataset/500_dataset/500_s1.jpg)
![Fake Currency](https://github.com/Aditi31kapil/Fake-Currency-Detection/blob/main/Fake%20Notes/500/500_f1.jpg)

## Challenges
- Difficulty in detecting faded or worn-out currency notes.
- Variations in image lighting affecting feature extraction.
- Processing time could be optimized further.

## Future Improvements
- Implementing deep learning models for enhanced accuracy.
- Expanding the dataset to include more denominations.
- Integrating smartphone-based real-time detection.

## Conclusion
The proposed system provides a reliable and automated approach to detecting fake currency. With future advancements, it can be widely adopted for real-time applications in banks, businesses, and public spaces, significantly reducing counterfeit-related financial fraud.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
