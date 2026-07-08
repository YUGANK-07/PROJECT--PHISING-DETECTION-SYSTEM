# GUARDIAN: An Intelligent Phishing Detection System Using Machine Learning

**A Report submitted**
In partial fulfilment for the Degree of
**B. Tech**
In
**COMPUTER SCIENCE & ENGINEERING**

**By**
- Ayush Dixit (UU2201010115)
- Nikhil Pranjal Juyal (UU2201010222)
- Vibhanshu Panchwan (UU2201010346)
- Vishal Negi (UU2201010348)
- Yugank Pant (UU220101358)

**Pursued in**
Department of Computer Science & Engineering
Uttaranchal Institute of Technology
Uttaranchal University, Dehradun
May 2026

---

## CERTIFICATE

This is to certify that the project report entitled “**GUARDIAN: An Intelligent Phishing Detection System Using Machine Learning**” submitted by Ayush Dixit, Nikhil Pranjal Juyal, Vibhanshu Panchwan, Vishal Negi, and Yugank Pant, students of B. Tech (Computer Science & Engineering), Uttaranchal University, in partial fulfillment for the award of the degree of Bachelor of Technology, is a bona fide record of the project work carried out by them under my supervision. 

It is further certified that this work has not been submitted, either in part or in full, to any other Institution or University for the award of any degree or diploma.

**Asst. Prof. Ms. Shreya Suman**  
Uttaranchal Institute of Technology  
Dehradun  
May 2026  

*(Counter Signature of Head of Department with Seal)*

---

## DECLARATION

I declare that this project report titled “**GUARDIAN: An Intelligent Phishing Detection System Using Machine Learning**” submitted in partial fulfillment of the degree of B. Tech in Computer Science & Engineering is a record of original work carried out by me under the supervision of Asst. Prof. Ms. Shreya Suman, and has not formed the basis for the award of any other degree or diploma, in this or any other Institution or University. In keeping with the ethical practice in reporting scientific information, due acknowledgements have been made wherever the findings of others have been cited.

- Ayush Dixit (UU2201010115)
- Nikhil Pranjal Juyal (UU2201010222)
- Vibhanshu Panchwan (UU2201010346)
- Vishal Negi (UU2201010348)
- Yugank Pant (UU220101358)

Date: ___________

---

## ACKNOWLEDGEMENT

We would like to express our sincere gratitude to the Director of Uttaranchal Institute of Technology, Uttaranchal University, Prof. Dr. Sumit Chaudhary, and the Head of the Department of Computer Science & Engineering, Dr. Madhu Kirola, for their constant support, encouragement, and for providing us with the necessary infrastructure and academic environment to successfully complete our project titled “GUARDIAN: An Intelligent Phishing Detection System Using Machine Learning.”

We are also deeply thankful to our respected supervisor, Ms. Shreya Suman, for her invaluable guidance, continuous support, and insightful suggestions throughout the development of this project. Her technical expertise, academic rigor, and mentorship played a vital role in shaping the direction and success of our work.

Furthermore, we extend our sincere thanks to all the faculty members of the Department of Computer Science & Engineering for their support, cooperation, and for sharing their knowledge during the course of this project. Finally, we would like to express our heartfelt gratitude to our family members and friends for their constant encouragement, motivation, and support throughout the completion of this project.

- Ayush Dixit
- Nikhil Pranjal Juyal
- Vibhanshu Panchwan
- Vishal Negi
- Yugank Pant

---

## ABSTRACT

Phishing attacks have emerged as one of the most significant cybersecurity threats, targeting users through deceptive websites, visually spoofed interfaces, and malicious URLs to steal sensitive personal and financial information. Traditional detection techniques often rely on static blacklists or simple heuristic rule-based systems, which consistently fail to identify newly generated (zero-day) or highly sophisticated phishing attempts. To critically address these systemic limitations, this project presents “GUARDIAN: An Intelligent Phishing Detection System Using Machine Learning,” engineered to provide highly accurate, real-time detection of complex phishing threats.

The proposed system radically departs from single-dimensional analysis by utilizing a massive, highly curated dataset to extract a comprehensive 102-dimensional feature vector. This vector encapsulates deep lexical analysis of URLs, domain metadata extraction (WHOIS, DNS records), website structural components (HTML, obfuscated JavaScript), and advanced Natural Language Processing (NLP) semantics. A dataset containing 250,000 URLs (both legitimate and phishing) is utilized to train and dynamically evaluate multiple machine learning algorithms. 

Instead of relying on a single classifier, GUARDIAN employs an advanced Stacking Ensemble architecture that intelligently combines the strengths of Random Forest and XGBoost base estimators, yielding a state-of-the-art predictive accuracy and an F1-Score of 1.0000 on a held-out test set of 37,500 URLs. Furthermore, to enhance transparency, trust, and actionable intelligence, the system incorporates Explainable AI (XAI) using SHAP (SHapley Additive exPlanations), dynamically generating human-readable justifications for every prediction.

The integrated solution operates via a high-performance FastAPI backend, supported by an asynchronous Redis caching layer and an interactive, visually sophisticated web interface. The final results definitively indicate that GUARDIAN provides a uniquely reliable, highly scalable, and structurally transparent solution for real-time phishing detection. This project substantially contributes to the cybersecurity landscape, providing an enterprise-grade conceptual model to proactively safeguard users against rapidly evolving phishing methodologies.

---

## TABLE OF CONTENTS
*(Will be automatically generated by Word)*

## LIST OF FIGURES
*(Will be automatically generated by Word)*

## LIST OF TABLES
*(Will be automatically generated by Word)*

## ABBREVIATIONS
- **API**: Application Programming Interface
- **CNN**: Convolutional Neural Network
- **DFD**: Data Flow Diagram
- **DNS**: Domain Name System
- **FNR**: False Negative Rate
- **HTML**: HyperText Markup Language
- **JS**: JavaScript
- **JWT**: JSON Web Token
- **ML**: Machine Learning
- **MLP**: Multilayer Perceptron
- **NLP**: Natural Language Processing
- **ROC-AUC**: Receiver Operating Characteristic - Area Under Curve
- **SHAP**: SHapley Additive exPlanations
- **TLD**: Top-Level Domain
- **URL**: Uniform Resource Locator
- **WHOIS**: Query and response protocol for querying registered internet resources
