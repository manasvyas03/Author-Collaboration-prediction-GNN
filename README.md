# 📘 Graph Neural Network for Predicting Academic Collaborations and Citations

## Introduction
This project develops and evaluates a **Graph Neural Network (GNN)** model to predict the likelihood of future collaborations or citation interactions between academic authors.  
We move from raw data collection using **OpenAlex API** to a full **link prediction pipeline** using **PyTorch Geometric (PyG)**.

---

##  Data Curation and Graph Construction
- **Source:** OpenAlex API  
- **Filters Applied:**
  - Field: Computer Science (concept ID: `C41008148`)
  - Language: English
  - Date range: `2020-01-01` to `2024-12-31`

### Files Created
- **`nodes.csv`**
  - Fields: `author_id`, `display_name`, `institution`, `publication_count`, `citation_count`  
  - Built by aggregating stats for each author.

- **`edges.csv`**
  - Fields: `source_author_id`, `target_author_id`  
  - Constructed from co-authorship pairs in each paper.

### Visualization
- Co-authorship graph built using **NetworkX + Matplotlib**.  

---

## GNN Architecture Design & Training
### Models Used
- **Graph Convolutional Network (GCN)**
- **GraphSAGE**

### Input Features
- Publication Count (standardized)  
- Citation Count (standardized)  
- Institution (encoded with `LabelEncoder`)  

### Decoder
- Initially: Simple dot product  
- Improved: **2-layer MLP** for more expressive link prediction  

### Training Strategy
- **Negative Sampling**: Generates non-collaborating author pairs.  
- **Binary Labels**: 1 = collaboration, 0 = no collaboration  
- **Loss Function**: Binary Cross-Entropy  
- **Optimizer**: Adam (`lr=0.003`, `weight_decay=5e-5`)  
- **Regularization**: Dropout (0.3) + BatchNorm  

---

##  Evaluation Results

| Model     | Accuracy | F1 Score | AUC Score |
|-----------|----------|----------|-----------|
| **GCN**   | 0.9850   | 0.9851   | 0.9986    |
| **GraphSAGE** | 0.9903   | 0.9904   | 0.9995    |

Improved from **random 0.5 accuracy** to **~0.99** using:  
- Feature engineering (institution, scaling)  
- MLP decoder  
- Dropout + BatchNorm  
- Hyperparameter tuning  

---

## 🖼Visualizations
- **Co-authorship Graph**: Shows the real-world collaboration network.  
- **Author Embeddings (t-SNE)**: 2D projections reveal clusters of similar authors.  

---

## Challenges & Improvements

| Problem                | What I Did                  | How I Solved It |
|-------------------------|-----------------------------|-----------------|
| Weak features           | Added institutional context | Encoded with `LabelEncoder` |
| Varying scales          | Standardized features       | Used `StandardScaler` |
| Decoder too simple      | Switched to MLP             | 2-layer `nn.Sequential` |
| Model underfitting      | Better architecture         | 3 GCN layers, hidden=128 |
| Training instability    | Added regularization        | Dropout(0.3) + BatchNorm |

---

## Future Work
- Add **edge features** (same institution, co-citation score).  
- Extend to **temporal GNNs** for time-aware predictions.  
- Predict **top-k collaborations per author** (recommender style).  
- Incorporate **text embeddings** from paper abstracts.  

---

## Conclusion
This project demonstrates:  
- Real-world **data curation** with OpenAlex  
- Powerful **representation learning** using GCN & GraphSAGE  
- Model performance boosted from **0.5 → 0.99 accuracy**  

---

## Example Visuals
- Co-authorship network (NetworkX)  
- t-SNE clusters of author embeddings  

---

## 👨‍💻 Author
**Manas Vyas** – B.Tech CSE (Final Year)  
