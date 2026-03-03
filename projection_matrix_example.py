# Example: Using Projection Matrix for Text Embedding

import numpy as np
import torch.nn.functional as F

# Load projection matrix
W = np.load("projection_matrix_768_to_16_lerf.npy")  # [16, 768]

# Project text embeddings from 768-dim to 16-dim
text_embed_768 = open_clip_model.encode_text(["chair", "table", "lamp"])  # [num_classes, 768]
text_embed_16d = text_embed_768 @ W.T  # [num_classes, 16]

# Compute similarity with model predictions
pred_16d = model.predict(points)  # [N, 16] - from LitePT

# Cosine similarity
similarity = F.cosine_similarity(
    pred_16d.unsqueeze(1),  # [N, 1, 16]
    torch.from_numpy(text_embed_16d).unsqueeze(0)  # [1, num_classes, 16]
).squeeze(1)  # [N, num_classes]

# Get predicted class for each point
predicted_class = similarity.argmax(dim=1)  # [N]
