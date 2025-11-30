import torch
import numpy as np
import json
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device, cos_sim

# Load the model
model = SentenceTransformer("TechWolf/JobBERT-v2")

def encode_batch(jobbert_model, texts):
    features = jobbert_model.tokenize(texts)
    features = batch_to_device(features, jobbert_model.device)
    features["text_keys"] = ["anchor"]
    with torch.no_grad():
        out_features = jobbert_model.forward(features)
    return out_features["sentence_embedding"].cpu().numpy()

def encode(jobbert_model, texts, batch_size: int = 8):
    # Sort texts by length and keep track of original indices
    sorted_indices = np.argsort([len(text) for text in texts])
    sorted_texts = [texts[i] for i in sorted_indices]
    
    embeddings = []
    
    # Encode in batches
    for i in tqdm(range(0, len(sorted_texts), batch_size)):
        batch = sorted_texts[i:i+batch_size]
        embeddings.append(encode_batch(jobbert_model, batch))
    
    # Concatenate embeddings and reorder to original indices
    sorted_embeddings = np.concatenate(embeddings)
    original_order = np.argsort(sorted_indices)
    return sorted_embeddings[original_order]

# Example usage
job_titles = [
    'Software Engineer',
    'Senior Software Developer',
    'Product Manager',
    'Data Scientist',
    'Economist',
    'Financial Analyst',
    'Marketing Manager',
    'Operations Manager',
    'Mechanical Engineer',
    'Registered Nurse',
    'Elementary School Teacher',
    'Graphic Designer',
    'Sales Representative',
    'Accountant',
    'Human Resources Manager',
    'Project Manager',
    'Business Consultant',
    'Pharmacist',
    'Civil Engineer',
    'Research Scientist',
    'Chef',
    'Real Estate Agent',
    'Lawyer',
    'Architect',
    'Social Worker',
    'Electrician',
    'Physical Therapist',
    'Content Writer',
    'Supply Chain Manager',
    'Environmental Scientist'
]

# Get embeddings
embeddings = encode(model, job_titles)

# Save embeddings to JSON file
embeddings_dict = {
    job_title: embedding.tolist() 
    for job_title, embedding in zip(job_titles, embeddings)
}

with open('job_embeddings.json', 'w') as f:
    json.dump(embeddings_dict, f, indent=2)

print(f"Saved embeddings for {len(job_titles)} jobs to job_embeddings.json")

# Calculate cosine similarity matrix
similarities = cos_sim(embeddings, embeddings)
similarities_np = similarities.cpu().numpy() if hasattr(similarities, 'cpu') else similarities

# Format and save cosine similarity matrix
similarity_dict = {
    job_titles[i]: {
        job_titles[j]: float(similarities_np[i, j])
        for j in range(len(job_titles))
    }
    for i in range(len(job_titles))
}

with open('job_similarities.json', 'w') as f:
    json.dump(similarity_dict, f, indent=2)

print(f"\nSaved cosine similarity matrix to job_similarities.json")

# Print formatted similarity matrix
print("\nCosine Similarity Matrix:")
print("=" * 80)
print(f"{'Job Title':<30} | Most Similar Jobs (Top 5)")
print("-" * 80)

for i, job_title in enumerate(job_titles):
    # Get similarities for this job (excluding self-similarity)
    job_similarities = [(job_titles[j], float(similarities_np[i, j])) 
                        for j in range(len(job_titles)) if i != j]
    # Sort by similarity (descending)
    job_similarities.sort(key=lambda x: x[1], reverse=True)
    # Get top 5
    top_5 = job_similarities[:5]
    
    top_5_str = ", ".join([f"{job} ({sim:.3f})" for job, sim in top_5])
    print(f"{job_title:<30} | {top_5_str}")

print("=" * 80)
