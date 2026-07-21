import torch


    


    

def alignment_loss(vectors, references, norm_vector=True, norm_ref=False, mixed_orientation=False, return_vector=False, eps=1e-8):
    if norm_vector:
        vectors_unit = vectors / (vectors.norm(dim=1, keepdim=True) + eps)
    else:
        vectors_unit = vectors
        
    if norm_ref:
        references_unit = references / (references.norm(dim=1, keepdim=True) + eps)
    else:
        references_unit = references

    if mixed_orientation:
        # Calculate cosine similarity and its negative
        cos_sim = torch.sum(vectors_unit * references_unit, dim=1)
        # Adjust for mixed orientation by flipping signs based on the sign of the cosine similarity
        cos_sim = torch.where(cos_sim < 0, -cos_sim, cos_sim)
    else:
        cos_sim = torch.sum(vectors_unit * references_unit, dim=1)

    if return_vector:
        return 1 - cos_sim
    else:
        return 1 - cos_sim.mean()
    

