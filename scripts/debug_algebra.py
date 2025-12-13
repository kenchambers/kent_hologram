
import torch
import torchhd
from hologram.core.operations import Operations
from hologram.core.vector_space import VectorSpace

def check_algebra():
    print("Checking VSA Algebra with Current Implementation...")
    space = VectorSpace(dimensions=10000)
    
    # 1. Generate vectors as currently implemented (Gaussian)
    a = space.random_vector(42)
    b = space.random_vector(43)
    
    # 2. Bind
    bound = Operations.bind(a, b)
    
    # 3. Unbind
    recovered = Operations.unbind(bound, a)
    
    # 4. Check Similarity
    sim = torch.cosine_similarity(recovered, b, dim=0).item()
    print(f"Gaussian Vectors - unbind(bind(a,b), a) similarity: {sim:.4f}")
    
    # Check self-inverse property
    # For MAP, we expect a * a = 1 (or close to identity)
    # But Operations.inverse might be identity
    inv_a = Operations.inverse(a)
    identity_approx = Operations.bind(a, inv_a)
    # What does identity look like?
    print(f"Norm of a*inv(a): {torch.norm(identity_approx)}")
    print(f"Mean of a*inv(a): {torch.mean(identity_approx)}")
    
    print("-" * 30)
    
    # 5. Check with Bipolar Vectors
    print("Checking with Bipolar Vectors (+1/-1)...")
    a_bi = torch.sign(a)
    b_bi = torch.sign(b)
    
    # Bind (element-wise multiply is correct for bipolar)
    bound_bi = a_bi * b_bi
    
    # Unbind (multiply by inverse, which is self for bipolar)
    recovered_bi = bound_bi * a_bi # (a*b)*a = a*a*b = 1*b = b
    
    sim_bi = torch.cosine_similarity(recovered_bi, b_bi, dim=0).item()
    print(f"Bipolar Vectors - unbind(bind(a,b), a) similarity: {sim_bi:.4f}")

if __name__ == "__main__":
    check_algebra()
