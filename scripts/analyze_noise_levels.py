
import random
import torch
import time
from hologram.container import HologramContainer
from hologram.config.constants import DEFAULT_DIMENSIONS
from hologram.core.operations import Operations
from hologram.memory.fact_store import FactStore

class ForcedFactStore(FactStore):
    """FactStore that forces full-strength storage for capacity testing."""
    
    def add_fact(self, subject, predicate, obj, source=None, confidence=1.0):
        # Normalize
        subject_norm = self._normalize(subject)
        predicate_norm = self._normalize(predicate)
        
        # Standard encoding
        s_vec = self._codebook.encode(subject_norm)
        p_vec = self._codebook.encode(predicate_norm)
        o_vec = self._codebook.encode(obj)
        
        key = Operations.bind(s_vec, p_vec)
        
        # Force storage with full weight (learning_rate=1.0, threshold=0.0)
        # We manually call store_with_surprise with overrides
        self._memory.store_with_surprise(
            key, 
            o_vec, 
            learning_rate=1.0, 
            surprise_threshold=0.0
        )
        
        # Update metadata
        self._value_vocab.add(obj)
        self._value_vectors_cache[obj] = o_vec
        self._subject_vocab.add(subject)
        return True # Just return success

def analyze_noise_levels(max_facts=500, step=10, dimensions=DEFAULT_DIMENSIONS):
    """
    Analyzes the accumulation of holographic noise as facts are stored.
    """
    print(f"\n🔬 Starting Holographic Noise Analysis (Full Strength Storage)")
    print(f"   Dimensions: {dimensions}")
    print(f"   Target Facts: {max_facts}")
    print(f"   Step Size: {step}\n")

    container = HologramContainer(dimensions=dimensions)
    
    # Use our custom store
    fs = ForcedFactStore(container.vector_space, container.codebook)
    
    # Track metrics
    results = []
    ground_truth = []
    
    print(f"{'Facts':<8} | {'Signal':<15} | {'Noise':<15} | {'SNR':<8} | {'Accuracy':<8}")
    print("-" * 65)

    for i in range(1, max_facts + 1):
        # Generate random unique fact
        subject = f"Subj_{i}"
        predicate = f"Pred_{i}"
        obj = f"Obj_{i}"
        
        fs.add_fact(subject, predicate, obj, source="noise_test")
        ground_truth.append((subject, predicate, obj))
        
        # Periodically measure
        if i % step == 0:
            total_signal = 0.0
            total_noise = 0.0
            correct_count = 0
            sample_size = min(len(ground_truth), 20)
            sample_indices = random.sample(range(len(ground_truth)), sample_size)
            
            for idx in sample_indices:
                s, p, expected_o = ground_truth[idx]
                
                s_vec = fs._codebook.encode(fs._normalize(s))
                p_vec = fs._codebook.encode(fs._normalize(p))
                expected_o_vec = fs._codebook.encode(expected_o)
                
                key = Operations.bind(s_vec, p_vec)
                
                # 1. Measure RAW signal
                noisy_result = fs._memory.query(key)
                signal_strength = torch.cosine_similarity(noisy_result, expected_o_vec, dim=0).item()
                total_signal += signal_strength
                
                # 2. Check accuracy via resonance
                # Use a fixed set of distractors to make noise metric consistent
                # 100 random words from vocab + correct word
                vocab = list(fs._value_vocab)
                if len(vocab) > 100:
                    distractors = random.sample(vocab, 100)
                else:
                    distractors = vocab
                
                if expected_o not in distractors:
                    distractors.append(expected_o)
                
                # Build candidate tensor
                candidate_vecs = []
                for d in distractors:
                    if d in fs._value_vectors_cache:
                        candidate_vecs.append(fs._value_vectors_cache[d])
                    else:
                        candidate_vecs.append(fs._codebook.encode(d))
                candidates_tensor = torch.stack(candidate_vecs)
                
                # Run resonance
                similarities = fs._memory.resonance(key, candidates_tensor)
                best_idx = torch.argmax(similarities).item()
                retrieved_val = distractors[best_idx]
                
                if retrieved_val == expected_o:
                    correct_count += 1
                
                # 3. Measure Noise Floor
                sim_values = similarities.tolist()
                try:
                    correct_idx = distractors.index(expected_o)
                    sim_values.pop(correct_idx)
                except ValueError:
                    pass
                
                if sim_values:
                    # Use MAX noise (worst case interference) as noise metric
                    # or AVG noise. SNR usually uses Variance, but here 
                    # we care about "highest distractor".
                    avg_noise_for_fact = max(sim_values) # Max noise determines misclassification
                else:
                    avg_noise_for_fact = 0.0
                total_noise += avg_noise_for_fact

            avg_signal = total_signal / sample_size
            avg_noise = total_noise / sample_size
            avg_snr = avg_signal / (avg_noise + 1e-9)
            accuracy = correct_count / sample_size
            
            results.append({
                "facts": i,
                "signal": avg_signal,
                "noise": avg_noise,
                "snr": avg_snr,
                "accuracy": accuracy
            })
            
            print(f"{i:<8} | {avg_signal:<15.4f} | {avg_noise:<15.4f} | {avg_snr:<8.2f} | {accuracy:<8.1%}")
            
            # Stop if system fails completely
            if accuracy < 0.1:
                print("\n⚠️  System saturated.")
                break

    return results

if __name__ == "__main__":
    analyze_noise_levels(max_facts=500, step=25)
