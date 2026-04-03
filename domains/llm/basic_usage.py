"""
Basic Usage Example: Real-Time Hallucination Detection with Kappa-LLM

This example demonstrates how to use Kappa-LLM for real-time hallucination
detection during LLM generation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kappa_llm import KappaDetector, ModelCalibration

def main():
    # ============================================================================
    # 1. Load Model and Tokenizer
    # ============================================================================
    print("Loading model...")
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # ============================================================================
    # 2. Initialize Kappa Detector
    # ============================================================================
    print("Initializing Kappa detector...")
    
    # Option A: Use pre-calibrated thresholds
    calibration = ModelCalibration.get_calibration("phi-3")
    
    detector = KappaDetector(
        model=model,
        window_size=calibration.window_size,          # 10 tokens
        check_interval=calibration.check_interval,    # Check every 5 tokens
        tau_warn=calibration.tau_warn,                # Warning threshold
        tau_abort=calibration.tau_abort,              # Abort threshold (0.74)
        persistence_k=calibration.persistence_k,      # K=2 consecutive windows
        max_retries=2,                                 # Retry with conservative params
        verbose=True                                   # Show monitoring info
    )
    
    # ============================================================================
    # 3. Generate with Real-Time Monitoring
    # ============================================================================
    
    # Test prompts (mix of safe and potentially problematic)
    test_prompts = [
        # Safe factual query
        "What is the capital of France?",
        
        # More complex but factual
        "Explain the process of photosynthesis in plants.",
        
        # Potentially hallucination-prone
        "Tell me about the discovery of the planet Vulcan between Mercury and the Sun.",
        
        # Another potentially problematic one
        "What are the main features of the Python 5.0 release?"
    ]
    
    print("\n" + "="*80)
    print("Running Kappa-LLM Real-Time Detection")
    print("="*80 + "\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'─'*80}")
        print(f"Test {i}/{len(test_prompts)}")
        print(f"{'─'*80}")
        print(f"📝 Prompt: {prompt}")
        print()
        
        # Reset detector state
        detector.reset()
        
        # Encode prompt
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        
        # Generate with monitoring
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                stopping_criteria=[detector],
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Display results
        print("📊 Detection Results:")
        print(f"   Status: {'⚠️  PRUNED' if detector.was_pruned else '✅ SAFE'}")
        
        if detector.was_pruned:
            print(f"   Prune Position: Token {detector.prune_position}")
            print(f"   Final Risk Score: {detector.final_risk:.3f}")
            print(f"   Reason: {detector.prune_reason}")
            print(f"   Retries: {detector.retry_count}")
        else:
            print(f"   Max Risk Score: {detector.max_risk:.3f}")
            print(f"   Total Tokens: {len(outputs[0]) - len(inputs[0])}")
        
        print("\n🗨️  Generated Text:")
        print(f"   {generated_text}")
        print()
    
    # ============================================================================
    # 4. Inspect Detection Statistics
    # ============================================================================
    print("\n" + "="*80)
    print("Detection Statistics Summary")
    print("="*80 + "\n")
    
    stats = detector.get_statistics()
    print(f"Total Prompts Processed: {len(test_prompts)}")
    print(f"Safe Generations: {stats['safe_count']}")
    print(f"Pruned Generations: {stats['pruned_count']}")
    print(f"Average Risk (Safe): {stats['avg_risk_safe']:.3f}")
    print(f"Average Risk (Pruned): {stats['avg_risk_pruned']:.3f}")
    
    # ============================================================================
    # 5. Advanced: Access Raw Observables
    # ============================================================================
    print("\n" + "="*80)
    print("Observable Analysis (Last Generation)")
    print("="*80 + "\n")
    
    if detector.observable_history:
        last_obs = detector.observable_history[-1]
        print("Kappa Observables at last checkpoint:")
        print(f"  Ω (Entropy):     {last_obs['omega']:.3f}")
        print(f"  Φ (Persistence): {last_obs['phi']:.3f}")
        print(f"  η (Rigidity):    {last_obs['eta']:.3f}")
        print(f"  Ξ (Diversity):   {last_obs['xi']:.3f}")
        print(f"  Δ (Divergence):  {last_obs['delta']:.3f}")
        print(f"\n  Kappa Score: {last_obs['kappa_score']:.3f}")
        print(f"  Risk Score:  {last_obs['risk_score']:.3f}")

if __name__ == "__main__":
    main()
