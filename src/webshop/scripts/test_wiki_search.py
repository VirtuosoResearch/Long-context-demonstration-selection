#!/usr/bin/env python3
"""
Quick test script to debug Wikipedia search issue
"""
import sys
sys.path.insert(0, '.')

from envs.wikienv import WikiEnv

def test_search():
    env = WikiEnv()
    env.reset()
    
    # Test a few searches
    test_entities = [
        "Irene Jacob",
        "Stuart Bird", 
        "Python programming",
        "Albert Einstein"
    ]
    
    for entity in test_entities:
        print(f"\n{'='*60}")
        print(f"Testing search for: {entity}")
        print(f"{'='*60}")
        
        obs, reward, done, info = env.step(f"search[{entity}]")
        
        print(f"\nObservation (length={len(obs)}):")
        print(f"{obs}")
        print(f"\nReward: {reward}, Done: {done}")
        print(f"Info: {info}")
        
        # Reset for next test
        env.reset()

if __name__ == "__main__":
    test_search()
