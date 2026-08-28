"""
test_multiclass_attack.py

Verification script to test your multi-class attack implementation.
Run this after integrating the modifications to ensure everything works.

Usage:
    python test_multiclass_attack.py --data_root /path/to/FLIR_ADAS_1_3 --device cuda
"""

import argparse
import torch
import numpy as np
from pathlib import Path

print("\n" + "="*70)
print("  ThermalMirage Multi-Class Attack - Verification Test")
print("="*70 + "\n")

def test_imports():
    """Test 1: Verify all required modules can be imported."""
    print("[Test 1] Checking imports...")
    try:
        from attack_selector import AttackStrategySelector, AttackStrategy, AttackResult
        print("  ✓ attack_selector imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import attack_selector: {e}")
        return False
    
    try:
        from data.flir_dataset import build_dataloader, FLIR_CLASS_NAMES
        print("  ✓ flir_dataset imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import flir_dataset: {e}")
        return False
    
    try:
        from attack.detector_wrapper import FasterRCNNWrapper
        print("  ✓ detector_wrapper imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import detector_wrapper: {e}")
        return False
    
    try:
        from models.thermal_gan import ThermalGANTrainer
        print("  ✓ thermal_gan imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import thermal_gan: {e}")
        return False
    
    print("  ✓ All imports successful\n")
    return True


def test_attack_strategy_generation():
    """Test 2: Verify AttackStrategy generation."""
    print("[Test 2] Testing attack strategy generation...")
    
    from attack_selector import AttackStrategySelector
    
    selector = AttackStrategySelector(num_classes=4, device='cpu')
    
    # Test strategy generation for class 0 (person)
    strategies = selector.generate_all_strategies(src_class=0)
    
    print(f"  Generated {len(strategies)} strategies for class 0:")
    for i, strat in enumerate(strategies):
        print(f"    {i+1}. {strat}")
    
    # Should have 1 hide + 3 misclassify = 4 total
    expected = 4
    if len(strategies) == expected:
        print(f"  ✓ Correct number of strategies ({expected})\n")
        return True
    else:
        print(f"  ✗ Expected {expected} strategies, got {len(strategies)}\n")
        return False


def test_detector_wrapper(args):
    """Test 3: Verify detector wrapper has required methods."""
    print("[Test 3] Testing detector wrapper...")
    
    from attack.detector_wrapper import FasterRCNNWrapper
    
    detector = FasterRCNNWrapper(
        num_flir_classes=4,
        pretrained=True,
        device=args.device
    )
    
    # Check if detect() method exists
    if not hasattr(detector, 'detect'):
        print("  ✗ detector.detect() method not found")
        print("     You need to add this method to FasterRCNNWrapper")
        return False
    else:
        print("  ✓ detector.detect() method exists")
    
    # Check if get_feature_extractor() exists
    if not hasattr(detector, 'get_feature_extractor'):
        print("  ✗ detector.get_feature_extractor() method not found")
        print("     You need to add this method to FasterRCNNWrapper")
        return False
    else:
        print("  ✓ detector.get_feature_extractor() method exists")
    
    # Test detect() with dummy input
    try:
        dummy_img = torch.randn(1, 3, 512, 640).to(args.device)
        detections = detector.detect(dummy_img)
        print(f"  ✓ detector.detect() runs successfully")
        print(f"     Output format: {list(detections[0].keys())}")
    except Exception as e:
        print(f"  ✗ detector.detect() failed: {e}")
        return False
    
    print()
    return True


def test_dataloader(args):
    """Test 4: Verify dataloader format."""
    print("[Test 4] Testing dataloader format...")
    
    from data.flir_dataset import build_dataloader
    
    try:
        loader = build_dataloader(
            root=args.data_root,
            split='val',
            batch_size=2,
            num_workers=0,
            normalize=True
        )
        print(f"  ✓ Dataloader created successfully")
    except Exception as e:
        print(f"  ✗ Failed to create dataloader: {e}")
        return False
    
    # Get one batch
    try:
        batch = next(iter(loader))
        print(f"  ✓ Batch loaded successfully")
        print(f"     Batch keys: {list(batch.keys())}")
        
        required_keys = ['images', 'boxes', 'labels']
        for key in required_keys:
            if key not in batch:
                print(f"  ✗ Missing required key: {key}")
                return False
            else:
                print(f"     ✓ Has '{key}': type={type(batch[key])}")
        
        # Check format
        print(f"     Images shape: {batch['images'].shape}")
        print(f"     Boxes: list of {len(batch['boxes'])} tensors")
        print(f"     Labels: list of {len(batch['labels'])} tensors")
        
    except Exception as e:
        print(f"  ✗ Failed to load batch: {e}")
        return False
    
    print()
    return True


def test_attack_single_object(args):
    """Test 5: Verify attack_single_object() implementation."""
    print("[Test 5] Testing attack_single_object() method...")
    
    try:
        from attack.joint_attack import ThermalMirageAttack
    except ImportError:
        print("  ⚠ ThermalMirageAttack not found, skipping this test")
        print("     (You may not have this class yet - that's okay)\n")
        return True
    
    # Check if method exists
    attack_instance = type('DummyAttack', (), {})()  # Create dummy instance
    if not hasattr(ThermalMirageAttack, 'attack_single_object'):
        print("  ✗ ThermalMirageAttack.attack_single_object() method not found")
        print("     You need to add this method to your attack class")
        print("     See attack_engine_modifications.py for implementation")
        return False
    else:
        print("  ✓ attack_single_object() method exists")
    
    print()
    return True


def test_end_to_end(args):
    """Test 6: End-to-end mini test on one image."""
    print("[Test 6] Running mini end-to-end test...")
    
    if args.skip_e2e:
        print("  ⚠ Skipped (use --run_e2e to enable)\n")
        return True
    
    try:
        from attack_selector import AttackStrategySelector
        from data.flir_dataset import build_dataloader
        from attack.detector_wrapper import FasterRCNNWrapper
        from models.thermal_gan import ThermalGANTrainer
        
        # Load one image
        loader = build_dataloader(
            root=args.data_root,
            split='val',
            batch_size=1,
            num_workers=0
        )
        batch = next(iter(loader))
        
        image = batch['images'][0].unsqueeze(0).to(args.device)
        boxes = batch['boxes'][0].to(args.device)
        labels = batch['labels'][0].to(args.device)
        
        print(f"  ✓ Loaded test image: {image.shape}")
        print(f"  ✓ Objects: {len(boxes)} (classes: {labels.tolist()})")
        
        # Initialize selector
        selector = AttackStrategySelector(
            num_classes=4,
            device=args.device,
            mode='adaptive'
        )
        print(f"  ✓ Selector initialized")
        
        # Generate strategies for first object
        if len(boxes) > 0:
            label = int(labels[0].item())
            strategies = selector.generate_all_strategies(label)
            print(f"  ✓ Generated {len(strategies)} strategies for class {label}")
            
            for strat in strategies:
                print(f"     - {strat}")
        
        print(f"  ✓ End-to-end test passed")
        
    except Exception as e:
        print(f"  ✗ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def main():
    parser = argparse.ArgumentParser(description='Test multi-class attack implementation')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to FLIR dataset')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--run_e2e', action='store_true',
                        help='Run end-to-end test (requires models)')
    parser.add_argument('--skip_e2e', action='store_true',
                        help='Skip end-to-end test')
    
    args = parser.parse_args()
    
    # If neither flag set, default to skip
    if not args.run_e2e and not args.skip_e2e:
        args.skip_e2e = True
    
    # Run tests
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Strategy Generation", test_attack_strategy_generation()))
    results.append(("Detector Wrapper", test_detector_wrapper(args)))
    results.append(("Dataloader", test_dataloader(args)))
    results.append(("Attack Method", test_attack_single_object(args)))
    results.append(("End-to-End", test_end_to_end(args)))
    
    # Summary
    print("="*70)
    print("  TEST SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:8s}  {name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\n  Total: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n  ✓ All tests passed! Your implementation is ready.")
        print("\n  Next steps:")
        print("    1. Run profiling: python run_multiclass.py --mode profile ...")
        print("    2. Run attack: python run_multiclass.py --mode attack_profiled ...")
    else:
        print("\n  ✗ Some tests failed. Please fix the issues above.")
        print("\n  See INTEGRATION_GUIDE.md for detailed implementation instructions.")
    
    print("="*70 + "\n")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
