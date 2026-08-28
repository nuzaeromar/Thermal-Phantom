"""
attack_selector.py
Multi-class adaptive attack strategy selector for ThermalMirage.

Automatically determines the best attack strategy (hide vs misclassify) 
for each detected object based on attack effectiveness.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json
from pathlib import Path


@dataclass
class AttackStrategy:
    """Defines an attack strategy for a specific class."""
    src_class: int
    tgt_class: int  # -1 for hide, 0-3 for misclassification
    mode: str       # 'hide' or 'misclassify'
    
    def __repr__(self):
        if self.mode == 'hide':
            return f"Hide(class_{self.src_class})"
        else:
            return f"Misclassify(class_{self.src_class}→{self.tgt_class})"


@dataclass
class AttackResult:
    """Result of testing a single attack strategy."""
    strategy: AttackStrategy
    success: bool
    confidence_drop: float
    perturbation_size: float
    detection_suppressed: bool  # True if object no longer detected
    misclassified: bool         # True if classified as target class
    
    def get_score(self) -> float:
        """
        Compute attack effectiveness score.
        Higher is better. Prioritizes success, then confidence drop, then minimal perturbation.
        """
        if not self.success:
            return 0.0
        # Success: weight confidence drop heavily, penalize large perturbations
        return self.confidence_drop * 10.0 - self.perturbation_size * 0.5


class AttackStrategySelector:
    """
    Selects optimal attack strategy for each object by testing multiple options.
    
    Two modes:
    1. Offline profiling: Pre-compute best strategies per class on validation set
    2. Online adaptive: Test strategies on-the-fly for each object
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        class_names: List[str] = None,
        device: str = "cuda",
        mode: str = "adaptive"  # 'adaptive' or 'profiled'
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.device = device
        self.mode = mode
        
        # Store profiled best strategies (if using profiled mode)
        self.best_strategies: Dict[int, List[AttackStrategy]] = {}
        self.strategy_stats: Dict[str, Dict] = defaultdict(lambda: {
            'attempts': 0,
            'successes': 0,
            'avg_conf_drop': 0.0,
            'avg_perturbation': 0.0
        })
    
    def generate_all_strategies(self, src_class: int) -> List[AttackStrategy]:
        """Generate all possible attack strategies for a source class."""
        strategies = []
        
        # Strategy 1: Hide (background)
        strategies.append(AttackStrategy(
            src_class=src_class,
            tgt_class=-1,
            mode='hide'
        ))
        
        # Strategies 2-N: Misclassify to each other class
        for tgt_class in range(self.num_classes):
            if tgt_class != src_class:  # Can't misclassify to itself
                strategies.append(AttackStrategy(
                    src_class=src_class,
                    tgt_class=tgt_class,
                    mode='misclassify'
                ))
        
        return strategies
    
    def test_strategy(
        self,
        attack_engine,  # Your ThermalMirageAttack instance
        image: torch.Tensor,
        box: torch.Tensor,
        label: int,
        strategy: AttackStrategy,
        verbose_refinement: bool = False,
    ) -> AttackResult:
        """
        Test a single attack strategy on one object.
        Returns metrics about attack success.
        """
        # Apply attack with this strategy
        try:
            adv_image, attack_info = attack_engine.attack_single_object(
                image=image,
                box=box,
                src_class=strategy.src_class,
                tgt_class=strategy.tgt_class,
                attack_mode=strategy.mode,
                verbose=verbose_refinement,
            )
            
            # Evaluate attack success
            original_conf = attack_info.get('original_confidence', 0.0)
            adv_conf = attack_info.get('adversarial_confidence', 0.0)
            conf_drop = original_conf - adv_conf
            perturbation = attack_info.get('perturbation_linf', 0.0)
            
            # Check if attack succeeded
            detection_suppressed = adv_conf < 0.3  # Threshold for detection
            
            if strategy.mode == 'hide':
                success = detection_suppressed
                misclassified = False
            else:
                # For misclassification: object detected as target class
                pred_class = attack_info.get('predicted_class', -1)
                misclassified = (pred_class == strategy.tgt_class)
                success = misclassified
            
            return AttackResult(
                strategy=strategy,
                success=success,
                confidence_drop=conf_drop,
                perturbation_size=perturbation,
                detection_suppressed=detection_suppressed,
                misclassified=misclassified
            )
            
        except Exception as e:
            print(f"[AttackSelector] Strategy {strategy} failed: {e}")
            return AttackResult(
                strategy=strategy,
                success=False,
                confidence_drop=0.0,
                perturbation_size=float('inf'),
                detection_suppressed=False,
                misclassified=False
            )
    
    def select_best_strategy(
        self,
        attack_engine,
        image: torch.Tensor,
        box: torch.Tensor,
        label: int,
        test_all: bool = True,
        top_k: int = 3,
        verbose_refinement: bool = False,
    ) -> Tuple[AttackStrategy, AttackResult]:
        """
        Select the best attack strategy for a single object.

        This version is robust to empty or incomplete profiled strategies.
        If a class has no stored profile entries, it falls back to generated
        strategies instead of returning best_result=None.
        """
        if self.mode == 'profiled' and label in self.best_strategies and not test_all:
            strategies_to_test = self.best_strategies.get(label, [])[:top_k]

            # Fix: if the loaded profile contains an empty list for this class,
            # fall back to generated strategies.
            if len(strategies_to_test) == 0:
                print(f"[AttackSelector] Empty profile for class {label}; "
                      f"falling back to generated strategies.")
                strategies_to_test = self.generate_all_strategies(label)
        else:
            strategies_to_test = self.generate_all_strategies(label)

        results = []
        for strategy in strategies_to_test:
            result = self.test_strategy(
                attack_engine, image, box, label, strategy,
                verbose_refinement=verbose_refinement,
            )
            results.append(result)

            # Update statistics
            key = str(strategy)
            self.strategy_stats[key]['attempts'] += 1
            if result.success:
                self.strategy_stats[key]['successes'] += 1
            self.strategy_stats[key]['avg_conf_drop'] += result.confidence_drop
            self.strategy_stats[key]['avg_perturbation'] += result.perturbation_size

        results.sort(key=lambda r: r.get_score(), reverse=True)

        if not results:
            # Defensive fallback: never return None for best_result.
            fallback = AttackStrategy(src_class=label, tgt_class=-1, mode='hide')
            fallback_result = AttackResult(
                strategy=fallback,
                success=False,
                confidence_drop=0.0,
                perturbation_size=0.0,
                detection_suppressed=False,
                misclassified=False
            )
            return fallback, fallback_result

        if results[0].get_score() == 0:
            # All tested strategies failed. Return a valid fallback result.
            fallback = AttackStrategy(src_class=label, tgt_class=-1, mode='hide')
            failed_result = results[0]
            failed_result.strategy = fallback
            return fallback, failed_result

        return results[0].strategy, results[0]

    def attack_multi_class_image(
        self,
        attack_engine,
        image: torch.Tensor,
        boxes: torch.Tensor,      # (N, 4)
        labels: torch.Tensor,     # (N,)
        test_all: bool = False,   # Set False for speed
        verbose: bool = True,
        verbose_refinement: bool = False,
    ) -> Dict:
        """
        Attack all objects in an image with automatically selected strategies.
        
        Returns:
            {
                'adv_image': final adversarial image,
                'strategies_used': list of (box_idx, strategy, result),
                'overall_success_rate': float,
                'per_class_success': dict
            }
        """
        N = len(boxes)
        if N == 0:
            return {
                'adv_image': image,
                'strategies_used': [],
                'overall_success_rate': 0.0,
                'per_class_success': {}
            }
        
        # Start with clean image
        current_image = image.clone()
        strategies_used = []
        per_class_results = defaultdict(list)
        
        if verbose:
            print(f"\n[AttackSelector] Processing {N} objects in image...")
        
        # For each object, select and apply best strategy
        for i, (box, label) in enumerate(zip(boxes, labels)):
            label_int = int(label.item())
            
            if verbose:
                print(f"  Object {i+1}/{N}: {self.class_names[label_int]} at {box.tolist()}")
            
            # Select best strategy for this object
            best_strategy, best_result = self.select_best_strategy(
                attack_engine=attack_engine,
                image=current_image,
                box=box,
                label=label_int,
                test_all=test_all,
                verbose_refinement=verbose_refinement,
            )

            if best_result is None:
                # Defensive guard: select_best_strategy should always return a
                # valid AttackResult, but keep this to avoid crashing long jobs.
                best_result = AttackResult(
                    strategy=best_strategy,
                    success=False,
                    confidence_drop=0.0,
                    perturbation_size=0.0,
                    detection_suppressed=False,
                    misclassified=False
                )
            
            if verbose:
                status = "✓ SUCCESS" if best_result.success else "✗ FAILED"
                print(f"    → {best_strategy} | {status} | "
                      f"conf_drop={best_result.confidence_drop:.3f} | "
                      f"pert={best_result.perturbation_size:.4f}")
            
            # Apply the selected strategy to accumulate attacks
            current_image, _ = attack_engine.attack_single_object(
                image=current_image,
                box=box,
                src_class=best_strategy.src_class,
                tgt_class=best_strategy.tgt_class,
                attack_mode=best_strategy.mode
            )
            
            strategies_used.append((i, best_strategy, best_result))
            per_class_results[label_int].append(best_result.success)
        
        # Compute success rates
        overall_success = np.mean([r.success for _, _, r in strategies_used])
        per_class_success = {
            cls: np.mean(results) for cls, results in per_class_results.items()
        }
        
        return {
            'adv_image': current_image,
            'strategies_used': strategies_used,
            'overall_success_rate': overall_success,
            'per_class_success': per_class_success,
            'num_objects': N
        }
    
    def profile_strategies(
        self,
        attack_engine,
        val_loader,
        save_path: str = "checkpoints/strategy_profile.json",
        max_images: int = 100
    ):
        """
        Pre-compute best strategies for each class on validation set.
        This creates a profile that can be used for faster inference.
        """
        print(f"\n[AttackSelector] Profiling attack strategies on validation set...")
        
        class_strategy_scores = defaultdict(lambda: defaultdict(list))
        
        for img_idx, batch in enumerate(val_loader):
            if img_idx >= max_images:
                break
            
            images = batch['images']
            boxes_list = batch['boxes']
            labels_list = batch['labels']
            
            for image, boxes, labels in zip(images, boxes_list, labels_list):
                for box, label in zip(boxes, labels):
                    label_int = int(label.item())
                    
                    # Test all strategies for this object
                    strategies = self.generate_all_strategies(label_int)
                    for strategy in strategies:
                        result = self.test_strategy(
                            attack_engine, image, box, label_int, strategy
                        )
                        class_strategy_scores[label_int][str(strategy)].append(
                            result.get_score()
                        )
            
            if (img_idx + 1) % 10 == 0:
                print(f"  Processed {img_idx + 1}/{max_images} images...")
        
        # Compute average scores and rank strategies per class
        for cls in range(self.num_classes):
            strategy_avg_scores = {}
            for strategy_str, scores in class_strategy_scores[cls].items():
                strategy_avg_scores[strategy_str] = np.mean(scores)
            
            # Sort strategies by average score
            sorted_strategies = sorted(
                strategy_avg_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Store top 3 strategies for this class
            self.best_strategies[cls] = []
            print(f"\n  Class {cls} ({self.class_names[cls]}) best strategies:")
            for i, (strat_str, score) in enumerate(sorted_strategies[:3]):
                print(f"    {i+1}. {strat_str}: {score:.3f}")
                # Parse strategy string back to AttackStrategy object
                # (simplified - you may need better parsing)
                self.best_strategies[cls].append(
                    self._parse_strategy_string(strat_str, cls)
                )
        
        # Save profile
        self.save_profile(save_path)
        print(f"\n[AttackSelector] Profile saved to {save_path}")
    
    def _parse_strategy_string(self, strategy_str: str, src_class: int) -> AttackStrategy:
        """Parse strategy string back to AttackStrategy object."""
        if "Hide" in strategy_str:
            return AttackStrategy(src_class=src_class, tgt_class=-1, mode='hide')
        else:
            # Extract target class from "Misclassify(class_X→Y)"
            import re
            match = re.search(r'→(\d+)', strategy_str)
            if match:
                tgt_class = int(match.group(1))
                return AttackStrategy(
                    src_class=src_class,
                    tgt_class=tgt_class,
                    mode='misclassify'
                )
        # Fallback
        return AttackStrategy(src_class=src_class, tgt_class=-1, mode='hide')
    
    def save_profile(self, path: str):
        """Save profiled strategies to JSON."""
        profile_data = {
            'best_strategies': {
                cls: [str(s) for s in strategies]
                for cls, strategies in self.best_strategies.items()
            },
            'statistics': dict(self.strategy_stats)
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(profile_data, f, indent=2)
    
    def load_profile(self, path: str):
        """Load profiled strategies from JSON."""
        with open(path, 'r') as f:
            profile_data = json.load(f)
        
        # Reconstruct best_strategies
        for cls_str, strat_strings in profile_data['best_strategies'].items():
            cls = int(cls_str)
            self.best_strategies[cls] = [
                self._parse_strategy_string(s, cls) for s in strat_strings
            ]
        
        self.strategy_stats = defaultdict(lambda: {
            'attempts': 0, 'successes': 0,
            'avg_conf_drop': 0.0, 'avg_perturbation': 0.0
        }, profile_data.get('statistics', {}))
        
        print(f"[AttackSelector] Loaded profile from {path}")
