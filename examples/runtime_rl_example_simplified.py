#!/usr/bin/env python3
"""
Runtime RL Example - Simplified with YAML Config
=================================================

This example demonstrates the SIMPLEST way to integrate Runtime RL
using YAML configuration.

Developer Code: ONLY 3 LINES!
------------------------------
1. Load config from YAML
2. Decide which method to use
3. Update after execution

No boilerplate, no complex initialization, just simple API.

Scenario:
---------
BTP construction company needs to estimate prices for building elements.
They have 3 pricing methods, and Runtime RL learns which one works best.

"""

# Fix import path to find adaptiq module
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adaptiq.core.runtime_rl import RuntimeRLHelper
from typing import Dict
import random


# ============================================================================
# GROUND TRUTH: Real Prices from Completed Projects
# ============================================================================
GROUND_TRUTH_PRICES = {
    # Concrete elements (price per m²: ~250€)
    ("concrete", 20, "north"): 5000.0,
    ("concrete", 20, "paris"): 5500.0,  # Paris more expensive
    ("concrete", 35, "north"): 8750.0,
    ("concrete", 35, "paris"): 9625.0,
    ("concrete", 50, "south"): 12000.0,

    # Brick elements (price per m²: ~180€)
    ("brick", 25, "north"): 4500.0,
    ("brick", 25, "paris"): 5000.0,
    ("brick", 40, "north"): 7200.0,
    ("brick", 40, "south"): 6800.0,
    ("brick", 60, "paris"): 11400.0,

    # Wood elements (price per m²: ~200€)
    ("wood", 15, "north"): 3000.0,
    ("wood", 15, "south"): 2850.0,
    ("wood", 30, "north"): 6000.0,
    ("wood", 30, "paris"): 6600.0,
    ("wood", 45, "south"): 8550.0,
}


# ============================================================================
# PRICING METHODS (Business Logic)
# ============================================================================

def method_db_standard(task_input: Dict) -> float:
    """
    Method 1: Standard Database Lookup (Batiprix-style)

    Uses FIXED national average prices from catalog.
    Does NOT account for regional variations.
    """
    material = task_input["material"]
    surface = task_input["surface"]

    # Fixed catalog prices (national average)
    price_per_m2_catalog = {
        "concrete": 250.0,
        "brick": 180.0,
        "wood": 200.0,
    }

    base_price = price_per_m2_catalog.get(material, 200.0) * surface
    variation = random.uniform(0.97, 1.03)  # ±3% catalog noise
    return base_price * variation


def method_knn_historical(task_input: Dict, k: int = 3) -> float:
    """
    Method 2: KNN (K-Nearest Neighbors) from Historical Data

    Algorithm:
    1. Filter by material + region (prefer exact match)
    2. Calculate distance (surface difference)
    3. Take K nearest neighbors
    4. Weighted average (inverse distance weighting)
    """
    material = task_input["material"]
    region = task_input["region"]
    surface = task_input["surface"]

    # Filter candidates by material and region
    candidates = []
    for (hist_material, hist_surface, hist_region), hist_price in GROUND_TRUTH_PRICES.items():
        if hist_material == material and hist_region == region:
            distance = abs(surface - hist_surface)
            candidates.append((distance, hist_surface, hist_price))

    # Fallback: if no exact region match, try same material
    if not candidates:
        for (hist_material, hist_surface, hist_region), hist_price in GROUND_TRUTH_PRICES.items():
            if hist_material == material:
                distance = abs(surface - hist_surface)
                candidates.append((distance, hist_surface, hist_price))

    # If still no candidates, use catalog as fallback
    if not candidates:
        return method_db_standard(task_input)

    # Sort by distance, take K neighbors
    candidates.sort(key=lambda x: x[0])
    k_neighbors = candidates[:min(k, len(candidates))]

    # Weighted average (inverse distance weighting)
    total_weight = 0.0
    weighted_sum = 0.0
    for distance, hist_surface, hist_price in k_neighbors:
        weight = 1.0 / (distance + 1.0)  # +1 to avoid division by zero
        weighted_sum += weight * hist_price
        total_weight += weight

    estimated_price = weighted_sum / total_weight

    # Add realistic variance (±3%)
    return estimated_price * random.uniform(0.97, 1.03)


def method_regional_adjust(task_input: Dict) -> float:
    """
    Method 3: Regional Adjustment on Catalog Prices

    Takes catalog price and applies regional multipliers.
    """
    material = task_input["material"]
    region = task_input["region"]
    surface = task_input["surface"]

    # Start with catalog price
    price_per_m2_catalog = {
        "concrete": 250.0,
        "brick": 180.0,
        "wood": 200.0,
    }

    base_price = price_per_m2_catalog.get(material, 200.0) * surface

    # Regional multipliers
    regional_factors = {
        "paris": 1.10,   # Paris 10% more expensive
        "north": 1.00,   # North baseline
        "south": 0.95,   # South 5% cheaper
    }

    adjusted_price = base_price * regional_factors.get(region, 1.00)
    variation = random.uniform(0.97, 1.03)  # ±3% noise
    return adjusted_price * variation


# ============================================================================
# AGENT CLASS (Simplified)
# ============================================================================

class BTPPricingAgent:
    """
    Simplified BTP Pricing Agent using Runtime RL.

    All configuration is in YAML file, agent code is minimal.
    """

    def __init__(self, config_path: str):
        """
        Initialize agent with YAML config.

        Args:
            config_path: Path to runtime_rl_config.yaml
        """
        # ✨ LINE 1: Load Runtime RL from YAML (ALL CONFIG IN YAML!)
        self.runtime_rl = RuntimeRLHelper.from_yaml(config_path)

        # Map action names to actual Python functions
        self.pricing_methods = {
            "method_db_standard": method_db_standard,
            "method_knn_historical": method_knn_historical,
            "method_regional_adjust": method_regional_adjust,
        }

    def estimate_price(self, element: Dict) -> Dict:
        """
        Estimate price for a construction element using Runtime RL.

        Args:
            element: {"material": str, "region": str, "surface": int}

        Returns:
            Result dictionary with estimation details
        """
        material = element["material"]
        region = element["region"]
        surface = element["surface"]

        print(f"\n{'='*70}")
        print(f"🏗️  Estimating: {material.upper()} | {region.upper()} | {surface}m²")
        print(f"{'='*70}")

        # ✨ LINE 2: Decide which method to use (SIMPLE API!)
        decision = self.runtime_rl.decide(
            subtask="estimate_price",
            metadata=element  # key_context auto-built from template in YAML
        )

        print(f"🎯 Method chosen: {decision.action.action}")
        print(f"   - Explored: {decision.explored}")
        print(f"   - Q-value: {decision.q_value:.4f}")

        # Execute the selected pricing method
        pricing_method = self.pricing_methods[decision.action.action]
        estimated_price = pricing_method(element)

        # Get ground truth for evaluation
        actual_price = GROUND_TRUTH_PRICES.get((material, surface, region))
        if actual_price is None:
            print(f"⚠️  No ground truth available, skipping update")
            return {"estimated": estimated_price, "actual": None}

        # Calculate error
        error = abs(estimated_price - actual_price)
        error_percent = (error / actual_price) * 100

        print(f"\n💰 Prices:")
        print(f"   - Estimated: {estimated_price:.2f}€")
        print(f"   - Actual: {actual_price:.2f}€")
        print(f"   - Error: {error:.2f}€ ({error_percent:.2f}%)")

        # Build result for reward calculation
        result = {
            "estimated": estimated_price,
            "actual": actual_price,
            "error_percent": error_percent,
        }

        # ✨ LINE 3: Update Q-table with result (SIMPLE API!)
        self.runtime_rl.update(decision, result)

        return result


# ============================================================================
# MAIN: Training & Evaluation
# ============================================================================

def main():
    """
    Demonstrate simplified Runtime RL integration.
    """
    print("=" * 70)
    print("🚀 Runtime RL - Simplified YAML-based Example")
    print("=" * 70)
    print("\n📋 All configuration is in YAML file:")
    print("   agents/btp_agent/runtime_rl_config.yaml")
    print("\n💻 Agent code: ONLY 3 LINES!")
    print("   1. Load from YAML")
    print("   2. Decide")
    print("   3. Update")
    print("=" * 70)

    # Initialize agent with YAML config
    config_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "agents",
        "btp_agent",
        "runtime_rl_config.yaml"
    )

    agent = BTPPricingAgent(config_path)

    # Training: Run 50 estimations
    print("\n" + "=" * 70)
    print("📚 PHASE 1: TRAINING (50 estimations)")
    print("=" * 70)

    all_elements = list(GROUND_TRUTH_PRICES.keys())

    for i in range(50):
        # Random element from ground truth
        material, surface, region = random.choice(all_elements)
        element = {"material": material, "region": region, "surface": surface}

        # Estimate price (decide + execute + update happens inside)
        agent.estimate_price(element)

    # Save Q-table
    print("\n" + "=" * 70)
    print("💾 Saving Q-table...")
    print("=" * 70)
    agent.runtime_rl.save()

    # Evaluation: Analyze learned Q-values
    print("\n" + "=" * 70)
    print("📊 PHASE 2: EVALUATION - Learned Q-values")
    print("=" * 70)

    # Set epsilon to 0 for pure exploitation (no exploration)
    agent.runtime_rl.q_manager.epsilon = 0.0

    # Test on all elements
    print("\n🔍 Testing all elements with learned policy (ε=0, pure exploitation):\n")

    total_error = 0.0
    count = 0

    for (material, surface, region), actual_price in sorted(GROUND_TRUTH_PRICES.items()):
        element = {"material": material, "region": region, "surface": surface}

        # Decide (no exploration)
        decision = agent.runtime_rl.decide(subtask="estimate_price", metadata=element)

        # Execute
        pricing_method = agent.pricing_methods[decision.action.action]
        estimated_price = pricing_method(element)

        error_percent = abs(estimated_price - actual_price) / actual_price * 100
        total_error += error_percent
        count += 1

        print(f"{material:8s} | {region:6s} | {surface:3d}m² → {decision.action.action:25s} "
              f"(Q={decision.q_value:+.4f}) | Error: {error_percent:5.2f}%")

    avg_error = total_error / count
    print(f"\n{'='*70}")
    print(f"📊 Average Error: {avg_error:.2f}%")
    print(f"{'='*70}")

    # Show Q-table stats
    stats = agent.runtime_rl.get_stats()
    print(f"\n📈 Q-table Statistics:")
    print(f"   - Total states: {stats['q_table_size']}")
    print(f"   - Actions: {stats['actions']}")
    print(f"   - Storage: {stats['storage_path']}")
    print(f"   - Hyperparameters: α={stats['hyperparameters']['alpha']}, "
          f"γ={stats['hyperparameters']['gamma']}, ε={stats['hyperparameters']['epsilon']}")

    print("\n" + "=" * 70)
    print("✅ Runtime RL Example Complete!")
    print("=" * 70)
    print("\n💡 Key Takeaway:")
    print("   With YAML config, developer writes MINIMAL code:")
    print("   - Load from YAML: 1 line")
    print("   - Decide: 1 line")
    print("   - Update: 1 line")
    print("\n   ALL configuration (key_context, reward, actions) is in YAML!")
    print("=" * 70)


if __name__ == "__main__":
    main()
