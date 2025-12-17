#!/usr/bin/env python3
"""
DSPy Optimization for Entity Extraction and Keywords Extraction Prompts.

Uses DSPy MIPROv2 and BootstrapFewShot to optimize:
1. Entity extraction prompt - improve relation precision
2. Keywords extraction prompt - improve high-level keyword extraction

Usage:
    # Optimize entity extraction
    python optimize_extraction_prompts.py --type entity --quick

    # Optimize keywords extraction
    python optimize_extraction_prompts.py --type keywords --quick

    # Optimize both
    python optimize_extraction_prompts.py --type all

    # Full optimization (more training examples)
    python optimize_extraction_prompts.py --type all --mode medium
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean

import dspy
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from openai import AsyncOpenAI


# ============================================================================
# Test Data
# ============================================================================

ENTITY_EXTRACTION_TRAINING_DATA = [
    {
        "input_text": """
        Sanofi announced a partnership with Regeneron Pharmaceuticals to develop
        a new COVID-19 treatment. The collaboration will focus on monoclonal
        antibodies targeting the spike protein. Dr. Paul Hudson, CEO of Sanofi,
        stated that the partnership represents a significant step forward.
        """,
        "expected_entities": ["Sanofi", "Regeneron Pharmaceuticals", "Paul Hudson", "COVID-19", "Monoclonal Antibodies", "Spike Protein"],
        "expected_relations": [
            ("Sanofi", "Regeneron Pharmaceuticals", "partnership"),
            ("Paul Hudson", "Sanofi", "leads"),
            ("Monoclonal Antibodies", "Spike Protein", "targets"),
        ],
    },
    {
        "input_text": """
        The FDA approved Keytruda (pembrolizumab) manufactured by Merck for the
        treatment of non-small cell lung cancer. The approval was based on the
        KEYNOTE-024 clinical trial which showed a 40% reduction in disease
        progression. Keytruda works by blocking PD-1.
        """,
        "expected_entities": ["FDA", "Keytruda", "Pembrolizumab", "Merck", "Non-Small Cell Lung Cancer", "KEYNOTE-024", "PD-1"],
        "expected_relations": [
            ("FDA", "Keytruda", "approved"),
            ("Merck", "Keytruda", "manufactures"),
            ("Keytruda", "PD-1", "blocks"),
        ],
    },
    {
        "input_text": """
        BioNTech and Pfizer collaborated to develop the BNT162b2 mRNA vaccine
        against COVID-19. Dr. Ugur Sahin, CEO of BioNTech, led the research team.
        The vaccine received Emergency Use Authorization from the FDA in December 2020.
        """,
        "expected_entities": ["BioNTech", "Pfizer", "BNT162b2", "COVID-19", "Ugur Sahin", "FDA", "Emergency Use Authorization"],
        "expected_relations": [
            ("BioNTech", "Pfizer", "collaborated"),
            ("BioNTech", "BNT162b2", "developed"),
            ("Ugur Sahin", "BioNTech", "leads"),
            ("FDA", "BNT162b2", "authorized"),
        ],
    },
    {
        "input_text": """
        AstraZeneca acquired Alexion Pharmaceuticals for $39 billion to strengthen
        its rare disease portfolio. The acquisition included Soliris (eculizumab),
        a treatment for paroxysmal nocturnal hemoglobinuria. Pascal Soriot, CEO of
        AstraZeneca, called it a transformational deal.
        """,
        "expected_entities": ["AstraZeneca", "Alexion Pharmaceuticals", "Soliris", "Eculizumab", "Paroxysmal Nocturnal Hemoglobinuria", "Pascal Soriot"],
        "expected_relations": [
            ("AstraZeneca", "Alexion Pharmaceuticals", "acquired"),
            ("Alexion Pharmaceuticals", "Soliris", "produces"),
            ("Soliris", "Paroxysmal Nocturnal Hemoglobinuria", "treats"),
            ("Pascal Soriot", "AstraZeneca", "leads"),
        ],
    },
    {
        "input_text": """
        Moderna's mRNA-1273 vaccine was developed in collaboration with the
        National Institutes of Health (NIH). Dr. Anthony Fauci, director of NIAID,
        supported the rapid development program. The vaccine targets the
        SARS-CoV-2 spike protein and achieved 94% efficacy in Phase 3 trials.
        """,
        "expected_entities": ["Moderna", "mRNA-1273", "NIH", "National Institutes of Health", "Anthony Fauci", "NIAID", "SARS-CoV-2", "Spike Protein"],
        "expected_relations": [
            ("Moderna", "NIH", "collaborated"),
            ("Moderna", "mRNA-1273", "developed"),
            ("Anthony Fauci", "NIAID", "directs"),
            ("mRNA-1273", "Spike Protein", "targets"),
        ],
    },
]


KEYWORDS_EXTRACTION_TRAINING_DATA = [
    {
        "query": "What is the mechanism of action of Keytruda for lung cancer treatment?",
        "expected_high_level": ["mechanism of action", "cancer treatment"],
        "expected_low_level": ["Keytruda", "lung cancer"],
    },
    {
        "query": "How does CRISPR-Cas9 gene editing compare to traditional methods?",
        "expected_high_level": ["gene editing", "comparison", "methods"],
        "expected_low_level": ["CRISPR-Cas9"],
    },
    {
        "query": "What are the FDA approval requirements for biosimilars in 2024?",
        "expected_high_level": ["approval requirements", "regulatory process"],
        "expected_low_level": ["FDA", "biosimilars", "2024"],
    },
    {
        "query": "Describe the partnership between Pfizer and BioNTech for COVID-19 vaccine development",
        "expected_high_level": ["partnership", "collaboration", "vaccine development"],
        "expected_low_level": ["Pfizer", "BioNTech", "COVID-19"],
    },
    {
        "query": "What were the Phase 3 trial results for Ozempic in diabetes management?",
        "expected_high_level": ["clinical trial results", "diabetes management", "efficacy"],
        "expected_low_level": ["Phase 3", "Ozempic", "diabetes"],
    },
    {
        "query": "How does Humira work for rheumatoid arthritis and what are its side effects?",
        "expected_high_level": ["mechanism of action", "side effects", "treatment"],
        "expected_low_level": ["Humira", "rheumatoid arthritis"],
    },
    {
        "query": "What regulatory challenges did Biogen face with Aduhelm approval?",
        "expected_high_level": ["regulatory challenges", "drug approval", "controversy"],
        "expected_low_level": ["Biogen", "Aduhelm"],
    },
    {
        "query": "Compare the efficacy of mRNA vaccines vs viral vector vaccines for COVID-19",
        "expected_high_level": ["efficacy comparison", "vaccine technology", "immunization"],
        "expected_low_level": ["mRNA vaccines", "viral vector vaccines", "COVID-19"],
    },
]


# ============================================================================
# DSPy Signatures
# ============================================================================


class EntityExtractionSignature(dspy.Signature):
    """Extract entities and relationships from text for a knowledge graph.

    Entities should be meaningful concepts, people, organizations, or things.
    Relationships should capture how entities are connected.
    """

    text: str = dspy.InputField(desc="Input text to extract entities and relationships from")
    entities: str = dspy.OutputField(
        desc="Comma-separated list of extracted entities (names only, no types)"
    )
    relationships: str = dspy.OutputField(
        desc="List of relationships in format: (entity1, entity2, relationship_type)"
    )


class EntityExtractionWithTypesSignature(dspy.Signature):
    """Extract entities with types and relationships from text.

    Focus on identifying:
    - Organizations (companies, institutions)
    - People (names with roles)
    - Products/Artifacts (drugs, treatments, technologies)
    - Concepts (diseases, mechanisms, processes)
    - Events (trials, approvals, studies)

    For relationships, identify clear connections like:
    - partnership, collaboration, acquisition
    - develops, manufactures, produces
    - treats, targets, blocks, inhibits
    - leads, directs, founded
    """

    text: str = dspy.InputField(desc="Input text to extract from")
    entity_types: str = dspy.InputField(desc="Available entity types to use")
    entities_with_types: str = dspy.OutputField(
        desc="Entities with types in format: entity_name (type), separated by newlines"
    )
    relationships: str = dspy.OutputField(
        desc="Relationships in format: source -> target [relationship_type], separated by newlines"
    )


class KeywordsExtractionSignature(dspy.Signature):
    """Extract search keywords from a user query for RAG retrieval.

    High-level keywords are thematic concepts that capture the query intent.
    Low-level keywords are specific entities, names, and technical terms.
    """

    query: str = dspy.InputField(desc="User's search query")
    high_level_keywords: str = dspy.OutputField(
        desc="Comma-separated thematic/conceptual keywords (2-4 keywords)"
    )
    low_level_keywords: str = dspy.OutputField(
        desc="Comma-separated specific entities and technical terms (1-4 keywords)"
    )


# ============================================================================
# DSPy Modules
# ============================================================================


class EntityExtractionModule(dspy.Module):
    """Module for entity and relationship extraction."""

    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(EntityExtractionWithTypesSignature)

    def forward(self, text: str, entity_types: str = "Person, Organization, Location, Event, Concept, Artifact"):
        return self.extract(text=text, entity_types=entity_types)


class KeywordsExtractionModule(dspy.Module):
    """Module for keyword extraction from queries."""

    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(KeywordsExtractionSignature)

    def forward(self, query: str):
        return self.extract(query=query)


# ============================================================================
# Metrics
# ============================================================================


def entity_extraction_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Evaluate entity extraction quality."""
    # Parse predicted entities
    pred_entities_raw = getattr(pred, "entities_with_types", "") or getattr(pred, "entities", "")
    pred_entities = set()
    for line in pred_entities_raw.replace(",", "\n").split("\n"):
        # Extract entity name (remove type annotation)
        name = re.sub(r"\s*\([^)]*\)\s*", "", line.strip())
        if name:
            pred_entities.add(name.lower())

    # Parse predicted relationships
    pred_relations_raw = getattr(pred, "relationships", "")
    pred_relations = set()
    for line in pred_relations_raw.split("\n"):
        # Parse formats like "A -> B [type]" or "(A, B, type)"
        match = re.search(r"([^->(),]+)\s*(?:->|,)\s*([^->(),\[\]]+)", line)
        if match:
            src = match.group(1).strip().lower()
            tgt = match.group(2).strip().lower()
            pred_relations.add((src, tgt))

    # Get expected values
    exp_entities = {e.lower() for e in example.expected_entities}
    exp_relations = {(r[0].lower(), r[1].lower()) for r in example.expected_relations}

    # Calculate entity metrics
    entity_tp = len(pred_entities & exp_entities)
    entity_precision = entity_tp / len(pred_entities) if pred_entities else 0.0
    entity_recall = entity_tp / len(exp_entities) if exp_entities else 0.0
    entity_f1 = (
        2 * entity_precision * entity_recall / (entity_precision + entity_recall)
        if (entity_precision + entity_recall) > 0
        else 0.0
    )

    # Calculate relationship metrics
    relation_tp = len(pred_relations & exp_relations)
    relation_precision = relation_tp / len(pred_relations) if pred_relations else 0.0
    relation_recall = relation_tp / len(exp_relations) if exp_relations else 0.0
    relation_f1 = (
        2 * relation_precision * relation_recall / (relation_precision + relation_recall)
        if (relation_precision + relation_recall) > 0
        else 0.0
    )

    # Combined score (weight relations higher since that's our weak point)
    score = 0.4 * entity_f1 + 0.6 * relation_f1

    if trace:
        print(f"  Entity F1: {entity_f1:.3f} (P={entity_precision:.2f}, R={entity_recall:.2f})")
        print(f"  Relation F1: {relation_f1:.3f} (P={relation_precision:.2f}, R={relation_recall:.2f})")
        print(f"  Combined: {score:.3f}")

    return score


def keywords_extraction_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Evaluate keyword extraction quality."""
    # Parse predicted keywords
    pred_high = {k.strip().lower() for k in (getattr(pred, "high_level_keywords", "") or "").split(",") if k.strip()}
    pred_low = {k.strip().lower() for k in (getattr(pred, "low_level_keywords", "") or "").split(",") if k.strip()}

    # Get expected keywords
    exp_high = {k.lower() for k in example.expected_high_level}
    exp_low = {k.lower() for k in example.expected_low_level}

    # Calculate high-level keyword metrics
    high_tp = len(pred_high & exp_high)
    high_precision = high_tp / len(pred_high) if pred_high else 0.0
    high_recall = high_tp / len(exp_high) if exp_high else 0.0
    high_f1 = (
        2 * high_precision * high_recall / (high_precision + high_recall)
        if (high_precision + high_recall) > 0
        else 0.0
    )

    # Calculate low-level keyword metrics
    low_tp = len(pred_low & exp_low)
    low_precision = low_tp / len(pred_low) if pred_low else 0.0
    low_recall = low_tp / len(exp_low) if exp_low else 0.0
    low_f1 = (
        2 * low_precision * low_recall / (low_precision + low_recall)
        if (low_precision + low_recall) > 0
        else 0.0
    )

    # Combined score (weight high-level higher since that's our weak point)
    score = 0.6 * high_f1 + 0.4 * low_f1

    if trace:
        print(f"  High-level F1: {high_f1:.3f} (P={high_precision:.2f}, R={high_recall:.2f})")
        print(f"  Low-level F1: {low_f1:.3f} (P={low_precision:.2f}, R={low_recall:.2f})")
        print(f"  Combined: {score:.3f}")

    return score


# ============================================================================
# Optimization
# ============================================================================


def prepare_entity_examples() -> list[dspy.Example]:
    """Prepare DSPy examples for entity extraction."""
    examples = []
    for data in ENTITY_EXTRACTION_TRAINING_DATA:
        example = dspy.Example(
            text=data["input_text"].strip(),
            entity_types="Person, Organization, Location, Event, Concept, Artifact",
            expected_entities=data["expected_entities"],
            expected_relations=data["expected_relations"],
        ).with_inputs("text", "entity_types")
        examples.append(example)
    return examples


def prepare_keywords_examples() -> list[dspy.Example]:
    """Prepare DSPy examples for keywords extraction."""
    examples = []
    for data in KEYWORDS_EXTRACTION_TRAINING_DATA:
        example = dspy.Example(
            query=data["query"],
            expected_high_level=data["expected_high_level"],
            expected_low_level=data["expected_low_level"],
        ).with_inputs("query")
        examples.append(example)
    return examples


def optimize_with_bootstrap(
    module: dspy.Module,
    trainset: list[dspy.Example],
    metric,
    max_demos: int = 4,
) -> dspy.Module:
    """Optimize module using BootstrapFewShot."""
    from dspy.teleprompt import BootstrapFewShot

    print("\n🔄 Running BootstrapFewShot optimization...")

    optimizer = BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=max_demos,
        max_labeled_demos=max_demos,
        max_rounds=2,
    )

    optimized = optimizer.compile(module, trainset=trainset)
    return optimized


def optimize_with_mipro(
    module: dspy.Module,
    trainset: list[dspy.Example],
    metric,
    mode: str = "light",
) -> dspy.Module:
    """Optimize module using MIPROv2."""
    from dspy.teleprompt import MIPROv2

    print(f"\n🔄 Running MIPROv2 optimization (mode={mode})...")

    optimizer = MIPROv2(
        metric=metric,
        auto=mode,
        num_threads=2,
    )

    optimized = optimizer.compile(
        module,
        trainset=trainset,
        requires_permission_to_run=False,
    )

    return optimized


def evaluate_module(
    module: dspy.Module,
    testset: list[dspy.Example],
    metric,
    name: str = "Module",
) -> float:
    """Evaluate a module on test set."""
    from dspy.evaluate import Evaluate

    print(f"\n📊 Evaluating {name}...")

    evaluator = Evaluate(
        devset=testset,
        num_threads=2,
        display_progress=True,
        display_table=0,
    )

    result = evaluator(module, metric=metric)

    if hasattr(result, "score"):
        score = float(result.score)
    elif isinstance(result, (int, float)):
        score = float(result)
    else:
        score = float(str(result).split("%")[0].split()[-1]) / 100 if "%" in str(result) else 0.0

    print(f"✅ {name} score: {score:.3f}")
    return score


def extract_optimized_instructions(module: dspy.Module) -> str:
    """Extract optimized instructions from a DSPy module."""
    parts = []

    for name, predictor in module.named_predictors():
        if hasattr(predictor, "signature"):
            sig = predictor.signature
            if sig.__doc__:
                parts.append(f"# Signature: {name}")
                parts.append(sig.__doc__)

        if hasattr(predictor, "demos") and predictor.demos:
            parts.append(f"\n# Examples from {name}:")
            for i, demo in enumerate(predictor.demos[:3]):  # Show first 3
                parts.append(f"\n## Example {i+1}")
                for key, value in demo.items():
                    if isinstance(value, str):
                        value = value[:200] + "..." if len(value) > 200 else value
                    parts.append(f"  {key}: {value}")

        if hasattr(predictor, "extended_signature"):
            parts.append(f"\n# Extended Instructions for {name}:")
            parts.append(str(predictor.extended_signature))

    return "\n".join(parts)


# ============================================================================
# Main
# ============================================================================


async def main():
    parser = argparse.ArgumentParser(description="Optimize Extraction Prompts with DSPy")
    parser.add_argument(
        "--type", "-t",
        choices=["entity", "keywords", "all"],
        default="all",
        help="Type of prompt to optimize",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["light", "medium", "heavy"],
        default="light",
        help="Optimization intensity",
    )
    parser.add_argument(
        "--optimizer", "-o",
        choices=["bootstrap", "mipro", "both"],
        default="bootstrap",
        help="Optimizer to use",
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode (fewer examples)",
    )
    parser.add_argument(
        "--output", "-O",
        type=str,
        default="optimization_results",
        help="Output directory",
    )
    args = parser.parse_args()

    # Configure DSPy
    lm = dspy.LM(
        model=os.getenv("LLM_MODEL", "openai/gpt-4o-mini"),
        api_key=os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("LLM_BINDING_HOST", "https://api.openai.com/v1"),
        temperature=0.1,
    )
    dspy.configure(lm=lm)

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"extraction_opt_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🚀 DSPy PROMPT OPTIMIZATION")
    print("=" * 70)
    print(f"Type: {args.type}")
    print(f"Mode: {args.mode}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    results = {}

    # =========================================================================
    # Entity Extraction Optimization
    # =========================================================================
    if args.type in ["entity", "all"]:
        print("\n" + "=" * 60)
        print("📦 ENTITY EXTRACTION OPTIMIZATION")
        print("=" * 60)

        examples = prepare_entity_examples()
        if args.quick:
            examples = examples[:3]

        split_idx = max(1, len(examples) - 1)
        trainset = examples[:split_idx]
        testset = examples[split_idx:]

        print(f"Train: {len(trainset)} examples, Test: {len(testset)} examples")

        module = EntityExtractionModule()

        # Evaluate baseline
        baseline_score = evaluate_module(module, testset, entity_extraction_metric, "Baseline")

        # Optimize
        if args.optimizer in ["bootstrap", "both"]:
            module = optimize_with_bootstrap(module, trainset, entity_extraction_metric)
            bootstrap_score = evaluate_module(module, testset, entity_extraction_metric, "Bootstrap")

        if args.optimizer in ["mipro", "both"]:
            module = optimize_with_mipro(module, trainset, entity_extraction_metric, args.mode)
            mipro_score = evaluate_module(module, testset, entity_extraction_metric, "MIPRO")

        # Extract and save optimized instructions
        instructions = extract_optimized_instructions(module)
        with open(output_dir / "entity_extraction_optimized.txt", "w") as f:
            f.write(instructions)

        results["entity_extraction"] = {
            "baseline": baseline_score,
            "optimized": bootstrap_score if args.optimizer in ["bootstrap", "both"] else mipro_score,
        }

        print(f"\n✅ Entity extraction optimization complete!")
        print(f"   Baseline: {baseline_score:.3f}")
        print(f"   Optimized: {results['entity_extraction']['optimized']:.3f}")

    # =========================================================================
    # Keywords Extraction Optimization
    # =========================================================================
    if args.type in ["keywords", "all"]:
        print("\n" + "=" * 60)
        print("🔑 KEYWORDS EXTRACTION OPTIMIZATION")
        print("=" * 60)

        examples = prepare_keywords_examples()
        if args.quick:
            examples = examples[:4]

        split_idx = max(1, len(examples) - 2)
        trainset = examples[:split_idx]
        testset = examples[split_idx:]

        print(f"Train: {len(trainset)} examples, Test: {len(testset)} examples")

        module = KeywordsExtractionModule()

        # Evaluate baseline
        baseline_score = evaluate_module(module, testset, keywords_extraction_metric, "Baseline")

        # Optimize
        if args.optimizer in ["bootstrap", "both"]:
            module = optimize_with_bootstrap(module, trainset, keywords_extraction_metric)
            bootstrap_score = evaluate_module(module, testset, keywords_extraction_metric, "Bootstrap")

        if args.optimizer in ["mipro", "both"]:
            module = optimize_with_mipro(module, trainset, keywords_extraction_metric, args.mode)
            mipro_score = evaluate_module(module, testset, keywords_extraction_metric, "MIPRO")

        # Extract and save optimized instructions
        instructions = extract_optimized_instructions(module)
        with open(output_dir / "keywords_extraction_optimized.txt", "w") as f:
            f.write(instructions)

        results["keywords_extraction"] = {
            "baseline": baseline_score,
            "optimized": bootstrap_score if args.optimizer in ["bootstrap", "both"] else mipro_score,
        }

        print(f"\n✅ Keywords extraction optimization complete!")
        print(f"   Baseline: {baseline_score:.3f}")
        print(f"   Optimized: {results['keywords_extraction']['optimized']:.3f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 OPTIMIZATION SUMMARY")
    print("=" * 70)

    for prompt_type, scores in results.items():
        improvement = scores["optimized"] - scores["baseline"]
        print(f"{prompt_type}:")
        print(f"  Baseline: {scores['baseline']:.3f}")
        print(f"  Optimized: {scores['optimized']:.3f}")
        print(f"  Improvement: {improvement:+.3f} ({improvement/max(scores['baseline'], 0.001)*100:+.1f}%)")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
