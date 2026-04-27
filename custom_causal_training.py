"""Custom causality detection model training for mobile prompt compression."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
import json
import numpy as np
from typing import List, Dict, Any, Tuple
import datasets
from sklearn.model_selection import train_test_split


class CausalDataset(Dataset):
    """Dataset for causal relationship detection."""
    
    def __init__(self, examples: List[Dict[str, Any]], tokenizer, max_length: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Create input text
        text1 = example["text1"]
        text2 = example["text2"]
        input_text = f"{text1} [SEP] {text2}"
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(example["label"], dtype=torch.long)
        
        return item


class CausalDataGenerator:
    """Generate synthetic causal training data."""
    
    def __init__(self):
        self.causal_templates = [
            "{cause} because {effect}",
            "{effect} because {cause}",
            "{cause} since {effect}",
            "{effect} since {cause}",
            "{cause} due to {effect}",
            "{effect} due to {cause}",
            "{cause} leads to {effect}",
            "{cause} causes {effect}",
            "{cause} results in {effect}",
            "As a result of {cause}, {effect}",
            "Because of {cause}, {effect}",
            "{cause}, therefore {effect}",
            "{cause}, thus {effect}",
            "{cause}, hence {effect}",
            "{cause}, consequently {effect}"
        ]
        
        self.non_causal_templates = [
            "{text1}. {text2}",
            "{text1} and {text2}",
            "{text1} but {text2}",
            "{text1} or {text2}",
            "{text1} while {text2}",
            "{text1} during {text2}",
            "{text1} before {text2}",
            "{text1} after {text2}",
            "{text1} with {text2}",
            "{text1} without {text2}"
        ]
        
        # Sample causes and effects
        self.causes = [
            "the heavy rain", "the economic crisis", "the new policy", "the temperature rise",
            "the lack of funding", "the technical failure", "the high demand", "the poor planning",
            "the strong winds", "the internet outage", "the power shortage", "the traffic jam"
        ]
        
        self.effects = [
            "the streets flooded", "many people lost jobs", "prices increased", "ice melted",
            "the project was cancelled", "the system crashed", "supplies ran out", "delays occurred",
            "trees fell down", "communication failed", "factories stopped", "people were late"
        ]
        
        self.neutral_statements = [
            "the weather was nice", "the meeting was productive", "the report was detailed",
            "the team worked hard", "the solution was effective", "the data was accurate",
            "the presentation was clear", "the results were positive", "the process was efficient",
            "the quality was high", "the performance improved", "the goals were achieved"
        ]
    
    def generate_causal_examples(self, num_examples: int = 1000) -> List[Dict[str, Any]]:
        """Generate causal relationship examples."""
        examples = []
        
        for _ in range(num_examples):
            # Randomly choose cause and effect
            cause = np.random.choice(self.causes)
            effect = np.random.choice(self.effects)
            
            # Randomly choose template
            template = np.random.choice(self.causal_templates)
            
            # Generate text
            text = template.format(cause=cause, effect=effect)
            
            # Split into text1 and text2
            if " because " in text:
                text1, text2 = text.split(" because ", 1)
            elif " since " in text:
                text1, text2 = text.split(" since ", 1)
            elif " due to " in text:
                text1, text2 = text.split(" due to ", 1)
            elif " leads to " in text:
                text1, text2 = text.split(" leads to ", 1)
            elif " causes " in text:
                text1, text2 = text.split(" causes ", 1)
            elif " results in " in text:
                text1, text2 = text.split(" results in ", 1)
            else:
                # Fallback split
                parts = text.split(", ")
                text1 = parts[0]
                text2 = ", ".join(parts[1:])
            
            examples.append({
                "text1": text1.strip(),
                "text2": text2.strip(),
                "label": 1,  # Causal
                "template": template
            })
        
        return examples
    
    def generate_non_causal_examples(self, num_examples: int = 1000) -> List[Dict[str, Any]]:
        """Generate non-causal relationship examples."""
        examples = []
        
        for _ in range(num_examples):
            # Randomly choose statements
            text1 = np.random.choice(self.neutral_statements)
            text2 = np.random.choice(self.neutral_statements)
            
            # Randomly choose template
            template = np.random.choice(self.non_causal_templates)
            
            # Generate text
            text = template.format(text1=text1, text2=text2)
            
            examples.append({
                "text1": text1,
                "text2": text2,
                "label": 0,  # Non-causal
                "template": template
            })
        
        return examples
    
    def generate_mixed_dataset(self, num_causal: int = 1000, num_non_causal: int = 1000) -> List[Dict[str, Any]]:
        """Generate mixed causal and non-causal examples."""
        causal_examples = self.generate_causal_examples(num_causal)
        non_causal_examples = self.generate_non_causal_examples(num_non_causal)
        
        # Combine and shuffle
        all_examples = causal_examples + non_causal_examples
        np.random.shuffle(all_examples)
        
        return all_examples


class CausalModelTrainer:
    """Train custom causality detection models using real data."""
    
    def __init__(self, model_name: str = "FacebookAI/roberta-large-mnli"):
        # Use the best model for causal detection
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with ignore_mismatched_sizes to handle label change
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2,  # Binary classification: causal vs non-causal
            ignore_mismatched_sizes=True
        )
        
        # Ensure the config is updated
        self.model.config.num_labels = 2
        
        # Add special tokens if needed
        if self.tokenizer.sep_token is None:
            self.tokenizer.add_special_tokens({'sep_token': '[SEP]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
    
    def load_real_training_data(self, data_path: str = "real_causal_data.json") -> List[Dict[str, Any]]:
        """Load real causal data from web collection."""
        try:
            with open(data_path, 'r') as f:
                real_examples = json.load(f)
            
            print(f"Loaded {len(real_examples)} real causal examples")
            return real_examples
            
        except FileNotFoundError:
            print(f"Real data file {data_path} not found. Run data collection first!")
            return []
    
    def create_balanced_dataset(self, real_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create balanced dataset with real causal and synthetic non-causal examples."""
        if not real_examples:
            print("No real examples available. Using synthetic data only.")
            return self._create_synthetic_dataset()
        
        # Real causal examples
        causal_examples = []
        for example in real_examples:
            if example.get('label') == 1:
                causal_examples.append({
                    "text1": example["text1"],
                    "text2": example["text2"],
                    "label": 1,
                    "source": example.get("source", "unknown")
                })
        
        # Generate non-causal examples (synthetic, but more realistic)
        non_causal_examples = self._generate_realistic_non_causal(causal_examples)
        
        # Balance the dataset
        min_size = min(len(causal_examples), len(non_causal_examples))
        balanced_examples = causal_examples[:min_size] + non_causal_examples[:min_size]
        
        # Shuffle
        np.random.shuffle(balanced_examples)
        
        print(f"Created balanced dataset: {len(balanced_examples)} examples")
        print(f"  Causal: {len(causal_examples[:min_size])}")
        print(f"  Non-causal: {len(non_causal_examples[:min_size])}")
        
        return balanced_examples
    
    def _generate_realistic_non_causal(self, causal_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate realistic non-causal examples based on real data patterns."""
        non_causal_examples = []
        
        # Extract real text patterns from causal examples
        real_texts = []
        for example in causal_examples:
            real_texts.extend([example["text1"], example["text2"]])
        
        # Generate non-causal pairs using real text
        for i in range(len(causal_examples)):
            # Randomly pair texts that don't have causal relationship
            text1 = np.random.choice(real_texts)
            text2 = np.random.choice(real_texts)
            
            # Ensure they're different and not causally related
            if text1 != text2 and not self._has_causal_indicators(text1 + " " + text2):
                non_causal_examples.append({
                    "text1": text1,
                    "text2": text2,
                    "label": 0,
                    "source": "synthetic_non_causal"
                })
        
        return non_causal_examples
    
    def _has_causal_indicators(self, text: str) -> bool:
        """Check if text contains causal indicators."""
        causal_words = [
            'because', 'since', 'due to', 'cause', 'causes', 'caused', 'causing',
            'leads to', 'lead to', 'led to', 'result', 'results', 'resulted',
            'therefore', 'thus', 'hence', 'consequently', 'as a result'
        ]
        
        text_lower = text.lower()
        return any(word in text_lower for word in causal_words)
    
    def _load_ecare_data(self) -> List[Dict[str, Any]]:
        """Load and format e-CARE dataset."""
        try:
            from datasets import load_dataset
            dataset = load_dataset("12ml/e-CARE")
            
            examples = []
            for split_name, split_data in dataset.items():
                for example in split_data:
                    # Convert to our format
                    formatted_example = {
                        "text1": example["premise"],
                        "text2": example["choice1"] if example["label"] == 0 else example["choice2"],
                        "label": 1,  # Causal relationship
                        "source": "ecare",
                        "split": split_name
                    }
                    examples.append(formatted_example)
            
            print(f"Loaded {len(examples)} e-CARE examples")
            return examples
            
        except Exception as e:
            print(f"Error loading e-CARE: {e}")
            return []
    
    def _load_copa_data(self) -> List[Dict[str, Any]]:
        """Load and format COPA dataset."""
        try:
            from datasets import load_dataset
            dataset = load_dataset("super_glue", "copa")
            
            examples = []
            for split_name, split_data in dataset.items():
                for example in split_data:
                    # Convert to our format
                    formatted_example = {
                        "text1": example["premise"],
                        "text2": example["choice1"] if example["label"] == 0 else example["choice2"],
                        "label": 1,  # Causal relationship
                        "source": "copa",
                        "split": split_name
                    }
                    examples.append(formatted_example)
            
            print(f"Loaded {len(examples)} COPA examples")
            return examples
            
        except Exception as e:
            print(f"Error loading COPA: {e}")
            return []
    
    def _create_balanced_ecare_copa_dataset(self, ecare_examples: List[Dict[str, Any]], 
                                           copa_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create balanced dataset with 70% e-CARE and 30% COPA."""
        if not ecare_examples and not copa_examples:
            return []
        
        # Target total dataset size
        target_size = 5000
        ecare_target_size = int(target_size * 0.7)  # 70%
        copa_target_size = int(target_size * 0.3)   # 30%
        
        # Sample from each dataset
        ecare_sample = []
        if ecare_examples:
            ecare_sample = np.random.choice(
                ecare_examples, 
                size=min(ecare_target_size, len(ecare_examples)), 
                replace=False
            ).tolist()
        
        copa_sample = []
        if copa_examples:
            copa_sample = np.random.choice(
                copa_examples, 
                size=min(copa_target_size, len(copa_examples)), 
                replace=False
            ).tolist()
        
        # Combine datasets
        combined_examples = ecare_sample + copa_sample
        
        # Shuffle
        np.random.shuffle(combined_examples)
        
        print(f"Created balanced dataset:")
        print(f"   e-CARE: {len(ecare_sample)} examples")
        print(f"   COPA: {len(copa_sample)} examples")
        print(f"   Total: {len(combined_examples)} examples")
        
        return combined_examples
    
    def _create_synthetic_dataset(self) -> List[Dict[str, Any]]:
        """Fallback to synthetic dataset if no real data available."""
        print("Using synthetic dataset as fallback")
        
        generator = CausalDataGenerator()
        causal_examples = generator.generate_causal_examples(500)
        non_causal_examples = generator.generate_non_causal_examples(500)
        
        return causal_examples + non_causal_examples
    
    def prepare_data(self, examples: List[Dict[str, Any]], test_size: float = 0.2) -> Tuple[Dataset, Dataset]:
        """Prepare training and test datasets."""
        # Split data
        train_examples, test_examples = train_test_split(
            examples, test_size=test_size, random_state=42, stratify=[ex["label"] for ex in examples]
        )
        
        # Create datasets
        train_dataset = CausalDataset(train_examples, self.tokenizer)
        test_dataset = CausalDataset(test_examples, self.tokenizer)
        
        return train_dataset, test_dataset
    
    def train_model(self, train_dataset: Dataset, test_dataset: Dataset, 
                   output_dir: str = "./causal_model", num_epochs: int = 3):
        """Train the causality detection model."""
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            learning_rate=2e-5,
            fp16=torch.cuda.is_available()
        )
        
        # Metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
            
            return {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )
        
        # Train model
        trainer.train()
        
        # Save model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        return trainer
    
    def evaluate_model(self, test_dataset: Dataset, model_path: str = None):
        """Evaluate the trained model."""
        if model_path:
            # Load trained model
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Create trainer for evaluation
        trainer = Trainer(model=self.model)
        
        # Evaluate
        results = trainer.evaluate(test_dataset)
        
        return results


def create_domain_specific_data() -> List[Dict[str, Any]]:
    """Create domain-specific causal data for prompt compression context."""
    domain_examples = [
        # Question-Answer causality
        {
            "text1": "The user asked about climate change causes",
            "text2": "The system provided information about greenhouse gases",
            "label": 1,
            "domain": "qa_causal"
        },
        {
            "text1": "The user requested a summary",
            "text2": "The system compressed the text",
            "label": 1,
            "domain": "qa_causal"
        },
        
        # Graph traversal causality
        {
            "text1": "The similarity threshold was high",
            "text2": "Fewer nodes were included in the traversal",
            "label": 1,
            "domain": "graph_causal"
        },
        {
            "text1": "The graph had many connected nodes",
            "text2": "The BFS traversal explored more paths",
            "label": 1,
            "domain": "graph_causal"
        },
        
        # Non-causal examples
        {
            "text1": "The system processed the query",
            "text2": "The user was satisfied with the response",
            "label": 0,
            "domain": "qa_non_causal"
        },
        {
            "text1": "The embedding model was loaded",
            "text2": "The graph was created",
            "label": 0,
            "domain": "system_non_causal"
        }
    ]
    
    return domain_examples


def train_custom_causal_model():
    """Main function to train custom causal model with e-CARE + COPA datasets."""
    print("Training Custom Causality Detection Model with e-CARE + COPA")
    
    # Step 1: Initialize trainer with best model
    print("Step 1: Initializing trainer with FacebookAI/roberta-large-mnli...")
    trainer = CausalModelTrainer("FacebookAI/roberta-large-mnli")
    
    # Step 2: Load e-CARE and COPA datasets
    print("Step 2: Loading e-CARE and COPA datasets...")
    ecare_examples = trainer._load_ecare_data()
    copa_examples = trainer._load_copa_data()
    
    if not ecare_examples and not copa_examples:
        print("Failed to load datasets. Using synthetic data.")
        training_examples = trainer._create_synthetic_dataset()
    else:
        # Step 3: Create balanced dataset (70% e-CARE, 30% COPA)
        print("Step 3: Creating balanced dataset (70% e-CARE, 30% COPA)...")
        training_examples = trainer._create_balanced_ecare_copa_dataset(ecare_examples, copa_examples)

    print(f"   Total training examples: {len(training_examples)}")

    # Step 4: Prepare data
    train_dataset, test_dataset = trainer.prepare_data(training_examples)

    # Step 5: Train model
    print("Step 4: Training model...")
    trained_trainer = trainer.train_model(train_dataset, test_dataset, "./custom_causal_model", num_epochs=3)
    
    # Step 6: Evaluate model
    print("Step 5: Evaluating model...")
    results = trainer.evaluate_model(test_dataset)
    
    print("Training complete!")
    print(f"   Accuracy: {results['eval_accuracy']:.4f}")
    print(f"   F1 Score: {results['eval_f1']:.4f}")
    print(f"   Precision: {results['eval_precision']:.4f}")
    print(f"   Recall: {results['eval_recall']:.4f}")
    
    # Step 7: Test with examples
    print("\nStep 6: Testing with real examples:")
    test_cases = [
        ("The heavy rain caused the streets to flood", "Traffic was disrupted"),
        ("The user asked about climate change", "The system provided information"),
        ("Because the similarity threshold was high", "Fewer nodes were included"),
        ("The meeting was scheduled for 3 PM", "The team prepared the presentation")
    ]
    
    for text1, text2 in test_cases:
        input_text = f"{text1} {trainer.tokenizer.sep_token} {text2}"
        encoding = trainer.tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=128
        )
        with torch.no_grad():
            outputs = trainer.model(**encoding)
            probs = torch.softmax(outputs.logits, dim=-1)
            confidence = float(probs[0][1].item())
        print(f"   '{text1[:30]}...' → '{text2[:30]}...' : {confidence:.3f}")
    
    return results


if __name__ == "__main__":
    train_custom_causal_model()
