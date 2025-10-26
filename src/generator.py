"""
Phase 3: Grounded Generation via PEFT-Tuned SLM

This module implements the generation component using a small language model
fine-tuned with LoRA (Low-Rank Adaptation) for parameter-efficient training.
The model generates grounded answers with citations to source chunks.
"""

import torch
from typing import List, Dict, Optional, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)
from datasets import Dataset
import json

from .retriever import RetrievalResult


class GroundedGenerator:
    """
    Implements grounded answer generation using a PEFT-tuned Small Language Model.
    
    The generator takes retrieved chunks as context and produces answers with
    citations, preventing hallucination by grounding responses in the provided evidence.
    """
    
    def __init__(self, config: Dict, training_mode: bool = False):
        """
        Initialize the GroundedGenerator with an SLM and optional LoRA configuration.
        
        Args:
            config: Configuration dictionary containing generation settings
            training_mode: If True, prepare model for training; if False, for inference
        """
        self.config = config
        gen_config = config.get('generation', {})
        
        # Model configuration
        self.model_name = gen_config.get('model_name', 'microsoft/Phi-3-mini-4k-instruct')
        self.max_length = gen_config.get('max_length', 512)
        self.temperature = gen_config.get('temperature', 0.7)
        self.top_p = gen_config.get('top_p', 0.9)
        self.num_beams = gen_config.get('num_beams', 4)
        
        # LoRA configuration
        self.use_lora = gen_config.get('use_lora', True)
        self.lora_r = gen_config.get('lora_r', 16)
        self.lora_alpha = gen_config.get('lora_alpha', 32)
        self.lora_dropout = gen_config.get('lora_dropout', 0.05)
        self.lora_target_modules = gen_config.get('lora_target_modules', ['q_proj', 'v_proj'])
        
        # Prompt template
        self.prompt_template = gen_config.get('prompt_template', 
            "Context: {context}\n\nQuestion: {question}\n\nBased on the provided context, "
            "answer the question concisely and provide the citation to the source chunk.\nAnswer: "
        )
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        print(f"Loading tokenizer for {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        print(f"Loading model: {self.model_name}")
        if training_mode:
            # Load with reduced precision for training
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Apply LoRA if enabled
            if self.use_lora:
                self.model = self._apply_lora(self.model)
        else:
            # Load for inference
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
        
        self.model.eval()
        
    def _apply_lora(self, model):
        """
        Apply LoRA (Low-Rank Adaptation) to the model for parameter-efficient fine-tuning.
        
        Args:
            model: The base model to apply LoRA to
            
        Returns:
            PEFT model with LoRA adapters
        """
        print("Applying LoRA configuration...")
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.lora_r,  # Rank of the low-rank matrices
            lora_alpha=self.lora_alpha,  # Scaling factor
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.lora_target_modules  # Apply to attention projection layers
        )
        
        # Prepare model for training (gradient checkpointing, etc.)
        model = prepare_model_for_kbit_training(model)
        
        # Get PEFT model
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        return model
    
    def generate(self, question: str, retrieved_chunks: List[RetrievalResult]) -> Tuple[str, str]:
        """
        Generate a grounded answer with citation from retrieved chunks.
        
        Args:
            question: The input question
            retrieved_chunks: List of RetrievalResult objects from the retriever
            
        Returns:
            Tuple of (answer, citation) where citation is the chunk ID
        """
        # Format context from retrieved chunks
        context = self._format_context(retrieved_chunks)
        
        # Create prompt
        prompt = self.prompt_template.format(context=context, question=question)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Generate answer
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (remove the prompt)
        answer = generated_text[len(prompt):].strip()
        
        # Parse answer and citation
        answer, citation = self._parse_answer_citation(answer, retrieved_chunks)
        
        return answer, citation
    
    def _format_context(self, retrieved_chunks: List[RetrievalResult]) -> str:
        """
        Format retrieved chunks into a context string for the prompt.
        
        Args:
            retrieved_chunks: List of retrieved chunks with scores
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for result in retrieved_chunks:
            chunk = result.chunk
            # Format: [chunk_id] text content
            context_parts.append(f"[{chunk.chunk_id}] {chunk.text}")
        
        return "\n\n".join(context_parts)
    
    def _parse_answer_citation(self, generated_text: str, 
                               retrieved_chunks: List[RetrievalResult]) -> Tuple[str, str]:
        """
        Parse the generated text to extract answer and citation.
        
        The model is trained to output: "answer [citation: chunk_id]"
        If no explicit citation is found, default to the top-ranked chunk.
        
        Args:
            generated_text: The generated answer text
            retrieved_chunks: List of retrieved chunks
            
        Returns:
            Tuple of (answer, citation)
        """
        # Look for citation pattern [citation: chunk_id] or [chunk_id]
        import re
        
        citation_pattern = r'\[(?:citation:\s*)?([^\]]+)\]'
        matches = re.findall(citation_pattern, generated_text)
        
        if matches:
            # Extract the last citation found
            citation = matches[-1].strip()
            # Remove citation from answer
            answer = re.sub(citation_pattern, '', generated_text).strip()
        else:
            # Default to top-ranked chunk
            answer = generated_text
            citation = retrieved_chunks[0].chunk.chunk_id if retrieved_chunks else "unknown"
        
        return answer, citation
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """
        Fine-tune the model on a DocQA dataset using LoRA.
        
        Args:
            train_dataset: Training dataset with question-answer-citation triplets
            eval_dataset: Optional evaluation dataset
        """
        training_config = self.config.get('generation', {})
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.get('training', {}).get('output_dir', 'models/checkpoints'),
            num_train_epochs=training_config.get('num_epochs', 3),
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
            learning_rate=training_config.get('learning_rate', 2e-4),
            weight_decay=training_config.get('weight_decay', 0.01),
            warmup_ratio=training_config.get('warmup_ratio', 0.1),
            logging_dir=self.config.get('training', {}).get('logging_dir', 'logs'),
            logging_steps=self.config.get('training', {}).get('logging_steps', 100),
            save_steps=self.config.get('training', {}).get('save_steps', 500),
            eval_steps=self.config.get('training', {}).get('eval_steps', 500),
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=self.config.get('training', {}).get('save_total_limit', 3),
            load_best_model_at_end=self.config.get('training', {}).get('load_best_model_at_end', True),
            fp16=torch.cuda.is_available() and self.config.get('training', {}).get('fp16', True),
            gradient_checkpointing=training_config.get('gradient_checkpointing', True),
            report_to=["tensorboard"],
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save the model
        trainer.save_model()
        print(f"Model saved to {training_args.output_dir}")
    
    def save_model(self, path: str):
        """
        Save the model (and LoRA adapters if applicable).
        
        Args:
            path: Directory to save the model
        """
        if self.use_lora:
            # Save only LoRA adapters (much smaller)
            self.model.save_pretrained(path)
        else:
            # Save full model
            self.model.save_pretrained(path)
        
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load a fine-tuned model (with LoRA adapters if applicable).
        
        Args:
            path: Directory containing the saved model
        """
        if self.use_lora:
            # Load LoRA adapters
            print(f"Loading LoRA adapters from {path}")
            self.model = PeftModel.from_pretrained(self.model, path)
        else:
            # Load full fine-tuned model
            print(f"Loading model from {path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
        
        self.model.eval()
        print("Model loaded successfully")
