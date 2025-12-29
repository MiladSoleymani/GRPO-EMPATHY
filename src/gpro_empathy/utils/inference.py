import yaml
from typing import Dict, Any, Optional, List
from vllm import SamplingParams

from ..training.grpo_trainer import GPROEmpathyTrainer
from ..data.dataset_loader import get_system_prompt


class EmpathyInference:
    """Inference utility for empathy-trained models."""
    
    def __init__(self, config_path: Optional[str] = None, lora_path: Optional[str] = None):
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
        
        self.trainer = GPROEmpathyTrainer(
            model_name=self.config['model']['name'],
            max_seq_length=self.config['model']['max_seq_length'],
            lora_rank=self.config['model']['lora_rank'],
            load_in_4bit=self.config['model']['load_in_4bit'],
            fast_inference=self.config['model']['fast_inference'],
            gpu_memory_utilization=self.config['model']['gpu_memory_utilization'],
        )
        
        self.lora_request = None
        if lora_path:
            self.lora_request = self.trainer.load_lora(lora_path)
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration if no config file provided."""
        return {
            'model': {
                'name': "meta-llama/meta-Llama-3.1-8B-Instruct",
                'max_seq_length': 1024,
                'lora_rank': 32,
                'load_in_4bit': True,
                'fast_inference': True,
                'gpu_memory_utilization': 0.6,
            },
            'inference': {
                'temperature': 0.8,
                'top_p': 0.95,
                'max_tokens': 1024,
            }
        }
    
    def create_empathy_prompt(self, user_message: str, include_system: bool = True) -> str:
        """Create a properly formatted prompt for empathy generation."""
        if include_system:
            messages = [
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": user_message},
            ]
        else:
            messages = [
                {"role": "user", "content": user_message}
            ]

        return self.trainer.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    
    def generate_empathetic_response(
        self,
        user_message: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        include_system: bool = True,
    ) -> str:
        """Generate an empathetic response to a user message."""
        prompt = self.create_empathy_prompt(user_message, include_system=include_system)
        
        temperature = temperature or self.config['inference']['temperature']
        top_p = top_p or self.config['inference']['top_p']
        max_tokens = max_tokens or self.config['inference']['max_tokens']
        
        response = self.trainer.generate_sample(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            lora_request=self.lora_request,
        )
        
        return response
    
    def batch_generate(
        self,
        user_messages: List[str],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        include_system: bool = True,
    ) -> List[str]:
        """Generate empathetic responses for multiple user messages."""
        prompts = [self.create_empathy_prompt(msg, include_system) for msg in user_messages]
        
        temperature = temperature or self.config['inference']['temperature']
        top_p = top_p or self.config['inference']['top_p']
        max_tokens = max_tokens or self.config['inference']['max_tokens']
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        
        outputs = self.trainer.model.fast_generate(
            prompts,
            sampling_params=sampling_params,
            lora_request=self.lora_request,
        )
        
        return [output.outputs[0].text for output in outputs]
    
    def extract_answer_from_response(self, response: str) -> str:
        """Extract the answer content from XML-formatted response."""
        import re
        answer_pattern = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.DOTALL | re.IGNORECASE)
        match = answer_pattern.search(response)
        return match.group(1).strip() if match else response.strip()


def load_inference_model(config_path: str, lora_path: str) -> EmpathyInference:
    """Convenience function to load inference model with config and LoRA."""
    return EmpathyInference(config_path=config_path, lora_path=lora_path)