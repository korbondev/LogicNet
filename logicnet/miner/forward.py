from logicnet.protocol import LogicSynapse
import openai
import bittensor as bt
import traceback
import re
from typing import List

async def solve(
    synapse: LogicSynapse, openai_client: openai.AsyncOpenAI, model: str
) -> LogicSynapse:
    """Original solve function for single synapse requests."""
    try:
        bt.logging.info(f"Received synapse: {synapse}")
        logic_question: str = synapse.logic_question
        messages = [
            {"role": "user", "content": logic_question},
        ]
        response = await openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000,  # Adjust based on your model's capacity
            temperature=0.7,
        )
        synapse.answer = response.choices[0].message["content"]
        return synapse

    except Exception as e:
        bt.logging.error(f"Error solving synapse: {e}")
        traceback.print_exc()
        return synapse


async def solve_batch(
    synapses: List[LogicSynapse], openai_client: openai.AsyncOpenAI, model: str
) -> List[LogicSynapse]:
    """New function for batch synapse requests."""
    try:
        # Log the incoming batch
        bt.logging.info(f"Received synapse batch of size: {len(synapses)}")
        
        # Extract logic questions from each synapse in the batch
        messages = [{"role": "user", "content": synapse.logic_question} for synapse in synapses]

        # Send all questions as a batch to the VLLM model
        response = await openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000,  # Adjust based on your model's capacity
            temperature=0.7,
        )

        # Process and return results back to the calling function
        results = []
        for synapse, completion in zip(synapses, response.choices):
            synapse.answer = completion.message["content"]
            results.append(synapse)

        return results

    except Exception as e:
        bt.logging.error(f"Error solving batch: {e}")
        traceback.print_exc()
        return []
