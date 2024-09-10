import time
from typing import Tuple, List
import bittensor as bt
from logicnet.base.miner import BaseMinerNeuron
import logicnet
from logicnet.protocol import LogicSynapse, Information
from logicnet.miner.forward import solve
import traceback
import openai
import asyncio

class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        self.validator_logs = {}
        self.volume_per_validator = (
            logicnet.utils.volume_setting.get_rate_limit_per_validator()
        )

        # Batching related setup
        self.batch_size = 5  # Customize this for your system. 
        self.batch_timeout = 0.5  # Maximum time to wait before sending a batch (in seconds)
        self.batch_queue = []  # Queue to store batch requests

    async def handle_synapse(self, synapse: LogicSynapse):
        """Handle incoming LogicSynapse requests and batch them."""
        self.batch_queue.append(synapse)

        # If we reached the batch size, send the batch for solving
        if len(self.batch_queue) >= self.batch_size:
            await self.process_batch()

        # Alternatively, if there's not enough requests, wait for a timeout
        await asyncio.sleep(self.batch_timeout)

        if len(self.batch_queue) > 0:
            # If timeout triggered and there are still requests in queue, send them
            await self.process_batch()

    async def process_batch(self):
        """Processes the accumulated batch of synapse requests."""
        if not self.batch_queue:
            return

        # Extract batch and reset the queue
        batch = self.batch_queue[:]
        self.batch_queue = []

        # Prepare batch for VLLM processing (this passes a list of synapses)
        try:
            # Assuming solve function can handle batch of synapses
            results = await solve_batch(batch, self.openai_client, self.model_name)
            for result in results:
                bt.logging.info(f"Solved synapse: {result}")
        except Exception as e:
            bt.logging.error(f"Error processing batch: {e}")
            traceback.print_exc()

# Ensure to modify wherever handle_synapse is called to accommodate the new batching logic.
