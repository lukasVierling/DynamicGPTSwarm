
from swarm.graph import Node
from swarm.llm.format import Message


class CrosswordsOperation(Node):
    async def llm_query_with_cache(self, prompt):
        cache = self.memory.query_by_id("cache")
        if len(cache) == 0:
            cache = {}
            self.memory.add("cache", cache)
        else:
            cache = cache[0]
        if not prompt in cache.keys():    
            cache[prompt] = await self.llm.agen([Message(role="user", content=prompt)], temperature=0.0001)#TODO change this value was originally 0 but problems with gemma model
        return cache[prompt]