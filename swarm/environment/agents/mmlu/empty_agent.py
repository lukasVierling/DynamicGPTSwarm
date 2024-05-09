from swarm.graph import Graph
from swarm.environment.operations.mmlu.empty_node import EmptyNode
from swarm.environment.agents.agent_registry import AgentRegistry


@AgentRegistry.register('EmptyAgent')
class EmptyAgent(Graph):

    def __init__(self, domain, model_name=None, meta_prompt=False):
        super().__init__(domain, model_name, meta_prompt)

    def build_graph(self):


        empty_node = EmptyNode(domain=self.domain, id="empty_node")

        self.input_nodes = [empty_node]
        self.output_nodes = [empty_node]

        self.add_node(empty_node)

