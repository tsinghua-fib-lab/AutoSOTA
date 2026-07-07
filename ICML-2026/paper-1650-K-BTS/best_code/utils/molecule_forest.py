import pickle
from graphviz import Digraph


class EvolutionNode:
    def __init__(self, smiles, score, qed, sa, scaffold, step_id, parent_smiles=None, action_taken=None):
        self.smiles = smiles
        self.score = score
        self.qed = qed
        self.sa = sa
        self.scaffold = scaffold
        self.step_id = step_id
        self.parent_smiles = parent_smiles
        self.action_taken = action_taken
        self.children = []


class MolecularForest:
    def __init__(self):
        self.all_nodes = {}
        self.current_step = 0

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def add_root(self, smiles, score, qed, sa, scaffold):
        if smiles in self.all_nodes: return
        node = EvolutionNode(smiles, score, qed, sa, scaffold, step_id=0, parent_smiles=None)
        self.all_nodes[smiles] = node

    def add_molecule(self, smiles, score, qed, sa, scaffold, parent_smiles, action):
        if smiles in self.all_nodes: return

        self.current_step += 1

        node = EvolutionNode(smiles, score, qed, sa, scaffold,
                             step_id=self.current_step,
                             parent_smiles=parent_smiles,
                             action_taken=action)

        self.all_nodes[smiles] = node

        if parent_smiles in self.all_nodes:
            self.all_nodes[parent_smiles].children.append(node)

    def generate_svg(self, filename="evolution", max_step=None):
        limit = max_step if max_step is not None else float('inf')
        dot = Digraph(comment='Molecular Evolution Forest', format='svg')
        dot.attr(rankdir='LR', fontname='Helvetica', fontsize='10', splines='ortho')
        dot.attr('node', shape='record', style='filled', fontname='Helvetica')

        nodes_to_draw = set()
        for s, node in self.all_nodes.items():
            if 0 < node.step_id <= limit:
                nodes_to_draw.add(s)
                if node.parent_smiles: nodes_to_draw.add(node.parent_smiles)
            if node.step_id == 0 and any(child.step_id <= limit for child in node.children):
                nodes_to_draw.add(s)

        for smiles in nodes_to_draw:
            node = self.all_nodes[smiles]
            smi_display = (smiles[:15] + '...') if len(smiles) > 15 else smiles
            tag = "Root" if node.step_id == 0 else f"Step {node.step_id}"

            label = f"{{ {tag} | {smi_display} | Score: {node.score:.2f} }}"
            fillcolor = "#BBDEFB" if node.step_id == 0 else ("#C8E6C9" if node.score < -9.5 else "#F5F5F5")
            dot.node(smiles, label=label, fillcolor=fillcolor)

            if node.parent_smiles and node.parent_smiles in nodes_to_draw:
                action_str = node.action_taken

                if action_str and "[SCAFFOLD]" in str(action_str):
                    edge_color = "#9b59b6" # purple
                    edge_style = "dashed"
                    edge_width = "2.0"
                    label_text = " Scaffold Hop "

                elif action_str is None or action_str == "Exploration" or action_str == "":
                    edge_color = "#e67e22"  # orange
                    edge_style = "dotted"
                    edge_width = "1.5"
                    label_text = " LLM Exploration "
                else:
                    edge_color = "#455A64"  # grey
                    edge_style = "solid"
                    edge_width = "1.0"
                    label_text = f" {str(action_str).replace('[26*]', '').replace('>>', ' → ')} "

                dot.edge(node.parent_smiles, smiles, label=label_text,
                         color=edge_color, style=edge_style, penwidth=edge_width,
                         fontcolor=edge_color, fontsize='9')

        dot.render(filename, cleanup=True)
        return f"{filename}.svg"