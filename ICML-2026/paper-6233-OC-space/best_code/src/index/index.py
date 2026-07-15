import heapq
import time
import veritas
import numpy as np
import util

from itertools import chain
from depth_first_ocenum import enumerate_ocs, OCSBatch, OCSFinished

class IndexNode:
    def __init__(self, box, value=None):
        self.box = box
        self.ymax = None
        self.ymin = None
        self.children = []
        self.value = value
        self.pos_boxes = None
        self.neg_boxes = None
        self.pos_y = None
        self.neg_y = None

    def is_leaf(self):
        return len(self.children) == 0

class Index:
    def __init__(self, at: veritas.AddTree, featmap: np.ndarray):
        self.at = at
        self.featmap = featmap
        self.active_map = np.where(self.featmap != -1)[0]
        self.n_features = np.count_nonzero(self.featmap != -1)
        self.pos_count = 0
        self.neg_count = 0

    def _init_index(self):
        pass

    def store(self, boxes_buffer, outvalues_buffer):
        pass

    def to_arrays(self):
        pass


class PosNegIndex(Index):
    def __init__(self, at: veritas.AddTree, featmap: np.ndarray):
        super().__init__(at, featmap)
        self._init_index()

    def _init_index(self):
        infinite_box = np.empty((2, self.n_features), dtype=np.float32)
        infinite_box[0, :] = -np.inf
        infinite_box[1, :] = np.inf

        self.index = [IndexNode(infinite_box)]

    def store(self, boxes_buffer, outvalues_buffer):
        for j in range(boxes_buffer.shape[0]):
            self.index[0].children.append(IndexNode(boxes_buffer[j], outvalues_buffer[j]))

    def to_arrays(self):
        node = self.index[0]
        pos_boxes = []
        neg_boxes = []
        pos_y = []
        neg_y = []

        for child in node.children:
            if child.value > 0:
                pos_boxes.append(child.box)
                pos_y.append(child.value)
                self.pos_count += 1
            else:
                neg_boxes.append(child.box)
                neg_y.append(child.value)
                self.neg_count += 1

            node.ymin = child.value if node.ymin is None else min(node.ymin, child.value)
            node.ymax = child.value if node.ymax is None else max(node.ymax, child.value)


        node.pos_boxes = np.stack(pos_boxes, axis=0).astype(np.float32) if pos_boxes else np.empty((0, 2, self.n_features), dtype=np.float32)
        node.neg_boxes = np.stack(neg_boxes, axis=0).astype(np.float32) if neg_boxes else np.empty((0, 2, self.n_features), dtype=np.float32)
        node.pos_y = np.array(pos_y, dtype=np.float32) if pos_y else np.empty((0,), dtype=np.float32)
        node.neg_y = np.array(neg_y, dtype=np.float32) if neg_y else np.empty((0,), dtype=np.float32)
        node.children=None

    def get_boxes_at_index(self, label):
        if label == 1:
            return self.index[0].pos_boxes
        else:
            return self.index[0].neg_boxes

    

    def find_closest_adversarial_example(self, example, target_label):
        # Returns the distance to the closest adversarial example in the index for the given example and target label
        boxes = self.get_boxes_at_index(target_label)
        if len(boxes) == 0:
            return np.inf

        # Compute distance to closest box
        return util.dist_to_closest_box(example, boxes, self.active_map)

    
    

class RootboxIndex(Index):
    def __init__(self, at: veritas.AddTree, featmap: np.ndarray):
        super().__init__(at, featmap)
        self._define_splits()
        self._calculate_strides()
        self._build_index()

        # Additional initialization code can go here

    def _define_splits(self):
        # Code to find the rootbox splits in the ensemble
        splits = {}
        for t in self.at:
            root_split = t.get_split(0) # Root box is at index 0
            if root_split.feat_id not in splits:
                splits[root_split.feat_id] = set()
            splits[root_split.feat_id].add(np.float32(root_split.split_value))

        self.splits = {k: np.array(sorted(list(v))) for k, v in sorted(splits.items())}
        self.nb_rootboxes = np.prod([len(v) + 1 for v in self.splits.values()])

    def _calculate_strides(self):
        self.strides = {}
        for feat_id in self.splits.keys():
            self.strides[feat_id] = int(np.prod([len(self.splits[k]) + 1 for k in self.splits.keys() if k > feat_id]))

    def _build_index(self):
        self.nb_rootboxes = np.prod([len(v) + 1 for v in self.splits.values()])
        self.index = []

        for i in range(self.nb_rootboxes):
            box = np.empty((2, self.n_features), dtype=np.float32)
            box[0, :] = -np.inf
            box[1, :] = np.inf

            for feat_id, values in self.splits.items():
                j = self.featmap[feat_id]
                interval_size = self.strides[feat_id] # How many combinations of other features (with larger feat_id) exist.
                idx = (i // interval_size) % (len(values) + 1)
                box[0, j] = values[int(idx) - 1] if idx > 0 else -np.inf
                box[1, j] = values[int(idx)] if idx < len(values) else np.inf

            node = IndexNode(box)
            self.index.append(node)
    
    def store(self, boxes_buffer, outvalues_buffer):
        indices = self._find_indices(boxes_buffer)
        
        for j, i in enumerate(indices):
            self.index[i].children.append(IndexNode(boxes_buffer[j], outvalues_buffer[j]))
        
    def to_arrays(self):
        def _collapse_rootboxes(node):
            if node.children is not None:
                pos_boxes = []
                neg_boxes = []
                pos_y = []
                neg_y = []

                for child in node.children:
                    if child.value > 0:
                        pos_boxes.append(child.box)
                        pos_y.append(child.value)
                        self.pos_count += 1
                    else:
                        neg_boxes.append(child.box)
                        neg_y.append(child.value)
                        self.neg_count += 1

                    node.ymin = child.value if node.ymin is None else min(node.ymin, child.value)
                    node.ymax = child.value if node.ymax is None else max(node.ymax, child.value)

                # Allocate arrays once
                if pos_boxes:
                    pos_boxes = np.stack(pos_boxes).astype(np.float32)
                else:
                    pos_boxes = np.empty((0, 2, self.n_features), dtype=np.float32)

                if neg_boxes:
                    neg_boxes = np.stack(neg_boxes).astype(np.float32)
                else:
                    neg_boxes = np.empty((0, 2, self.n_features), dtype=np.float32)

                # Replace children with arrays
                node.children = None
                node.pos_boxes = pos_boxes
                node.neg_boxes = neg_boxes
                node.pos_y = np.array(pos_y, dtype=np.float32)
                node.neg_y = np.array(neg_y, dtype=np.float32)

        for rootbox in self.index:
            _collapse_rootboxes(rootbox)

    
    def find_index(self, iput: np.ndarray):
        # Finds the rootbox index that a given box/instance belongs to
        if iput.ndim == 1:
            x = iput[self.active_map]
        else:
            x = iput[0, :]

        index = 0
    
        for feat_id, values in self.splits.items():
            count = np.searchsorted(values, x[self.featmap[feat_id]], side='right')
            index += count * self.strides[feat_id]
        return int(index)
    
    def _find_indices(self, boxes):
        # boxes: (N, D)
        X = boxes[:, 0, :]

        indices = np.zeros(X.shape[0], dtype=np.int64)

        for feat_id, split_vals in self.splits.items():
            col = X[:, self.featmap[feat_id]]
            counts = np.searchsorted(split_vals, col, side="right")
            indices += counts * self.strides[feat_id]

        return indices

    def get_boxes_at_index(self, index, label):
        if label == 1:
            return self.index[index].pos_boxes
        else:
            return self.index[index].neg_boxes
    
    def rootbox_generator(self, x):
        start = self.find_index(x)

        visited = set()
        heap = [start]

        while heap:
            idx = heapq.heappop(heap)
            if idx in visited:
                continue
            visited.add(idx)

            yield idx

            for nb_idx in self._neighbors(idx, x):
                if nb_idx not in visited:
                    heapq.heappush(heap, nb_idx)
        
    def _neighbors(self, index, x):
        for feat_id, values in self.splits.items():
            j = self.featmap[feat_id]
            interval_size = self.strides[feat_id]
            current_idx = (index // interval_size) % (len(values) + 1)

            # Check lower neighbor
            if current_idx > 0:
                nb_index = index - interval_size
                # split_value = values[current_idx - 1]
                # cost = max(0.0, split_value - x[j])
                # yield (nb_index, cost)
                yield nb_index

            # Check upper neighbor
            if current_idx < len(values):
                nb_index = index + interval_size
                # split_value = values[current_idx]
                # cost = max(0.0, x[j] - split_value)
                # yield (nb_index, cost)
                yield nb_index

    
    
    def get_index_node(self, box):
        # Returns the leaf node in the index that overlaps with the given box, or None if no such node exists. 
        # Assumes that the tree is properly built and that there are no overlapping boxes at the same level of the tree.
        def find_node(index, box):
            for node in index:
                if node.pos_boxes is not None and node.neg_boxes is not None:
                    if util.overlaps(node.box, box):
                        return node
                elif util.overlaps(node.box, box):
                    return find_node(node.children, box)
        return find_node(self.index, box)


    def get_boxes_same_bin(self, x, label):
        box = x
        if x.ndim == 1:
            b = np.array([x[f] for f in np.where(self.featmap != -1)[0]])
            box = np.array([b, np.nextafter(b, +np.inf, dtype=np.float32)], dtype=np.float32) # The guard is added here so that the box of a single example does not land on exactly a split, then no boxes are returned and this errors.
        
        leaf_node = self.get_index_node(box)
        if leaf_node is not None:
            return leaf_node.pos_boxes if label == 1 else leaf_node.neg_boxes

    def find_closest_adversarial_example(self, example, target_label):
        delta_lo = np.inf

        for idx in self.rootbox_generator(example):
            if util.linf(example, self.index[idx].box, self.active_map) < delta_lo:
                delta = util.dist_to_closest_box(example, self.get_boxes_at_index(idx, target_label), self.active_map)
                delta_lo = min(delta_lo, delta)
            else:
                continue
        
        return delta_lo

    def is_robust(self, example, label, l_inf, l_1):
        # TODO this can for sure be optimized more by generalizing the rootbox generator.
        print("Checking robustness of example", example, "with label", label, "and l_inf", l_inf, "and l_1", l_1)
        for idx in self.rootbox_generator(example):
            if util.linf(example, self.index[idx].box, self.active_map) >= l_inf or util.l1(example, self.index[idx].box, self.active_map) >= l_1:
                continue
            boxes = self.get_boxes_at_index(idx, label)
            l_inf_dist = util.dist_to_boxes(example, boxes, self.active_map)
            l_1_dist = util.dist_to_boxes_l1(example, boxes, self.active_map)
            if (np.any((l_inf_dist < l_inf) & (l_1_dist < l_1))):
                return False
        return True

class OCTreeIndex(Index):
    def __init__(self, at: veritas.AddTree, featmap: np.ndarray, depth: int = None):
        super().__init__(at, featmap)
        self.depth = depth if depth is not None else max(0, min(len(at) - 2, 8))
        self.init_index()
        self.build_index(self.depth)

    def init_index(self):
        infinite_box = np.empty((2, self.n_features), dtype=np.float32)
        infinite_box[0, :] = -np.inf
        infinite_box[1, :] = np.inf
        self.index = [IndexNode(infinite_box)]
        

    def build_index(self, depth):
        for d in range(1, depth + 1):
            sub_at = util.get_sub_addtree(self.at, d)
            for batch in enumerate_ocs(sub_at, 1000 * 8192, feat_map=self.featmap): # Use featmap from the original AddTree to ensure that the boxes in the index are defined in the same feature space.
                if isinstance(batch, OCSFinished):
                    break
                elif isinstance(batch, OCSBatch):
                    boxes_buffer = batch.boxes
                    outvalues_buffer = batch.values
                    self.bulk_insert(self.index, boxes_buffer, outvalues_buffer)

    def bulk_insert(self, node_list, boxes, values):
        """
        node_list : list[OCNode]
        boxes     : (N, 2, D)
        values    : (N,)
        """
        if boxes.shape[0] == 0:
            return

        used = np.zeros(boxes.shape[0], dtype=bool)

        # Try to place boxes into existing nodes
        for node in node_list:
            mask = util.overlaps_batch(node.box, boxes)
            mask &= ~used  # Only process boxes not yet assigned to any node
            if not mask.any():
                continue

            used |= mask

            # Update node's min/max values with values that overlap this node
            vals = values[mask]
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            if node.ymin is None or node.ymax is None:
                node.ymin = vmin
                node.ymax = vmax
            else:
                if vmin < node.ymin:
                    node.ymin = vmin
                if vmax > node.ymax:
                    node.ymax = vmax

            # Recurse once per node, not once per box
            self.bulk_insert(
                node.children,
                boxes[mask],
                values[mask],
            )

        # Boxes that didn't overlap any existing node → new nodes
        new_idx = np.where(~used)[0]
        for i in new_idx:
            n = IndexNode(boxes[i])
            n.value = values[i]
            node_list.append(n)
            
    def store(self, boxes_buffer, outvalues_buffer):
        self.bulk_insert(self.index, boxes_buffer, outvalues_buffer)

    def to_arrays(self):
        def collapse_leaves(node, n_features):
            """
            Converts lowest-level children of `node` into pos/neg arrays.
            Modifies the tree in place.
            """
            if not node.children:
                print("Warning: node with no children found during collapse_leaves. This should not happen if the tree is properly built.")
                print(node.box)
                return False  # nothing to do here

            # Check if all children are leaves
            if all(child.is_leaf() for child in node.children):
                pos_boxes = []
                neg_boxes = []
                pos_y = []
                neg_y = []

                for child in node.children:
                    if child.value > 0:
                        pos_boxes.append(child.box)
                        pos_y.append(child.value)
                        self.pos_count += 1
                    else:
                        neg_boxes.append(child.box)
                        neg_y.append(child.value)
                        self.neg_count += 1

                # Allocate arrays once
                if pos_boxes:
                    pos_boxes = np.stack(pos_boxes).astype(np.float32)
                else:
                    pos_boxes = np.empty((0, 2, n_features), dtype=np.float32)

                if neg_boxes:
                    neg_boxes = np.stack(neg_boxes).astype(np.float32)
                else:
                    neg_boxes = np.empty((0, 2, n_features), dtype=np.float32)

                # Replace children with arrays
                node.children = None
                node.pos_boxes = pos_boxes
                node.neg_boxes = neg_boxes
                node.pos_y = np.array(pos_y, dtype=np.float32)
                node.neg_y = np.array(neg_y, dtype=np.float32)

                return True

            # Otherwise recurse deeper
            for child in node.children:
                collapse_leaves(child, n_features)

            return False
        
        for node in self.index:
            collapse_leaves(node, self.n_features)

    def get_index_node(self, box):
        # Returns the leaf node in the index that overlaps with the given box, or None if no such node exists. 
        # Assumes that the tree is properly built and that there are no overlapping boxes at the same level of the tree.
        def find_node(index, box):
            for node in index:
                if node.pos_boxes is not None and node.neg_boxes is not None:
                    if util.overlaps(node.box, box):
                        return node
                elif util.overlaps(node.box, box):
                    return find_node(node.children, box)
        return find_node(self.index, box)


    def get_boxes_same_bin(self, x, label):
        box = x
        if x.ndim == 1:
            b = np.array([x[f] for f in np.where(self.featmap != -1)[0]])
            box = np.array([b, np.nextafter(b, +np.inf, dtype=np.float32)], dtype=np.float32) # The guard is added here so that the box of a single example does not land on exactly a split, then no boxes are returned and this errors.
        
        leaf_node = self.get_index_node(box)
        if leaf_node is not None:
            return leaf_node.pos_boxes if label == 1 else leaf_node.neg_boxes

    def generator_boxes(self, condition=lambda box, ymin, ymax: True, label=None):
        # Only use these with monotonic conditions such as overlaps or distance...
        def _generator(index):
            for node in index:
                if condition(node.box, node.ymin, node.ymax):
                    if node.pos_boxes is not None and node.neg_boxes is not None:
                        if (label == 1 or label is None) and node.pos_boxes.shape[0] > 0:
                            yield node.pos_boxes, node.pos_y 
                        if (label == 0 or label is None) and node.neg_boxes.shape[0] > 0:
                            yield node.neg_boxes, node.neg_y
                    else:
                        yield from _generator(node.children)
        return _generator(self.index)

    def find_closest_adversarial_example(self, example, target_label):
        delta_lo = util.dist_to_closest_box(example, self.get_boxes_same_bin(example, target_label), self.active_map) # Distance to closest adversarial example in the same bin as the example
        if target_label:
            condition = lambda box, ymin, ymax: (ymax is None or ymax > 0) and util.linf(example, box, self.active_map) < delta_lo
        else:
            condition = lambda box, ymin, ymax: (ymin is None or ymin < 0) and util.linf(example, box, self.active_map) < delta_lo
        
        for boxes, values in self.generator_boxes(condition=condition, label=target_label):
            assert boxes.shape[0] > 0, "Generator should never yield empty list of boxes"
            delta = util.dist_to_closest_box(example, boxes, self.active_map)
            delta_lo = min(delta_lo, delta)
        
        return delta_lo

    def find_unfair_boxes(self, protected_idx, timeout):
        start_time = time.time()
        violating_regions = []

        if self.pos_count == 0 or self.neg_count == 0 or len(self.featmap) <= protected_idx or self.featmap[protected_idx] == -1:
            return violating_regions

        smallest_class = 1 if self.pos_count < self.neg_count else 0
        boxes_to_compare = self.generator_boxes(label=smallest_class)
        for box in chain.from_iterable(boxes for boxes, _ in boxes_to_compare):
            if time.time() - start_time >= timeout:
                break
            
            if box[0, self.featmap[protected_idx]] == float('-inf') and box[1, self.featmap[protected_idx]] == float('inf'): # if the box doesn't allow changing the protected feature, skip it
                continue

            relaxed_box = util.relax_box(box, self.featmap[protected_idx])

            overlapping_boxes = self.generator_boxes(label=1 - smallest_class, condition=lambda b, ymin, ymax: util.overlaps(relaxed_box, b))

            for overlapping_box in chain.from_iterable(boxes for boxes, _ in overlapping_boxes):
                relaxed_overlapping_box = util.relax_box(overlapping_box, self.featmap[protected_idx])
                if util.overlaps(relaxed_box, relaxed_overlapping_box):
                    intersection = util.intersect(relaxed_box, relaxed_overlapping_box)
                    violating_regions.append(intersection)
        return violating_regions

    def find_lipschitz_constant(self, minDx, timeout):
        start_time = time.time()
        max_c = 0
        dx = None
        dy = None

        for b1, y1 in (pair for boxes, ys in self.generator_boxes() for pair in zip(boxes, ys)):
            if time.time() - start_time >= timeout:
                break
            near_boxes = self.generator_boxes(lambda b2, ymin, ymax: max(abs(y1 - ymin), abs(ymax - y1))/max(minDx, util.min_dist(b1, b2)) > max_c)
            for b2, y2 in (pair for boxes, ys in near_boxes for pair in zip(boxes, ys)):
                Dy = abs(y2 - y1)
                Dx = max(minDx, util.min_dist(b1, b2))
                c = Dy / Dx
                if c > max_c:
                    max_c = c
                    dx = util.min_dist(b1, b2)
                    dy = Dy

        print(type(dx), type(float(dy)), type(max_c))
        return max_c, float(dx), float(dy)
        
    


if __name__ == "__main__":
    pass
          