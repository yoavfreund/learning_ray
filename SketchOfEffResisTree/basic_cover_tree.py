import numpy as np
import random
from time import perf_counter
import matplotlib.pyplot as plt
import os
from utilities.metrics import _dist
from utilities.data_generators import generate_2Dplane_10Dembedding


@ray.remote()
class CoverTreeNode():
    def __init__(self, center, radius, path, Queue_in ):
        """
        Queue_in is a ray.queue from where the node takes it's input.
        """
        self.center = center
        self.radius = radius
        self.Queue = Queue
        # The children, parent, root and path parameters
        # needs to be updated once node is inserted into a cover-tree
        self.children = []
        self.parent = parent
        self.root = None
        self.path = path  # Identifier that indicates the position of the node in the tree
        return

    def read_block_from_queue()
    """ Read a pointer from the queue and fetch it from the object store"""
    
    def __str__(self):
        return 'Node path=%s, radius=%5.2f, %d children' % \
               (self.path, self.radius, len(self.children)) + 'center=' + str(self.center)

    def get_children(self):
        return self.children

    def num_children(self):
        return len(self.children)

    def get_level(self):
        return len(self.path)


class BasicCoverTree():
    """A cover tree class that link cover tree nodes together to form a forest of
    cover trees. The BasicCoverTree is linked to the first existing node that is close enough,
    which is not necessarily the nearest existing node

    find_path check the children nodes randomly, and selects the first node that is close
    enough as a potential new parent
    """

    debug = False
    def set_debug(self):
        BasicCoverTree.debug = True

    max_depth = 10  # -1
    def set_max_depth(self, d):
        BasicCoverTree.max_depth = d

    def __init__(self, init_radius, r_reduce_factor):
        self.init_radius = init_radius
        self.r_reduce_factor = r_reduce_factor
        self.nodes = {}
        self.centers = {}
        self.roots = []
        return

    def insert(self, x):
        path = self.find_path(x, self.roots)
        if path is None:
            self.add_new_root(x)
        else:
            leaf = path[-1]
            self.add_new_node(x, leaf)
        return True

    def find_path(self, x, nodes):
        """Find path to location in tree, where node should be inserted"""
        if len(nodes) == 0:
            return None

        # We add a randomization step, to prevent first node in nodes to always be checked first
        indices = self.randomizer(np.arange(len(nodes)))
        for idx in indices:
            node = nodes[idx]
            if _dist(x, node.center) < node.radius:
                node_path = self.find_path(x, node.get_children())
                if node_path is None:
                    return [node]
                else:
                    return [node] + node_path
            else:
                continue
        return None

    def randomizer(self, indices):
        """ returns list with randomly shuffled indices."""
        random.Random(43).shuffle(indices)
        return indices

    def add_new_node(self, x, leaf):
        """
        Add a new node as a child to leaf
        :param x: center of new node
        :param leaf: parent of new node
        """
        new = CoverTreeNode(x, leaf.radius / 2, leaf.path + (leaf.num_children(),))
        new.parent = leaf
        new.root = leaf.root
        leaf.children.append(new)

        depth = len(new.path)
        if depth in self.nodes:
            self.centers[depth].append(new.center)
            self.nodes[depth].append(new)
        else:
            self.centers[depth] = [new.center]
            self.nodes[depth] = [new]

        if self.debug:
            print(f'Added new node {new.__str__()}')

        return new

    def add_new_root(self, x):
        """
        Add a new node as root
        :param x: center of new root
        """
        new = CoverTreeNode(x, self.init_radius / 2, (len(self.roots),))
        new.parent = None
        new.root = new

        self.roots.append(new)
        depth = len(new.path)
        if depth in self.nodes:
            self.centers[depth].append(new.center)
            self.nodes[depth].append(new)
        else:
            self.centers[depth] = [new.center]
            self.nodes[depth] = [new]

        if self.debug:
            print(f'Added new root {new.__str__()}')
        return

    def get_nodes(self):
        return self.nodes

    def get_centers(self):
        return self.centers


def run_speed_test(data, r_reduce_factor, init_radius, size):
    start_total = perf_counter()
    tree = BasicCoverTree(init_radius, r_reduce_factor)
    # tree.set_debug()
    avg_insert_time = []
    for x in data:
        start_average = perf_counter()
        tree.insert(x)
        stop_average = perf_counter()
        avg_insert_time.append(stop_average-start_average)
    avg_insert_time = np.mean(avg_insert_time)
    stop_total = perf_counter()
    print(f'Total time to build cover tree {stop_total - start_total} s, with {size} points')
    print(f'Average time to insert a point {avg_insert_time} s\n')
    return tree

def run_cover_properties_test(allcenters, r_reduce_factor, init_radius, indices):
    max_lvl = min(5, len(allcenters))
    for lvl in range(1, max_lvl+1):
        centers = allcenters[lvl]
        print(f'Number of centers lvl {lvl} = {len(centers)}\n')
        radius = init_radius / (r_reduce_factor ** lvl)
        fig, ax = plt.subplots()
        plt.title(f'Nodes at level {lvl}')
        for center in centers:
            ax.add_patch(plt.Circle(tuple([center[indices[0]], center[indices[1]]]),
                                    radius=radius, color='r', fill=False))
            plt.scatter(center[indices[0]], center[indices[1]], s=20, c='k', marker='o', zorder=2)
        plt.legend()
    plt.show()
    return

def check(allcenters):
    sum = 0
    for lvl in range(1, len(allcenters)+1):
        centers = allcenters[lvl]
        sum = sum + len(centers)
    print("Check sum of nodes: ", sum)


if __name__ == '__main__':
    N = 10000  # Number of samples
    d = 2  # Dim of plane embedded in D dim space
    D = 10  # embedding dim
    indices = [0, 1]
    eps = 0.001  # Noise

    r_reduce_factor = 2
    init_radius = 1

    directory = os.path.join('..', 'Data')
    filename = f'{d}Dplane_{D}Dembed_N{N}_eps{eps}'
    data = generate_2Dplane_10Dembedding(N, d, D, eps=eps)

    tree = run_speed_test(data, r_reduce_factor, init_radius, N)
    run_cover_properties_test(tree.get_centers(), r_reduce_factor, init_radius, indices)
    check(tree.get_centers())
