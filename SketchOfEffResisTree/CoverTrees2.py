"""
Generate and manage cover trees.

Minimal description
"""

import random
import numpy as np
#import pandas as pd
import pylab as py
from Kmeans import Kmeans
from Generators import SwissGenerator, DataGenerator, findKNN

import sys
_MAX = sys.float_info.max

# access global parameters

def _dist(x,y):
    """
    Compute the distance between two points. Euclidean distance between 1D numpy arrays.

    Parameters
    ----------
    x: numpy array
        First Point
    y: numpy array
        Second Point 

    Returns
    -------
    float
    Euclidean distance

    """
    return np.sqrt(np.sum((x-y)**2))
    
class CoverTreeNode(object):
    """ A generic tree node for a space partitioning tree
    :param object: _description_
    :type object: _type_
    :return: _description_
    :rtype: _type_
    """    
    debug=False
    def set_debug(self):
        CoverTreeNode.debug=True
        
    max_depth=-1
    def set_max_depth(self,d):
        CoverTreeNode.max_depth=d
        
    def __init__(self,center,radius, path,r_reduce_factor=2,parent=None):
        """
        Defines a node in a covertree

        Parameters
        ----------        
        center: point  
            The center of the region for this node
        radius: positive float
            Only points whose distance from center is at most radius are included in the node
        path:   a list of CoverTrees
            A sequence of nodes that defines the tree path leading to this node.
 
        Returns
        -------
        None.

        """
        self.parent=parent
        self.center=center
        self.radius=radius
        self.r_reduce_factor =  r_reduce_factor
        self.counter=1  # number of points covered by this tree
        self.too_far=0  # count how many points were too far (should only be non-zero at root
        self.path=path
        self.children = []
    
    def dist_to_x(self,x):
        return _dist(x,self.center)

    def covers(self,x):
        return self.dist_to_x(x) < self.radius
        
    def no_of_children(self):
        return len(self.children)

    def get_level(self):
        return len(self.path)
    
    def find_path(self,x):
        """
        Find the path in the current tree for the point x.

        Parameters
        ----------
        x : point
            point for which we want to know the tree path.
        failPossible: a flag controlling whether or not the insertion fails.

        Returns
        -------
        list of CoverTreeNode s
            The list of nodes that contain the point x

        """
        d= _dist(x,self.center)
        #print(str(self.path),d)
        if d>self.radius:
             return None
        if len(self.children)==0:
            return [self]
        else:
            for child in self.children:
                child_path = child.find_path(x)
                if child_path is None:
                    return [self]
                else:
                    return [self]+child_path
            return [self]

    def insert(self,x,,failPossible=True):
        """
        Add a new node that contains a given point.  
        Fails if node is outside of the ball of the root.

        Parameters
        ----------
        x : point
            The point to seed the new node
        FailPossible: Boolean (default=True)
            Insertion fail if point is outside radius and FailPossible=True

        Returns
        -------
        bool
            Success/Failure 

        """
        path=self.find_path(x)
        if path is None and failPossible:
            return False

        #found a non-trivial path
        leaf=path[-1]
        new=CoverTreeNode(x,leaf.radius/leaf.r_reduce_factor,leaf.path+(leaf.no_of_children(),))
        leaf.children.append(new)
        for node in path:
            node.counter +=1
        return True
                    
    def collect_centers(self,depth=-1):
        """
        Collect all of the centers defined by a tree up to a given depth

        Parameters
        ----------
        depth=the desired depth in the tree. Default=-1, all depths.

        Returns
        -------
        C : list
            returns a list where each element is a center, followed by the level of the center
        """
        d = len(self.path)
        C=[(self.center,d,self.radius,self.state.get_state())]
        if (depth==-1 or d<depth) and len(self.children)>0:
            for child in self.children:
                C = C+ child.collect_centers()
        return C
           
    def collect_nodes(self,depth=-1):
        """
        Create list of nodes.
        
        Returns
        -------
        N : list
            returns a list of all nodes in the tree
        """
        path=self.path
        d = len(self.path)
        if depth!=-1 and  d>=depth:
            return []
        N=[self]
        if len(self.children)>0:
            for child in self.children:
                N=N+child.collect_nodes(depth=depth)
        return N

    def __str__(self):
        """
        Create a string describing this node.

        Returns
        -------
        string
            description of the node

        """
        return str(self.path)+': r=%4.2f, no_child=%d, count=%d'%(self.radius,len(self.children),self.counter)

    def _str_to_level(self,max_level=0,print_leaves=False):
        """
        Create a string that descibes the nodes along the path to this node.

        Parameters
        ----------
        max_level : integer
            The maximal level  The default is 0 which correspond to unlimited level.
        print_initial : Boolean
            print all nodes, including leaves, The default is False (don't print leaves')

        Returns
        -------
        s : string
            string.

        """
        s=self.__str__()+'\n'
        if self.get_level() < max_level and len(self.children)>0:
            for i in range(len(self.children)):
                child=self.children[i]
                if child.state.get_state() != 'initial':
                    s+=child._str_to_level(max_level)
        return s    

class NodeState:
    """
       states = ('initial', # initial state, collect statistics
              'seed',    # collect centers
              'refine',  # refine centers
              'passThrough') # only collect statistics, advance children from 'initial' to 'seed'
    """
    states = ('initial', # initial state, collect statistics
              'seed',    # collect centers
              'refine',  # refine centers
              'passThrough') # only collect statistics, advance children from 'initial' to 'seed'
    def __init__(self):
        self.state = 'initial'
    def get_state(self):
        return self.state
    def set_state(self,state):
        assert(str(state) in NodeState.states)
        self.state=state
        
class ElaboratedTreeNode(CoverTreeNode):
    def __init__(self,center,radius,path,thr=0.9,alpha=0.1,r_reduce_factor=2,max_children=10,parent=None):
        """
        Initialize and elaboratedTreeNode

        Parameters
        ----------
        center, radius, path : 
            as defined in CoverTreeNode.__init__()
        thr : TYPE, optional
            The minimal estimated coverage of the node to allow it's children to grow. The default is 0.9.
        alpha : TYPE, optional
                The mixing coefficient for the estimator: estim=(1-alpha)*estim + alpha*new value
        r_reduce_factor : factor by which to reduce the radius for the children (default 2)
        Returns
        -------
        None.

        """
        super().__init__(center,radius,path,parent=parent)
        self.covered_fraction = 0
        self.state = NodeState()
        self.thr=thr
        self.alpha=alpha
        self.r_reduce_factor =  r_reduce_factor 
        self.points=[]  # collects point the are covered by this node
        self.max_children=max_children
        self.cost=-1
        self.debug=False

    def find_closest_child(self,x):
        """
        Find the child of this node whose center is closest to x

        Parameters
        ----------
        x : point

        Returns
        -------
        closest_child: ElaboratedTreeNode
        the closest child
        
        _min_d: float
        distance from the closest child.
        """
        if self.no_of_children() ==0:
            return None,None
        _min_d = _MAX
        closest_child=None
        for child in self.children:
            _d = child.dist_to_x(x)
            if _d < _min_d:
                closest_child=child
                _min_d=_d
        assert(not closest_child is None), "find_closest_child failed, x=%f"%x+'node=\n'+str(self)
        return closest_child,_min_d

    def find_path(self,x):
        """ Find path in tree that corresponds to the point x

        :param x: the input point

        :returns: path from root to leaf
        :rtype: a list of nodes
        """

        if len(self.children)==0:
            return [self]
        else:
            closest_child,distance = self.find_closest_child(x)
            if closest_child is None:
                return [self]
            else:
                child_path = closest_child.find_path(x)
                return [self]+child_path
    
    def conditionally_add_child(self,x):
        """ Decide whether to add new child 
        :param x: 
        :returns: covered or init or filter-add or filter-discard
        :rtype: 

        """
        if self.no_of_children()==0:
            self.add_child(self.center)  # add parent center as child
            return 'init'
        _child,d = self.find_closest_child(x)
        assert(d != None)

        r=_child.radius
        if d <= r: 
            return 'covered'
        else:                   # if d>r far from center use modified kmeans++ rule
            P=min(1.0,((d-r)/r)**2)
            #print('d=%4.2f, r=%4.2f'%(d,r),end=' ')
            #print('adding point with P=%f'%P,end=', ')
            if random.random()<P:
                #print(self.path,' Success') 
                self.add_child(x)
                return 'filter-add'
            else:
                #print(self.path,' Fail')
                return 'filter-discard'
            

    def add_child(self,x):
        """ Add child to node

        :param x: 

        """
        new=ElaboratedTreeNode(x,radius=self.radius/self.r_reduce_factor,path=self.path+(self.no_of_children(),),thr=self.thr,parent=self)
        self.children.append(new)
        
    def insert(self,x):
        """ insert an example into this node.
        :param x: the example
        :returns: Flag indicating whether example was rejected.
        :rtype: Flag
        """

        self.points.append(x)
        if self.debug:
            print('in insert',self,' debug=',self.debug, 'max_depth=',self.max_depth)
        state = self.state.get_state()
        if state=='initial': # initial state, do nothing
            pass
        if state=='seed':                    
            add_status = self.conditionally_add_child(x);
            #print('add_status=',add_status,'covered fraction',self.covered_fraction)
            if add_status in ['init','covered']: 
                self.covered_fraction = (1-self.alpha)*self.covered_fraction + self.alpha
            else:
                self.covered_fraction = (1-self.alpha)*self.covered_fraction
            if self.covered_fraction>self.thr:
                if  self.debug:
                    print('node'+str(self.path)+
                      'finished seeding frac=%7.5f, no of points = %d, no of children=%2d'\
                      %(self.covered_fraction,len(self.points),self.no_of_children()),end=' ')
                self.state.set_state('refine')
                self.refine()
                if self.debug:
                    print('cost = %7.5f'%self.cost)
                if self.max_depth==-1 or len(self.path) < self.max_depth:  # Do not expand tree beyond max_depth
                    self.state.set_state('passThrough')
        if state=='passThrough':
            _child,d = self.find_closest_child(x)
            _child.insert(x)

    def refine(self):
        Centers=[_child.center for _child in self.children]
        self.cost,newCenters=Kmeans(self.points,Centers,stationary=[0])
        push_through = self.no_of_children() <= self.max_children
        for i in range(len(self.children)):
            child=self.children[i]
            child.center = newCenters[i]
            if push_through: 
                child.state.set_state('seed')
        
    def __str__(self):
        return str(self.path)+': r=%4.2f, center='%self.radius+str(self.center)+\
        ' state= %s, no_child=%d, count=%d, cov_frac=%4.3f, cost=%4.3f'\
        %(self.state.get_state(),len(self.children),len(self.points),self.covered_fraction,self.cost)

    def compute_graphs(self,max_depth=-1,k=6,debug=False):
        """
        Computer k-nearest graphs or each level of the tree

        Parameters
        ----------
        max_depth : int, optional
            max-depth for which to generate a graph. The default is -1.
        k : TYPE, optional
            The number f neighbors. The default is 6.
        debug : TYPE, optional
            Set to true to get denugging messages. The default is False.

        Returns
        -------
        Layers : a list of dicts:
            {depth: depth of nodes used for graph,
            nodes: ndarray of center coordinates
            radius: the radius used at this depth of the tree,
            edges: list of edges, each defined by the indices of the end-point nodes
            }
        """
        T=self
        nodes=T.collect_nodes(depth=max_depth)
        if self.debug:
            print('in compute_graph, number of nodes=',len(nodes),'max_depth=',max_depth)
        X=np.stack([x.center for x in nodes])
        depth=np.array([len(x.path) for x in nodes])
        radius = np.array([x.radius for x in nodes])
        parent_state=[]
        for node in nodes:
            if len(node.path) == 0:  # if root node
                parent_state.append('passThrough')
            else:
                parent_state.append(node.parent.state.get_state())
        
        if debug:
            print('compute_graph, no. of nodes=',len(nodes),'depth min/max=',np.min(depth),np.max(depth))
            
    
        md=np.max(depth)
        if max_depth!=-1 and max_depth<md:
            md=max_depth
        Layers=[]
        for d in range(md):
            E=X[depth==d,:]
            if debug:
                print('plot_graphs, d=',d,'E.shape=',E.shape)
            first=np.nonzero(depth==d)[0][0]
            Layer={'depth':d,
                   'nodes':E,
                   'radius':radius[first]
                }
            if d>0:
                S=findKNN(E,k=k)
                Layer['edges']=S['pairs']
            Layers.append(Layer)
            
        return Layers
    def extract_epsilon_cover(self,):
        X=self.collect_nodes()
        centers={}
        coverage=np.zeros([20,2]) # accumulate coverage statistics
        for x in X:
            depth=len(x.path)
            
            node={'center':x.center, 'count':len(x.points)}
            coverage[depth,:]=coverage[depth,:]+np.array([1,x.covered_fraction])
            if depth in centers:
                centers[depth].append(node)
            else:
                centers[depth]=[node]
        coverage=coverage[:,1]/(coverage[:,0]+1)
        return centers,coverage

def gen_scatter(T,data,level=0):
    C=[]
    for i in range(data.shape[0]):
        point=np.array(data.iloc[i,:])
        #print(T.find_path(point))
        C.append(T.find_path(point)[-1].path[level])

    py.figure(figsize=[15,5])
    py.scatter(data[0],data[1],s=1,c=C,alpha=0.2)
    t='Level=%d, colors=%d'%(level,max(C)+1)
    py.title(t);
