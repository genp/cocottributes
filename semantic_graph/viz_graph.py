import asciitree
import config

def make_graph(dtree, depth):
    '''
    Expects a dictionary storing the tree created by an SGraph instance
    '''
    root = AttributeNode([], dtree['[]'], depth, dtree)
    print asciitree.draw_tree(root)    

    return

def get_child(elem):
    
    return

class AttributeNode(object):
    def __init__(self, knowns, last_known, depth, dtree):
        self.knowns = knowns
        self.last_known = last_known
        self.depth = depth
        self.dtree = dtree

    def __str__(self):
        return config.attr[self.last_known]

    @property
    def children(self):
        if self.depth <= 0:
            return []
        childs = []
        left = sorted(self.knowns + [(self.last_known,1.0)])
        try:
            lnxt = self.dtree[str(left)]
            childs.append(AttributeNode(left, lnxt, self.depth-1, self.dtree))
        except KeyError, e:
            print ' no left node '
        right = sorted(self.knowns + [(self.last_known,0.0)])
        try:
            rnxt = self.dtree[str(right)]
            childs.append(AttributeNode(right, rnxt, self.depth-1, self.dtree))
        except KeyError, e:
            print ' no right node ' 

        
        return childs
        




