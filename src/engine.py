#The scalar autograd engine. A 'value' wraps a single number and records
#the operations performed on it, building a computation graph that we 
#can later walk backwards to compute gradients

class Value:
    '''Stores a single scalar number and its gradient'''
    
    def __init__(self,data,_children=(),_op=''):
        #the actual number this node holds (the 'forward' value)
        self.data = data

        #the gradient of the final output with respect to THIS value.
        #it starts at 0.0 -- meaning 'no effect known yet.'
        self.grad = 0.0

        #the graph bookkeeping (memory of how this value was made)
        #The set of value objects that were the inputs of this operation
        #that produced this one. We use a set so a value used twice isn't stored
        #twice, leading underscore = 'internal, not for outside use'
        self._prev = set(_children)

        #A short string label for the operation that created this value
        # (e.g. + '+' or '*'). Empty for values you create directly
        self._op = _op

        #each value stores a function that knows how to push gradient
        #back to its inputs. By default a value has no inputs 
        #so its backwards step does nothing - an empty function.
        self._backward = lambda: None

    def __add__(self,other):
        #Called when you write self + other
        # build a NEW value whose data is the sum, and record that it came 
        #from (self,other) via the '+' operation
        out = Value(self.data + other.data, (self,other), '+')

        #Define how gradient flows backwards through THIS addition.
        #local gradient of (a+b) w.r.t each input is 1\        def _backward():
            self.grad += out.grad # 1*out.grad
            other.grad +=out.grad #1*out.grad
        out._backward = _backward #attach it to the result node
        return out
    
    def __mul__(self,other):
        #Called when you write self * other
        out = Value(self.data*other.data,(self,other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data *out.grad
        out._backward = _backward
            
        return out

    def __repr__(self):
        #Controls how a Value prints, so it's readable at the terminal
        #instead of showing something incorrect
        #we adjust the shown 
        return f"Value(data = {self.data},grad = {self.grad})"