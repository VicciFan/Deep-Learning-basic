import math
class initializer(object):
    def __init__(self):
        pass
    def initialize(self,tensor):
        pass
class xavier_initializer(initializer):
    def __init__(self,type="uniform"):
        super(xavier_initializer,self).__init__()
        self.type=type
    def initialize(self,tensor):
        s=sum(list(tensor.size()))
        if self.type=="gaussian":
            x = math.sqrt(2. / s)
            return tensor.normal_(std=x)
        if self.type=="uniform":
            x = math.sqrt(6. / s)        
            return tensor.uniform_(-x, x)
class gaussian_initializer(initializer):
    def __init__(self,mean=0.0,std=1.0):
        super(gaussian_initializer,self).__init__()
        self.mean=mean
        self.std=std
    def initialize(self,tensor):
        return tensor.normal_(mean=self.mean,std=self.std)
class zeros_initializer(initializer):
    def __init__(self):
        super(zeros_initializer,self).__init__()
    def initialize(self,tensor):
        return tensor.zero_() 
class uniform_initializer(initializer):
    def __init__(self,a=-1.0,b=1.0):
        super(uniform_initializer,self).__init__()
        self.a=a
        self.b=b
    def initialize(self,tensor):
        return tensor.uniform_(self.a,self.b)
        
