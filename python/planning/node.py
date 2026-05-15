from dataclasses import dataclass, field
import typing

@dataclass(order=True)
class Node:
    # We sort the heap by f_cost, so it MUST be the first attribute!
    f_cost: float
    
    # field(compare=False) tells the heap not to break ties using these values
    g_cost: float = field(compare=False)
    h_cost: float = field(compare=False)
    position: tuple = field(compare=False) 
    
    # The parent node (used to retrace the path at the end)
    parent: typing.Any = field(default=None, compare=False)