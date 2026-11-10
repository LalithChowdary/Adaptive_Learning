"""Deterministic real-answer adapter. Submitted code is NEVER executed by Python.

The completion evaluator interprets an explicit, bounded AST subset. Unsupported
syntax is unresolved, not a failed answer. No host attributes, imports, files,
network, eval, exec, arbitrary calls, or recursion are exposed.
"""
import ast
import math
import operator
from catalogue import source

class Unsupported(ValueError): pass
class Limit(Unsupported): pass
class Returned(Exception):
    def __init__(self, value): self.value = value

BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
COMPARE = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
           ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
ALLOWED = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign,
 ast.AugAssign, ast.For, ast.If, ast.Pass, ast.Expr, ast.Name, ast.Load, ast.Store,
 ast.Constant, ast.List, ast.Tuple, ast.Subscript, ast.Slice, ast.BinOp, ast.UnaryOp,
 ast.USub, ast.UAdd, ast.Not, ast.Compare, ast.BoolOp, ast.And, ast.Or, ast.IfExp,
 ast.Call, *BINARY.keys(), *COMPARE.keys())

class Interpreter:
    def __init__(self, text):
        if not isinstance(text,str) or len(text) > 8000: raise Unsupported('Code must be at most 8,000 characters.')
        tree = ast.parse(text)
        if len(tree.body)!=1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise Unsupported('Submit one function: def solve(x): ...')
        f = tree.body[0]
        if f.name!='solve' or len(f.args.args)!=1 or f.args.args[0].arg!='x' or f.decorator_list or f.args.defaults or f.args.kwonlyargs or f.args.vararg or f.args.kwarg or f.args.posonlyargs or f.returns or f.args.args[0].annotation:
            raise Unsupported('Use the signature def solve(x): without annotations or decorators.')
        nodes=list(ast.walk(tree))
        if len(nodes)>1200: raise Limit('Code is too large for this exercise runner.')
        for n in nodes:
            if not isinstance(n, ALLOWED): raise Unsupported(f'{type(n).__name__} is not supported by this exercise runner.')
            if isinstance(n, ast.FunctionDef) and n is not f: raise Unsupported('Nested functions are not supported.')
            if isinstance(n, ast.Name) and n.id.startswith('__'): raise Unsupported('Special names are not supported.')
            if isinstance(n, ast.Call) and (not isinstance(n.func,ast.Name) or n.func.id not in ('range','list','len','sum','min','max','abs') or n.keywords):
                raise Unsupported('Available calls: range, list, len, sum, min, max, abs (positional arguments).')
            if isinstance(n, ast.Constant) and type(n.value) not in (int,float,bool,type(None),str): raise Unsupported('Unsupported literal.')
        self.body=f.body

    def bounded(self,v):
        if isinstance(v,(list,tuple,range,str)) and len(v)>2000: raise Limit('Sequence limit exceeded (2,000 items).')
        if type(v) is int and v.bit_length()>256: raise Limit('Integer limit exceeded.')
        if type(v) is float and not math.isfinite(v): raise Limit('Non-finite calculation.')
        return v

    def run(self,x):
        self.env={'x':x}; self.steps=0
        try: self.block(self.body)
        except Returned as r: return r.value
        return None

    def tick(self):
        self.steps+=1
        if self.steps>10000: raise Limit('Step limit exceeded (10,000 operations).')

    def binary(self, op, a,b):
        if isinstance(op,ast.Mult):
            if isinstance(a,(list,tuple,str)) and type(b) is int and len(a)*max(0,b)>2000: raise Limit('Sequence limit exceeded.')
            if isinstance(b,(list,tuple,str)) and type(a) is int and len(b)*max(0,a)>2000: raise Limit('Sequence limit exceeded.')
        return self.bounded(BINARY[type(op)](a,b))

    def expression(self,n):
        self.tick()
        if isinstance(n,ast.Constant): return self.bounded(n.value)
        if isinstance(n,ast.Name): return self.env[n.id]
        if isinstance(n,(ast.List,ast.Tuple)): return self.bounded([self.expression(x) for x in n.elts]) if isinstance(n,ast.List) else tuple(self.expression(x) for x in n.elts)
        if isinstance(n,ast.BinOp): return self.binary(n.op,self.expression(n.left),self.expression(n.right))
        if isinstance(n,ast.UnaryOp): return {ast.USub:operator.neg,ast.UAdd:operator.pos,ast.Not:operator.not_}[type(n.op)](self.expression(n.operand))
        if isinstance(n,ast.Compare):
            a=self.expression(n.left)
            for op,b in zip(n.ops,n.comparators):
                b=self.expression(b)
                if not COMPARE[type(op)](a,b): return False
                a=b
            return True
        if isinstance(n,ast.BoolOp):
            for v in n.values:
                result=self.expression(v)
                if isinstance(n.op,ast.And) and not result or isinstance(n.op,ast.Or) and result: return result
            return result
        if isinstance(n,ast.IfExp): return self.expression(n.body if self.expression(n.test) else n.orelse)
        if isinstance(n,ast.Slice): return slice(*(self.expression(v) if v else None for v in (n.lower,n.upper,n.step)))
        if isinstance(n,ast.Subscript): return self.bounded(self.expression(n.value)[self.expression(n.slice)])
        if isinstance(n,ast.Call):
            args=[self.expression(a) for a in n.args]
            return self.bounded({'range':range,'list':list,'len':len,'sum':sum,'min':min,'max':max,'abs':abs}[n.func.id](*args))
        raise Unsupported(f'Unsupported expression: {type(n).__name__}')

    def assign(self,n,v):
        if not isinstance(n,ast.Name): raise Unsupported('Assignments must target a variable name.')
        self.env[n.id]=v

    def block(self, body):
        for n in body:
            self.tick()
            if isinstance(n,ast.Return): raise Returned(self.expression(n.value) if n.value else None)
            elif isinstance(n,ast.Assign):
                value=self.expression(n.value)
                for target in n.targets: self.assign(target,value)
            elif isinstance(n,ast.AugAssign): self.assign(n.target,self.binary(n.op,self.expression(n.target),self.expression(n.value)))
            elif isinstance(n,ast.If): self.block(n.body if self.expression(n.test) else n.orelse)
            elif isinstance(n,ast.For):
                for value in self.expression(n.iter): self.assign(n.target,value); self.block(n.body)
                self.block(n.orelse)
            elif isinstance(n,ast.Pass): pass
            elif isinstance(n,ast.Expr): self.expression(n.value)
            else: raise Unsupported(f'Unsupported statement: {type(n).__name__}')

def normalize_output(text):
    if not isinstance(text,str) or len(text)>8000: raise ValueError('Enter a short text answer.')
    text=text.strip()
    try:
        value=ast.literal_eval(text)
        if type(value) in (int,list,tuple): return (type(value).__name__,value)
    except (ValueError,SyntaxError,RecursionError): pass
    return ('text',text)

def grade(task, answer):
    fmt=task['task_type']; details=[]
    if fmt in ('single_choice','multiple_select'):
        result=source.grade(task,answer)
    elif fmt in ('output_prediction','fill_in_the_blank'):
        if not isinstance(answer,list) or len(answer)!=len(task['answer_key']): raise ValueError('Answer every field.')
        if any(not isinstance(v,str) or not v.strip() for v in answer): raise ValueError('Answer every field.')
        if fmt=='fill_in_the_blank':
            flags=[]
            for value,key in zip(answer,task['answer_key']):
                try:
                    literal=ast.literal_eval(value.strip())
                    flags.append(type(literal) is int and literal==int(key))
                except (ValueError,SyntaxError,RecursionError): flags.append(False)
        else: flags=[normalize_output(v)==normalize_output(k) for v,k in zip(answer,task['answer_key'])]
        result=source.grade(task,flags)
    else:
        flags=[]
        try:
            runner=Interpreter(answer)
            for i,test in enumerate(task['tests']):
                try: passed=runner.run(test['input'])==test['expected']; message='Passed' if passed else 'Returned a different result'
                except (KeyError,TypeError,ValueError,IndexError,ZeroDivisionError) as e:
                    if isinstance(e,Unsupported): raise
                    passed=False; message=f'{type(e).__name__} during this case'
                flags.append(passed); details.append(dict(criterion=i+1,passed=passed,message=message))
        except (Unsupported,RecursionError,OverflowError) as e:
            return dict(grading_status='unresolved',score=None,full_success=None,criteria=[],details=[],message=str(e),grader='deterministic-dashboard-v1')
        except SyntaxError as e:
            flags=[False]*len(task['tests']); details=[dict(message=f'Syntax error at line {e.lineno}: {e.msg}')]
        result=source.grade(task,flags)
    return dict(**result, grading_status='resolved',details=details,grader='deterministic-dashboard-v1')
