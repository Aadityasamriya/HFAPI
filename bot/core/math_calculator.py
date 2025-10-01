"""
Mathematical Reasoning and Calculator Integration
Provides real mathematical capabilities instead of relying solely on LLMs
"""

import re
import logging
import math
import operator
import ast
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class MathOperationType(Enum):
    """Types of mathematical operations"""
    BASIC_ARITHMETIC = "basic_arithmetic"
    ALGEBRA = "algebra"
    CALCULUS = "calculus" 
    STATISTICS = "statistics"
    TRIGONOMETRY = "trigonometry"
    GEOMETRY = "geometry"
    LINEAR_ALGEBRA = "linear_algebra"
    COMPLEX_EXPRESSION = "complex_expression"

@dataclass
class MathResult:
    """Result of mathematical calculation"""
    success: bool
    result: Optional[float]
    explanation: str
    operation_type: MathOperationType
    original_expression: str
    steps: List[str]
    error_message: Optional[str] = None

class SafeMathCalculator:
    """
    Safe mathematical calculator with expression evaluation
    Provides actual mathematical capabilities beyond LLM limitations
    """
    
    def __init__(self):
        self.safe_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }
        
        self.safe_functions = {
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'sinh': math.sinh,
            'cosh': math.cosh,
            'tanh': math.tanh,
            'log': math.log,
            'log10': math.log10,
            'log2': math.log2,
            'exp': math.exp,
            'sqrt': math.sqrt,
            'abs': abs,
            'ceil': math.ceil,
            'floor': math.floor,
            'round': round,
            'factorial': math.factorial,
            'gcd': math.gcd,
            'pi': math.pi,
            'e': math.e,
            'max': max,
            'min': min,
            'sum': sum,
        }
        
        # Common mathematical constants
        self.constants = {
            'pi': math.pi,
            'e': math.e,
            'tau': math.tau,
            'inf': math.inf,
        }
        
    def extract_mathematical_expressions(self, text: str) -> List[str]:
        """Extract mathematical expressions from text"""
        patterns = [
            r'\b\d+\.?\d*\s*[+\-*/^%]\s*\d+\.?\d*\b',  # Basic operations
            r'\b\d+\s*[+\-*/^%]\s*\d+\s*[+\-*/^%]\s*\d+',  # Chained operations
            r'[a-z_]+\s*\(\s*[\d\.\s+\-*/^%,]+\s*\)',  # Function calls
            r'\b\d*\.?\d+\s*\*\s*\d*\.?\d+\s*\+\s*\d*\.?\d+',  # ax + b
            r'sqrt\s*\(\s*[\d\.\s+\-*/^%]+\s*\)',  # Square root
            r'log\s*\(\s*[\d\.\s+\-*/^%]+\s*\)',  # Logarithm
            r'\d+\s*\^\s*\d+',  # Exponentiation
            r'\d+\s*!\s*',  # Factorial
        ]
        
        expressions = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            expressions.extend(matches)
        
        return expressions
    
    def classify_math_operation(self, expression: str) -> MathOperationType:
        """Classify the type of mathematical operation"""
        expr_lower = expression.lower()
        
        if any(func in expr_lower for func in ['sin', 'cos', 'tan', 'asin', 'acos', 'atan']):
            return MathOperationType.TRIGONOMETRY
        elif any(func in expr_lower for func in ['log', 'exp', 'ln']):
            return MathOperationType.ALGEBRA
        elif any(func in expr_lower for func in ['sqrt', 'pow', '^']):
            return MathOperationType.ALGEBRA
        elif any(op in expr_lower for op in ['mean', 'std', 'var', 'median']):
            return MathOperationType.STATISTICS
        elif any(term in expr_lower for term in ['derivative', 'integral', 'limit']):
            return MathOperationType.CALCULUS
        elif any(char in expression for char in ['+', '-', '*', '/', '%']):
            return MathOperationType.BASIC_ARITHMETIC
        else:
            return MathOperationType.COMPLEX_EXPRESSION
    
    def safe_eval(self, node: ast.AST) -> Union[float, int]:
        """Safely evaluate AST node"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in self.constants:
                return self.constants[node.id]
            else:
                raise ValueError(f"Undefined variable: {node.id}")
        elif isinstance(node, ast.BinOp):
            left = self.safe_eval(node.left)
            right = self.safe_eval(node.right)
            op = self.safe_operators.get(type(node.op))
            if op:
                return op(left, right)
            else:
                raise ValueError(f"Unsupported operation: {type(node.op)}")
        elif isinstance(node, ast.UnaryOp):
            operand = self.safe_eval(node.operand)
            op = self.safe_operators.get(type(node.op))
            if op:
                return op(operand)
            else:
                raise ValueError(f"Unsupported unary operation: {type(node.op)}")
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else str(node.func)
            if func_name in self.safe_functions:
                args = [self.safe_eval(arg) for arg in node.args]
                return self.safe_functions[func_name](*args)
            else:
                raise ValueError(f"Unsupported function: {func_name}")
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")
    
    def calculate_expression(self, expression: str) -> MathResult:
        """Calculate a mathematical expression safely"""
        original_expression = expression
        steps = []
        
        try:
            # Clean and prepare expression
            expression = expression.replace('^', '**')  # Python power operator
            expression = expression.replace('π', 'pi')   # Replace pi symbol
            expression = expression.replace('×', '*')    # Replace multiplication symbol
            expression = expression.replace('÷', '/')    # Replace division symbol
            
            operation_type = self.classify_math_operation(expression)
            steps.append(f"Parsing expression: {expression}")
            
            # Parse and evaluate
            tree = ast.parse(expression, mode='eval')
            result = self.safe_eval(tree.body)
            
            steps.append(f"Calculation complete: {result}")
            
            explanation = f"Evaluated '{original_expression}' as {operation_type.value} operation"
            
            return MathResult(
                success=True,
                result=result,
                explanation=explanation,
                operation_type=operation_type,
                original_expression=original_expression,
                steps=steps
            )
            
        except Exception as e:
            error_msg = f"Mathematical calculation error: {str(e)}"
            logger.error(error_msg)
            
            return MathResult(
                success=False,
                result=None,
                explanation=f"Could not evaluate '{original_expression}': {str(e)}",
                operation_type=MathOperationType.COMPLEX_EXPRESSION,
                original_expression=original_expression,
                steps=steps,
                error_message=error_msg
            )
    
    def solve_equation(self, equation: str) -> MathResult:
        """Solve simple linear equations"""
        # Handle equations like "2x + 5 = 17" or "solve for x: 3x - 7 = 14"
        original_equation = equation
        steps = []
        
        try:
            # Extract equation parts
            if '=' in equation:
                left, right = equation.split('=')
                left = left.strip()
                right = right.strip()
                
                steps.append(f"Equation: {left} = {right}")
                
                # Simple linear equation solver (ax + b = c)
                # Extract coefficients using regex
                linear_pattern = r'([+-]?\d*\.?\d*)\s*\*?\s*x\s*([+-]\s*\d+\.?\d*)?'
                match = re.search(linear_pattern, left)
                
                if match:
                    a_str = match.group(1) if match.group(1) else "1"
                    b_str = match.group(2) if match.group(2) else "0"
                    
                    # Clean coefficient strings
                    a_str = a_str.replace('*', '').strip()
                    if a_str == '' or a_str == '+':
                        a = 1
                    elif a_str == '-':
                        a = -1
                    else:
                        a = float(a_str)
                    
                    b_str = b_str.replace(' ', '') if b_str else "0"
                    b = float(b_str) if b_str != "0" else 0
                    
                    c = float(right)
                    
                    steps.append(f"Identified linear equation: {a}x + {b} = {c}")
                    
                    # Solve: ax + b = c => x = (c - b) / a
                    if a != 0:
                        x = (c - b) / a
                        steps.append(f"Solution: x = ({c} - {b}) / {a} = {x}")
                        
                        return MathResult(
                            success=True,
                            result=x,
                            explanation=f"Solved linear equation: x = {x}",
                            operation_type=MathOperationType.ALGEBRA,
                            original_expression=original_equation,
                            steps=steps
                        )
                    else:
                        raise ValueError("Cannot solve: coefficient of x is zero")
                else:
                    raise ValueError("Could not parse as linear equation")
            else:
                raise ValueError("No equals sign found in equation")
                
        except Exception as e:
            error_msg = f"Equation solving error: {str(e)}"
            logger.error(error_msg)
            
            return MathResult(
                success=False,
                result=None,
                explanation=f"Could not solve equation '{original_equation}': {str(e)}",
                operation_type=MathOperationType.ALGEBRA,
                original_expression=original_equation,
                steps=steps,
                error_message=error_msg
            )
    
    def analyze_mathematical_content(self, text: str) -> Dict[str, Any]:
        """Analyze text for mathematical content and provide solutions"""
        results = {
            'has_math': False,
            'expressions': [],
            'equations': [],
            'calculations': [],
            'operation_types': set()
        }
        
        # Extract mathematical expressions
        expressions = self.extract_mathematical_expressions(text)
        
        if expressions:
            results['has_math'] = True
            results['expressions'] = expressions
            
            for expr in expressions:
                calc_result = self.calculate_expression(expr)
                results['calculations'].append(calc_result)
                results['operation_types'].add(calc_result.operation_type)
        
        # Look for equations to solve
        equation_patterns = [
            r'[^=]*\bx\b[^=]*=\s*[\d\.\s+\-*/^%]+',
            r'solve\s+for\s+x[^=]*=',
            r'\d*\.?\d*\s*x\s*[+-]\s*\d+\.?\d*\s*=\s*\d+\.?\d*'
        ]
        
        for pattern in equation_patterns:
            equations = re.findall(pattern, text, re.IGNORECASE)
            if equations:
                results['has_math'] = True
                results['equations'].extend(equations)
                
                for eq in equations:
                    solve_result = self.solve_equation(eq)
                    results['calculations'].append(solve_result)
                    results['operation_types'].add(solve_result.operation_type)
        
        results['operation_types'] = list(results['operation_types'])
        return results

class MathReasoningEnhancer:
    """
    Enhanced mathematical reasoning that combines calculator with LLM
    Provides superior math capabilities compared to pure LLM approaches
    """
    
    def __init__(self):
        self.calculator = SafeMathCalculator()
        
    async def enhance_mathematical_response(self, prompt: str, llm_response: str) -> str:
        """Enhance LLM response with actual mathematical calculations"""
        
        # Analyze the prompt for mathematical content
        math_analysis = self.calculator.analyze_mathematical_content(prompt)
        
        if not math_analysis['has_math']:
            return llm_response
        
        enhanced_response = llm_response
        calculations_added = []
        
        # Add actual calculations
        for calc_result in math_analysis['calculations']:
            if calc_result.success:
                calc_explanation = f"\n\n**Mathematical Calculation:**\n"
                calc_explanation += f"Expression: `{calc_result.original_expression}`\n"
                calc_explanation += f"Result: **{calc_result.result}**\n"
                calc_explanation += f"Operation Type: {calc_result.operation_type.value}\n"
                
                if calc_result.steps:
                    calc_explanation += "Steps:\n"
                    for i, step in enumerate(calc_result.steps, 1):
                        calc_explanation += f"{i}. {step}\n"
                
                calculations_added.append(calc_explanation)
        
        if calculations_added:
            enhanced_response += "\n\n" + "".join(calculations_added)
            enhanced_response += "\n*Note: Mathematical calculations verified using integrated calculator*"
        
        return enhanced_response
    
    def requires_calculator(self, prompt: str) -> bool:
        """Check if prompt requires calculator integration"""
        math_analysis = self.calculator.analyze_mathematical_content(prompt)
        return math_analysis['has_math'] and len(math_analysis['calculations']) > 0