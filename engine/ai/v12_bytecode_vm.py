"""
V12 Bytecode VM - GPU-native visual synthesis engine
Transforms Milkdrop presets into bytecode executed by a GLSL VM
"""

import struct
from enum import IntEnum


class Opcode(IntEnum):
    """V12 Bytecode opcodes"""

    # Stack operations
    PUSH_CONST = 1  # Push constant float onto stack
    PUSH_VAR = 2  # Push variable value onto stack
    STORE = 3  # Store stack top to variable
    DUP = 4  # Duplicate top of stack
    SWAP = 5  # Swap top two stack values
    POP = 6  # Pop top of stack

    # Arithmetic
    ADD = 10  # Add: a + b
    SUB = 11  # Subtract: a - b
    MUL = 12  # Multiply: a * b
    DIV = 13  # Divide: a / b
    MOD = 14  # Modulo: a % b
    POW = 15  # Power: a ^ b
    NEG = 16  # Negate: -a

    # Comparison
    GT = 20  # Greater than: a > b
    LT = 21  # Less than: a < b
    GE = 22  # Greater or equal: a >= b
    LE = 23  # Less or equal: a <= b
    EQ = 24  # Equal: a == b
    NE = 25  # Not equal: a != b

    # Logical
    AND = 30  # Logical AND: a && b
    OR = 31  # Logical OR: a || b
    NOT = 32  # Logical NOT: !a

    # Ternary
    TERNARY = 40  # Ternary: cond ? a : b

    # Math functions
    SIN = 50  # Sine
    COS = 51  # Cosine
    TAN = 52  # Tangent
    ASIN = 53  # Arc sine
    ACOS = 54  # Arc cosine
    ATAN = 55  # Arc tangent
    SQRT = 56  # Square root
    ABS = 57  # Absolute value
    FLOOR = 58  # Floor
    CEIL = 59  # Ceiling
    ROUND = 60  # Round
    SIGN = 61  # Sign (-1, 0, 1)
    EXP = 62  # Exponential
    LOG = 63  # Natural log
    LOG10 = 64  # Base-10 log
    MIN = 65  # Minimum
    MAX = 66  # Maximum
    RAND = 67  # Random

    # Special
    NOOP = 255  # No operation
    HALT = 254  # Stop execution


class BytecodeCompiler:
    """Compiles Milkdrop AST to V12 bytecode"""

    def __init__(self):
        self.bytecode: list[int] = []
        self.constants: list[float] = []
        self.var_index: dict[str, int] = {}
        self.next_var_index = 0

        # Standard variable indices (Milkdrop standard)
        self._init_standard_vars()

    def _init_standard_vars(self):
        """Initialize standard Milkdrop variable indices"""
        standard_vars = [
            "x",
            "y",
            "time",
            "frame",
            "bass",
            "mid",
            "treb",
            "bass_att",
            "mid_att",
            "treb_att",
            "rad",
            "zoom",
            "rot",
            "dx",
            "dy",
            "thresh",
            "dx_r",
            "dy_r",
            "aspectx",
            "aspecty",
            "fps",
            "q1",
            "q2",
            "q3",
            "q4",
            "q5",
            "q6",
            "q7",
            "q8",
            "q9",
            "q10",
            "q11",
            "q12",
            "q13",
            "q14",
            "q15",
            "q16",
            "q17",
            "q18",
            "q19",
            "q20",
            "q21",
            "q22",
            "q23",
            "q24",
            "q25",
            "q26",
            "q27",
            "q28",
            "q29",
            "q30",
            "q31",
            "q32",
            "tg1",
            "tg2",
            "tg3",
            "wave_r",
            "wave_g",
            "wave_b",
        ]

        for var in standard_vars:
            self._get_var_index(var)

    def _get_var_index(self, var_name: str) -> int:
        """Get or create variable index"""
        if var_name not in self.var_index:
            self.var_index[var_name] = self.next_var_index
            self.next_var_index += 1
        return self.var_index[var_name]

    def _get_const_index(self, value: float) -> int:
        """Get or create constant index"""
        if value not in self.constants:
            self.constants.append(value)
        return self.constants.index(value)

    def emit(self, opcode: Opcode, *args: int):
        """Emit bytecode instruction"""
        self.bytecode.append(opcode.value)
        self.bytecode.extend(args)

    def compile_number(self, value: float):
        """Compile a number literal"""
        const_idx = self._get_const_index(value)
        self.emit(Opcode.PUSH_CONST, const_idx)

    def compile_variable(self, var_name: str):
        """Compile a variable reference"""
        var_idx = self._get_var_index(var_name)
        self.emit(Opcode.PUSH_VAR, var_idx)

    def compile_binary_op(self, op: str, left, right):
        """Compile a binary operation"""
        # Compile operands (right first for stack order)
        right.compile(self)
        left.compile(self)

        # Emit operation
        op_map = {
            "+": Opcode.ADD,
            "-": Opcode.SUB,
            "*": Opcode.MUL,
            "/": Opcode.DIV,
            "%": Opcode.MOD,
            "^": Opcode.POW,
        }

        if op in op_map:
            self.emit(op_map[op])
        else:
            raise ValueError(f"Unknown binary operator: {op}")

    def compile_comparison_op(self, op: str, left, right):
        """Compile a comparison operation"""
        right.compile(self)
        left.compile(self)

        op_map = {
            ">": Opcode.GT,
            "<": Opcode.LT,
            ">=": Opcode.GE,
            "<=": Opcode.LE,
            "==": Opcode.EQ,
            "!=": Opcode.NE,
        }

        if op in op_map:
            self.emit(op_map[op])
        else:
            raise ValueError(f"Unknown comparison operator: {op}")

    def compile_function_call(self, func_name: str, args):
        """Compile a function call"""
        # Compile arguments (in reverse order for stack)
        for arg in reversed(args):
            arg.compile(self)

        # Emit function opcode
        func_map = {
            "sin": Opcode.SIN,
            "cos": Opcode.COS,
            "tan": Opcode.TAN,
            "asin": Opcode.ASIN,
            "acos": Opcode.ACOS,
            "atan": Opcode.ATAN,
            "sqrt": Opcode.SQRT,
            "abs": Opcode.ABS,
            "floor": Opcode.FLOOR,
            "ceil": Opcode.CEIL,
            "round": Opcode.ROUND,
            "sign": Opcode.SIGN,
            "exp": Opcode.EXP,
            "log": Opcode.LOG,
            "log10": Opcode.LOG10,
            "min": Opcode.MIN,
            "max": Opcode.MAX,
            "rand": Opcode.RAND,
        }

        if func_name in func_map:
            self.emit(func_map[func_name])
        else:
            raise ValueError(f"Unknown function: {func_name}")

    def compile_ternary(self, cond, true_val, false_val):
        """Compile a ternary expression: cond ? true_val : false_val"""
        false_val.compile(self)
        true_val.compile(self)
        cond.compile(self)
        self.emit(Opcode.TERNARY)

    def compile_assignment(self, var_name: str, expr):
        """Compile an assignment: var = expr"""
        expr.compile(self)
        var_idx = self._get_var_index(var_name)
        self.emit(Opcode.STORE, var_idx)

    def get_bytecode(self) -> bytes:
        """Get compiled bytecode as bytes"""
        # Pack bytecode as 32-bit integers
        return struct.pack(f"{len(self.bytecode)}i", *self.bytecode)

    def get_constants(self) -> bytes:
        """Get constants as float bytes"""
        return struct.pack(f"{len(self.constants)}f", *self.constants)

    def get_var_count(self) -> int:
        """Get total number of variables"""
        return self.next_var_index


class ASTNode:
    """Base class for AST nodes"""

    def compile(self, compiler: BytecodeCompiler):
        raise NotImplementedError


class NumberNode(ASTNode):
    def __init__(self, value: float):
        self.value = value

    def compile(self, compiler: BytecodeCompiler):
        compiler.compile_number(self.value)


class VariableNode(ASTNode):
    def __init__(self, name: str):
        self.name = name

    def compile(self, compiler: BytecodeCompiler):
        compiler.compile_variable(self.name)


class BinaryOpNode(ASTNode):
    def __init__(self, op: str, left: ASTNode, right: ASTNode):
        self.op = op
        self.left = left
        self.right = right

    def compile(self, compiler: BytecodeCompiler):
        compiler.compile_binary_op(self.op, self.left, self.right)


class ComparisonOpNode(ASTNode):
    def __init__(self, op: str, left: ASTNode, right: ASTNode):
        self.op = op
        self.left = left
        self.right = right

    def compile(self, compiler: BytecodeCompiler):
        compiler.compile_comparison_op(self.op, self.left, self.right)


class FunctionCallNode(ASTNode):
    def __init__(self, func_name: str, args: list[ASTNode]):
        self.func_name = func_name
        self.args = args

    def compile(self, compiler: BytecodeCompiler):
        compiler.compile_function_call(self.func_name, self.args)


class TernaryNode(ASTNode):
    def __init__(self, cond: ASTNode, true_val: ASTNode, false_val: ASTNode):
        self.cond = cond
        self.true_val = true_val
        self.false_val = false_val

    def compile(self, compiler: BytecodeCompiler):
        compiler.compile_ternary(self.cond, self.true_val, self.false_val)


class AssignmentNode(ASTNode):
    def __init__(self, var_name: str, expr: ASTNode):
        self.var_name = var_name
        self.expr = expr

    def compile(self, compiler: BytecodeCompiler):
        compiler.compile_assignment(self.var_name, self.expr)
