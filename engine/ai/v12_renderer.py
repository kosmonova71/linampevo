"""
V12 Renderer - GPU-native visual synthesis engine
Manages bytecode compilation, GPU buffer management, and VM execution
"""

import logging

from OpenGL import GL

from .v12_bytecode_vm import (
    AssignmentNode,
    BinaryOpNode,
    BytecodeCompiler,
    ComparisonOpNode,
    FunctionCallNode,
    NumberNode,
    TernaryNode,
    VariableNode,
)

logger = logging.getLogger(__name__)


class V12Renderer:
    """V12 GPU bytecode VM renderer"""

    def __init__(self):
        self.compiler = BytecodeCompiler()

        # GPU buffer IDs
        self.bytecode_buffer = 0
        self.constant_buffer = 0
        self.variable_buffer = 0

        # Shader program
        self.program = 0

        # Uniform locations
        self.u_time_loc = -1
        self.u_resolution_loc = -1
        self.u_audio_loc = -1
        self.u_frame_loc = -1
        self.u_per_frame_pc_loc = -1
        self.u_per_pixel_pc_loc = -1

        # Execution pointers
        self.per_frame_pc = -1
        self.per_pixel_pc = -1

        # Bytecode storage
        self.bytecode_data: list[int] = []
        self.constant_data: list[float] = []
        self.variable_data: list[float] = []

        # Preset state
        self.current_preset = None

        # Initialize
        self._init_buffers()
        self._init_shader()

    def _init_buffers(self):
        """Initialize GPU buffers for bytecode, constants, and variables"""
        # V12: Use uniform arrays instead of SSBOs for OpenGL 3.3 compatibility
        # Bytecode, constants, and variables will be uploaded as uniforms
        # No buffer objects needed for OpenGL 3.3

        print("V12Renderer: Using uniform arrays for OpenGL 3.3 compatibility")

    def _init_shader(self):
        """Initialize the V12 VM shader"""
        # Load shader source
        try:
            with open("/home/shadowyfigure/Music/shaders/v12_vm_shader.glsl") as f:
                shader_source = f.read()
        except FileNotFoundError:
            print("V12Renderer: ERROR - v12_vm_shader.glsl not found")
            return

        # Compile vertex shader
        vs = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vs, shader_source)
        GL.glCompileShader(vs)

        if not GL.glGetShaderiv(vs, GL.GL_COMPILE_STATUS):
            error = GL.glGetShaderInfoLog(vs)
            print(f"V12Renderer: Vertex shader error: {error}")
            GL.glDeleteShader(vs)
            return

        # Compile fragment shader (same as vertex for now)
        fs = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        # Create simple fragment shader for now
        fs_source = """
#version 330 core
in vec2 v_texcoord;
out vec4 FragColor;
void main() {
    FragColor = vec4(v_texcoord, 0.0, 1.0);
}
"""
        GL.glShaderSource(fs, fs_source)
        GL.glCompileShader(fs)

        if not GL.glGetShaderiv(fs, GL.GL_COMPILE_STATUS):
            error = GL.glGetShaderInfoLog(fs)
            print(f"V12Renderer: Fragment shader error: {error}")
            GL.glDeleteShader(vs)
            GL.glDeleteShader(fs)
            return

        # Link program
        self.program = GL.glCreateProgram()
        GL.glAttachShader(self.program, vs)
        GL.glAttachShader(self.program, fs)
        GL.glLinkProgram(self.program)

        if not GL.glGetProgramiv(self.program, GL.GL_LINK_STATUS):
            error = GL.glGetProgramInfoLog(self.program)
            print(f"V12Renderer: Program link error: {error}")
            GL.glDeleteProgram(self.program)
            self.program = 0
            return

        # Get uniform locations
        self.u_time_loc = GL.glGetUniformLocation(self.program, "u_time")
        self.u_resolution_loc = GL.glGetUniformLocation(self.program, "u_resolution")
        self.u_audio_loc = GL.glGetUniformLocation(self.program, "u_audio")
        self.u_frame_loc = GL.glGetUniformLocation(self.program, "u_frame")
        self.u_per_frame_pc_loc = GL.glGetUniformLocation(self.program, "u_per_frame_pc")
        self.u_per_pixel_pc_loc = GL.glGetUniformLocation(self.program, "u_per_pixel_pc")

        # Clean up shaders
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)

        print("V12Renderer: Shader program initialized")

    def parse_milkdrop_equation(self, equation: str) -> AssignmentNode | None:
        """Parse a Milkdrop equation into an AST node"""
        equation = equation.strip()
        if not equation or "=" not in equation:
            return None

        # Split assignment
        parts = equation.split("=", 1)
        var_name = parts[0].strip()
        expr_str = parts[1].strip()

        # Parse expression (simple recursive descent for now)
        try:
            expr = self._parse_expression(expr_str)
            if expr:
                return AssignmentNode(var_name, expr)
        except Exception:
            print(f"V12Renderer: Failed to parse equation: {equation}")
            logger.debug("Parse failure details", exc_info=True)

        return None

    def _parse_expression(self, expr: str):
        """Parse an expression string into an AST node (simplified)"""
        expr = expr.strip()

        # Try to parse as number
        try:
            value = float(expr)
            return NumberNode(value)
        except ValueError:
            pass

        # Try to parse as variable
        if expr.replace("_", "").replace(".", "").isalnum():
            return VariableNode(expr)

        # Try to parse as function call
        if "(" in expr and ")" in expr:
            func_name = expr[: expr.index("(")]
            args_str = expr[expr.index("(") + 1 : expr.rindex(")")]
            args = []
            if args_str.strip():
                # Simple comma-separated args
                for arg in args_str.split(","):
                    arg = arg.strip()
                    if arg:
                        parsed = self._parse_expression(arg)
                        if parsed:
                            args.append(parsed)
            return FunctionCallNode(func_name, args)

        # Try to parse as binary operation (very simplified)
        for op in ["+", "-", "*", "/", "%", "^"]:
            if op in expr:
                parts = expr.split(op, 1)
                if len(parts) == 2:
                    left = self._parse_expression(parts[0])
                    right = self._parse_expression(parts[1])
                    if left and right:
                        return BinaryOpNode(op, left, right)

        # Try to parse as comparison
        for op in [">", "<", ">=", "<=", "==", "!="]:
            if op in expr:
                parts = expr.split(op, 1)
                if len(parts) == 2:
                    left = self._parse_expression(parts[0])
                    right = self._parse_expression(parts[1])
                    if left and right:
                        return ComparisonOpNode(op, left, right)

        # Try to parse as ternary
        if "?" in expr and ":" in expr:
            parts = expr.split("?", 1)
            if len(parts) == 2:
                cond = self._parse_expression(parts[0])
                value_parts = parts[1].split(":", 1)
                if len(value_parts) == 2 and cond:
                    true_val = self._parse_expression(value_parts[0])
                    false_val = self._parse_expression(value_parts[1])
                    if true_val and false_val:
                        return TernaryNode(cond, true_val, false_val)

        print(f"V12Renderer: Could not parse expression: {expr}")
        return None

    def compile_preset(self, per_frame_eqs: list[str], per_pixel_eqs: list[str]) -> bool:
        """Compile a preset to bytecode"""
        print("V12Renderer: Compiling preset to bytecode...")

        # Reset compiler
        self.compiler = BytecodeCompiler()

        # Compile per-frame equations
        per_frame_start = len(self.compiler.bytecode)
        for eq in per_frame_eqs:
            ast = self.parse_milkdrop_equation(eq)
            if ast:
                ast.compile(self.compiler)

        per_frame_end = len(self.compiler.bytecode)
        self.per_frame_pc = per_frame_start if per_frame_end > per_frame_start else -1

        # Compile per-pixel equations
        per_pixel_start = len(self.compiler.bytecode)
        for eq in per_pixel_eqs:
            ast = self.parse_milkdrop_equation(eq)
            if ast:
                ast.compile(self.compiler)

        per_pixel_end = len(self.compiler.bytecode)
        self.per_pixel_pc = per_pixel_start if per_pixel_end > per_pixel_start else -1

        # Get bytecode and data
        self.bytecode_data = self.compiler.bytecode
        self.constant_data = self.compiler.constants
        self.variable_data = [0.0] * self.compiler.get_var_count()

        # Upload to GPU
        self._upload_to_gpu()

        print(f"V12Renderer: Preset compiled - {len(self.bytecode_data)} bytecode instructions")
        print(f"V12Renderer: per_frame PC: {self.per_frame_pc}, per_pixel PC: {self.per_pixel_pc}")
        print(
            f"V12Renderer: Variables: {len(self.variable_data)}, Constants: {len(self.constant_data)}"
        )

        return True

    def _upload_to_gpu(self):
        """Upload bytecode, constants, and variables to GPU as uniform arrays"""
        # Use glUniform for uniform arrays (OpenGL 3.3 compatible)
        GL.glUseProgram(self.program)

        # Pad arrays to fixed sizes
        max_bytecode = 1024
        max_constants = 256
        max_variables = 128

        # Pad bytecode
        bytecode_padded = self.bytecode_data + [0] * (max_bytecode - len(self.bytecode_data))
        # Pad constants
        constants_padded = self.constant_data + [0.0] * (max_constants - len(self.constant_data))
        # Pad variables
        variables_padded = self.variable_data + [0.0] * (max_variables - len(self.variable_data))

        # Upload bytecode as int array
        bytecode_loc = GL.glGetUniformLocation(self.program, "u_bytecode")
        GL.glUniform1iv(bytecode_loc, max_bytecode, bytecode_padded)

        # Upload constants as float array
        constants_loc = GL.glGetUniformLocation(self.program, "u_constants")
        GL.glUniform1fv(constants_loc, max_constants, constants_padded)

        # Upload variables as float array
        vars_loc = GL.glGetUniformLocation(self.program, "u_vars")
        GL.glUniform1fv(vars_loc, max_variables, variables_padded)

    def render(self, time: float, resolution: tuple, audio: tuple, frame: float):
        """Render a frame using the VM"""
        if self.program == 0:
            return

        GL.glUseProgram(self.program)

        # Set uniforms
        GL.glUniform1f(self.u_time_loc, time)
        GL.glUniform2f(self.u_resolution_loc, resolution[0], resolution[1])
        GL.glUniform2f(self.u_audio_loc, audio[0], audio[1])
        GL.glUniform1f(self.u_frame_loc, frame)
        GL.glUniform1i(self.u_per_frame_pc_loc, self.per_frame_pc)
        GL.glUniform1i(self.u_per_pixel_pc_loc, self.per_pixel_pc)

        # Draw full-screen quad
        # (This would need proper vertex buffer setup in actual implementation)
        # For now, just a placeholder

    def cleanup(self):
        """Clean up GPU resources"""
        if self.bytecode_buffer:
            GL.glDeleteBuffers(1, [self.bytecode_buffer])
        if self.constant_buffer:
            GL.glDeleteBuffers(1, [self.constant_buffer])
        if self.variable_buffer:
            GL.glDeleteBuffers(1, [self.variable_buffer])
        if self.program:
            GL.glDeleteProgram(self.program)

        print("V12Renderer: Cleaned up GPU resources")
