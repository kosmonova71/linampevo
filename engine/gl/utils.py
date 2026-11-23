"""
OpenGL utility functions for resource management
"""

from OpenGL import GL


def safe_delete_gl_resources(
    vao=None,
    vbo=None,
    ebo=None,
    program=None,
    textures=None,
    framebuffers=None,
    renderbuffers=None,
):
    """
    Safely delete OpenGL resources with validation to prevent GLError 1281.

    Args:
        vao: Vertex array object or list of VAOs
        vbo: Vertex buffer object or list of VBOs
        ebo: Element buffer object or list of EBOs
        program: Shader program or list of programs
        textures: Texture IDs or list of texture IDs
        framebuffers: Framebuffer objects or list of FBOs
        renderbuffers: Renderbuffer objects or list of RBOs
    """

    def _delete_if_valid(delete_func, validator_func, resources):
        """Helper to delete resources if they are valid OpenGL objects"""
        if resources is None:
            return

        # Convert single resource to list
        if not isinstance(resources, (list, tuple)):
            resources = [resources]

        # Filter out None values and convert to list for OpenGL functions
        valid_resources = []
        for resource in resources:
            if resource and validator_func(resource):
                valid_resources.append(resource)

        if valid_resources:
            delete_func(len(valid_resources), valid_resources)

    try:
        # Delete vertex arrays
        _delete_if_valid(GL.glDeleteVertexArrays, GL.glIsVertexArray, vao)

        # Delete buffers (VBO, EBO)
        _delete_if_valid(GL.glDeleteBuffers, GL.glIsBuffer, vbo)
        _delete_if_valid(GL.glDeleteBuffers, GL.glIsBuffer, ebo)

        # Delete shader programs
        _delete_if_valid(GL.glDeleteProgram, GL.glIsProgram, program)

        # Delete textures
        _delete_if_valid(GL.glDeleteTextures, GL.glIsTexture, textures)

        # Delete framebuffers
        _delete_if_valid(GL.glDeleteFramebuffers, GL.glIsFramebuffer, framebuffers)

        # Delete renderbuffers
        if renderbuffers:
            _delete_if_valid(GL.glDeleteRenderbuffers, GL.glIsRenderbuffer, renderbuffers)

    except Exception:
        # Silently ignore cleanup errors to prevent crashes during shutdown
        pass


def cleanup_vao_vbo(vao=None, vbo=None):
    """Convenience function for common VAO/VBO cleanup"""
    safe_delete_gl_resources(vao=vao, vbo=vbo)


def cleanup_shader_program(program=None):
    """Convenience function for shader program cleanup"""
    safe_delete_gl_resources(program=program)
