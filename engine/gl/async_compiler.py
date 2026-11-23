#!/usr/bin/env python3
"""
ASCEND V9 - Async Shader Compilation System

Zero-stutter shader compilation using worker thread.
Compile queue → worker thread → GL compile → ready queue.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from engine.gl.glsl_generator import GLSLGenerator
from engine.ir.ssa import IRBlock


@dataclass
class CompileRequest:
    """Shader compilation request."""

    ir: IRBlock
    version: str
    callback: Callable[[str], None] | None = None
    request_id: int = 0


@dataclass
class CompileResult:
    """Shader compilation result."""

    request_id: int
    glsl: str | None
    success: bool
    error: str | None = None


class AsyncShaderCompiler:
    """Async shader compiler with worker thread.

    Compiles shaders in background thread to prevent stutter.
    Results are queued for main thread to retrieve.
    """

    def __init__(self, generator: GLSLGenerator | None = None):
        """Initialize async compiler.

        Args:
            generator: GLSL generator instance (created if None)
        """
        self.generator = generator or GLSLGenerator()

        # Thread-safe queues
        self.compile_queue: queue.Queue[CompileRequest] = queue.Queue()
        self.ready_queue: queue.Queue[CompileResult] = queue.Queue()

        # Worker thread
        self.worker_thread: threading.Thread | None = None
        self.running = False
        self.request_counter = 0

        # Start worker thread
        self.start()

    def start(self) -> None:
        """Start worker thread."""
        if self.running:
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def stop(self) -> None:
        """Stop worker thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)

    def _worker_loop(self) -> None:
        """Worker thread main loop."""
        while self.running:
            try:
                # Get request with timeout to allow thread to stop
                request = self.compile_queue.get(timeout=0.1)

                # Compile shader
                try:
                    glsl = self.generator.generate(request.ir, request.version)
                    result = CompileResult(request_id=request.request_id, glsl=glsl, success=True)
                except Exception as e:
                    result = CompileResult(
                        request_id=request.request_id, glsl=None, success=False, error=str(e)
                    )

                # Put result in ready queue
                self.ready_queue.put(result)

                # Call callback if provided
                if request.callback and result.success:
                    request.callback(result.glsl)

            except Exception:
                continue

    def compile_async(
        self, ir: IRBlock, version: str = "330 core", callback: Callable[[str], None] | None = None
    ) -> int:
        """Submit shader for async compilation.

        Args:
            ir: SSA IR block
            version: GLSL version
            callback: Optional callback when shader is ready

        Returns:
            Request ID for tracking
        """
        self.request_counter += 1
        request_id = self.request_counter

        request = CompileRequest(ir=ir, version=version, callback=callback, request_id=request_id)

        self.compile_queue.put(request)
        return request_id

    def poll_result(self) -> CompileResult | None:
        """Check for completed compilation.

        Returns:
            CompileResult if available, None otherwise
        """
        try:
            return self.ready_queue.get_nowait()
        except Exception:
            return None

    def wait_for_result(self, request_id: int, timeout: float = 5.0) -> CompileResult | None:
        """Wait for specific compilation result.

        Args:
            request_id: Request ID to wait for
            timeout: Maximum time to wait

        Returns:
            CompileResult if available, None if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            result = self.poll_result()
            if result and result.request_id == request_id:
                return result
            time.sleep(0.01)

        return None

    def get_queue_size(self) -> dict[str, int]:
        """Get queue sizes for monitoring.

        Returns:
            Dictionary with queue sizes
        """
        return {
            "compile_queue": self.compile_queue.qsize(),
            "ready_queue": self.ready_queue.qsize(),
        }
