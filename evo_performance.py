from __future__ import annotations

import logging
import math
import os
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass

from OpenGL import GL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Persistent mapped PBOs — eliminate buffer allocation churn
# ---------------------------------------------------------------------------


class PersistentMappedPBO:
    """A pixel-pack buffer that stays mapped for the lifetime of the GL context.

    Uses ``GL_ARB_buffer_storage`` with ``GL_MAP_PERSISTENT_BIT`` and
    ``GL_MAP_COHERENT_BIT`` so the CPU never re-maps or re-allocates the
    buffer.  Reads are taken from the mapped pointer directly; writes are
    scheduled with ``glReadPixels`` into the buffer object.

    The triple-buffering pattern (3 PBOs rotating) is preserved so the GPU
    can fill one PBO while the CPU reads from another.
    """

    def __init__(self, res: int, n_buffers: int = 3):
        self.res = max(1, int(res))
        self.n_buffers = max(1, int(n_buffers))
        self._pbos: list[int] = []
        self._mapped: list[memoryview | None] = []
        self._write_idx = 0
        self._write_count = 0
        self._ready = False
        self._nbytes = self.res * self.res * 4

    def init_gl(self) -> bool:
        try:
            self._pbos = list(GL.glGenBuffers(self.n_buffers))
            for p in self._pbos:
                GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, p)
                GL.glBufferStorage(
                    GL.GL_PIXEL_PACK_BUFFER,
                    self._nbytes,
                    None,
                    GL.GL_MAP_READ_BIT | GL.GL_MAP_PERSISTENT_BIT | GL.GL_MAP_COHERENT_BIT,
                )
                ptr = GL.glMapBufferRange(
                    GL.GL_PIXEL_PACK_BUFFER,
                    0,
                    self._nbytes,
                    GL.GL_MAP_READ_BIT | GL.GL_MAP_PERSISTENT_BIT | GL.GL_MAP_COHERENT_BIT,
                )
                if ptr is not None:
                    self._mapped.append(memoryview(ptr))
                else:
                    self._mapped.append(None)
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
            self._ready = all(m is not None for m in self._mapped)
            return self._ready
        except Exception as _e:
            logger.warning("PersistentMappedPBO init failed: %s", _e)
            self._cleanup()
            return False

    def _cleanup(self) -> None:
        for i, p in enumerate(self._pbos):
            try:
                if self._mapped[i] is not None:
                    # glUnmapBuffer acts on the currently bound buffer, so
                    # bind each PBO individually before unmapping it.
                    GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, p)
                    GL.glUnmapBuffer(GL.GL_PIXEL_PACK_BUFFER)
            except Exception as _e:
                logger.warning("PBO unmap failed: %s", _e)
            try:
                GL.glDeleteBuffers(1, [p])
            except Exception as _e:
                logger.warning("PBO delete failed: %s", _e)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
        self._pbos.clear()
        self._mapped.clear()
        self._ready = False

    def cleanup(self) -> None:
        self._cleanup()

    def read_into_pbo(self) -> None:
        if not self._ready or not self._pbos:
            return
        pbo = self._pbos[self._write_idx % self.n_buffers]
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, pbo)
        GL.glReadPixels(0, 0, self.res, self.res, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, 0)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
        self._write_idx = (self._write_idx + 1) % self.n_buffers
        self._write_count += 1

    def collect(self) -> bytes | None:
        if not self._ready or not self._pbos:
            return None
        # Need at least n_buffers - 1 completed writes before any slot
        # is guaranteed to be finished on the GPU.  With 3 PBOs that
        # means we need at least 2 writes before reading "two frames ago".
        if self._write_count < self.n_buffers - 1:
            return None
        # The slot two writes ago: (write_idx - 2) % n_buffers.
        read_idx = (self._write_idx - 2) % self.n_buffers
        buf = self._mapped[read_idx]
        if buf is None:
            return None
        return bytes(buf[: self._nbytes])

    @property
    def ready(self) -> bool:
        return self._ready


# ---------------------------------------------------------------------------
# 2. Shader binary caching via glProgramBinary
# ---------------------------------------------------------------------------


class ShaderBinaryCache:
    """Wraps glProgramBinary / glGetProgramBinary for a single program.

    On contexts that expose GL_ARB_get_program_binary the compiled
    program binary is extracted after linking and stored on disk.
    Subsequent sessions attempt glProgramBinary before re-linking,
    skipping the expensive compile+link step entirely.

    The cache key is ``(driver_fingerprint, source_hash)`` so a
    binary compiled for one driver/version is never reused on
    another.
    """

    def __init__(self, cache_dir: str = ".shader_cache", max_entries: int = 512):
        self._cache_dir = cache_dir
        self._max_entries = max_entries
        self._available = False
        self._binary_format = 0
        self._driver_id = ""
        self._index: dict[str, bytes] = {}
        self._access: dict[str, float] = {}
        self._init_driver()

    def _init_driver(self) -> None:
        try:
            ext = GL.glGetString(GL.GL_EXTENSIONS)
            if ext is None:
                return
            exts = ext.decode("utf-8", errors="ignore") if isinstance(ext, bytes) else str(ext)
            if "GL_ARB_get_program_binary" not in exts.split():
                return
            # A driver may advertise the extension but expose zero binary
            # formats; treat that as unavailable.
            try:
                n_formats = GL.glGetIntegeriv(GL.GL_NUM_PROGRAM_BINARY_FORMATS, None)
                if not n_formats or int(n_formats[0]) <= 0:
                    return
            except Exception:
                return
            self._available = True
            self._driver_id = self._fingerprint()
        except Exception as _e:
            logger.warning("GPUProgram init failed: %s", _e)

    @staticmethod
    def _fingerprint() -> str:
        try:
            parts = [
                str(GL.glGetString(GL.GL_RENDERER) or ""),
                str(GL.glGetString(GL.GL_VERSION) or ""),
                str(GL.glGetString(GL.GL_VENDOR) or ""),
            ]
            import hashlib

            return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
        except Exception as _e:
            logger.warning("Fingerprint failed: %s", _e)
            return "unknown"

    def _key(self, source_hash: str) -> str:
        return f"{self._driver_id}:{source_hash}"

    def try_load(self, program: int, source_hash: str) -> bool:
        """Attempt to restore a cached binary into *program*.

        Returns True on success (the program is ready to use).
        """
        if not self._available:
            return False
        key = self._key(source_hash)
        binary = self._index.get(key)
        if binary is None:
            # Cold start: fall back to the persisted copy on disk.
            binary = self.load_persisted(source_hash)
        if binary is None:
            return False
        try:
            # glProgramBinary takes (program, format, binary, length).
            # Passing the length explicitly keeps the PyOpenGL binding happy
            # regardless of how the bytes object is wrapped.
            GL.glProgramBinary(program, self._binary_format, binary, len(binary))
            status = GL.glGetProgramiv(program, GL.GL_LINK_STATUS)
            if status:
                self._touch(key)
                return True
        except Exception as _e:
            logger.warning("Load failed: %s", _e)
        return False

    def store(self, program: int, source_hash: str) -> None:
        """Extract the binary from *program* and persist it."""
        if not self._available:
            return
        try:
            length = GL.glGetProgramiv(program, GL.GL_PROGRAM_BINARY_LENGTH)
            if length <= 0:
                return
            result = GL.glGetProgramBinary(program, length)
            if result is None:
                return
            if isinstance(result, tuple):
                fmt, data = int(result[0]), bytes(result[1])
            else:
                return
            self._binary_format = fmt
            key = self._key(source_hash)
            self._index[key] = data
            self._touch(key)
            self._persist(key, data, fmt)
        except Exception as _e:
            logger.warning("Store failed: %s", _e)

    def _touch(self, key: str) -> None:
        self._access[key] = time.time()
        if len(self._access) > self._max_entries:
            oldest = min(self._access, key=self._access.get)
            self._access.pop(oldest, None)
            self._index.pop(oldest, None)

    def _persist(self, key: str, data: bytes, fmt: int) -> None:
        try:
            path = os.path.join(self._cache_dir, "binaries")
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, f"{key}.bin"), "wb") as f:
                f.write(struct.pack("<I", fmt))
                f.write(struct.pack("<Q", len(data)))
                f.write(data)
        except Exception as _e:
            logger.warning("Persist failed: %s", _e)

    def load_persisted(self, source_hash: str) -> bytes | None:
        try:
            key = self._key(source_hash)
            path = os.path.join(self._cache_dir, "binaries", f"{key}.bin")
            if not os.path.exists(path):
                return None
            with open(path, "rb") as f:
                fmt = struct.unpack("<I", f.read(4))[0]
                length = struct.unpack("<Q", f.read(8))[0]
                data = f.read(length)
            self._binary_format = fmt
            self._index[key] = data
            self._touch(key)
            return data
        except Exception as _e:
            logger.warning("Load persisted failed: %s", _e)
            return None

    @property
    def available(self) -> bool:
        return self._available


# ---------------------------------------------------------------------------
# 3. Bindless textures (GL_ARB_bindless_texture)
# ---------------------------------------------------------------------------


class BindlessTexture:
    """A texture handle that can be used without binding.

    Requires ``GL_ARB_bindless_texture``.  The handle is
    obtained once and stored as a 64-bit integer; shaders
    receive it as a uniform ``uint64_t`` and sample with
    ``texture(sampler2D(handle, index), uv)``.

    This removes the per-organism ``glActiveTexture`` /
    ``glBindTexture`` calls that dominate the per-frame
    overhead when many organisms are rendered.
    """

    def __init__(self):
        self._handle: int = 0
        self._tex: int = 0
        self._available = False

    def init(self, tex: int, target: int = GL.GL_TEXTURE_2D) -> bool:
        try:
            if not self._bindless_available():
                return False
            self._tex = tex
            self._handle = GL.glGetTextureHandleARB(tex)
            if self._handle == 0:
                return False
            self.make_resident()
            return True
        except Exception as _e:
            logger.warning("Operation failed: %s", _e)
            return False

    @staticmethod
    def _bindless_available() -> bool:
        """True if GL_ARB_bindless_texture is present."""
        if not hasattr(GL, "glGetTextureHandleARB"):
            return False
        try:
            ext = GL.glGetString(GL.GL_EXTENSIONS)
            if ext is None:
                return False
            exts = ext.decode("utf-8", errors="ignore") if isinstance(ext, bytes) else str(ext)
            return "GL_ARB_bindless_texture" in exts.split()
        except Exception:
            return True  # extension funcs exist; assume present

    def make_resident(self) -> bool:
        """(Re)make the handle resident after a context loss.

        Call this again after the GL context has been recreated.
        """
        if self._handle == 0:
            return False
        try:
            GL.glMakeTextureHandleResidentARB(self._handle)
            self._available = True
            return True
        except Exception as _e:
            logger.warning("Make resident failed: %s", _e)
            return False

    def handle(self) -> int:
        return self._handle

    def release(self) -> None:
        if self._available and self._handle != 0:
            try:
                GL.glMakeTextureHandleNonResidentARB(self._handle)
            except Exception as _e:
                logger.warning("Texture handle release failed: %s", _e)
            self._handle = 0
        self._available = False

    def cleanup(self) -> None:
        self.release()

    @property
    def available(self) -> bool:
        return self._available


class BindlessTextureArray:
    """A 2D texture array where each layer is a bindless texture.

    For ecosystem maps we pack per-organism tiles into a single
    texture array and expose the layer index to the shader.
    The array itself is bound once; organisms only pass their
    layer index as a uniform.
    """

    def __init__(self):
        self._tex: int = 0
        self._handle: int = 0
        self._layers: int = 0
        self._available = False

    def init(
        self,
        layers: int,
        width: int,
        height: int,
        internal_fmt: int = GL.GL_RGBA16F,
        format: int = GL.GL_RGBA,
        dtype: int = GL.GL_FLOAT,
    ) -> bool:
        try:
            if not BindlessTexture._bindless_available():
                return False
            self._layers = max(1, int(layers))
            self._internal_fmt = internal_fmt
            self._format = format
            self._dtype = dtype
            self._tex = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, self._tex)
            GL.glTexImage3D(
                GL.GL_TEXTURE_2D_ARRAY,
                0,
                internal_fmt,
                width,
                height,
                self._layers,
                0,
                format,
                dtype,
                None,
            )
            GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, 0)
            self._handle = GL.glGetTextureHandleARB(self._tex)
            if self._handle == 0:
                return False
            self.make_resident()
            return True
        except Exception as _e:
            logger.warning("Operation failed: %s", _e)
            return False

    def handle(self) -> int:
        return self._handle

    def make_resident(self) -> bool:
        """(Re)make the handle resident after a context loss."""
        if self._handle == 0:
            return False
        try:
            GL.glMakeTextureHandleResidentARB(self._handle)
            self._available = True
            return True
        except Exception as _e:
            logger.warning("Make resident failed: %s", _e)
            return False

    def update_layer(
        self,
        layer: int,
        data: bytes,
        width: int,
        height: int,
        format: int | None = None,
        dtype: int | None = None,
    ) -> None:
        if not self._available or layer >= self._layers:
            return
        GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, self._tex)
        GL.glTexSubImage3D(
            GL.GL_TEXTURE_2D_ARRAY,
            0,
            0,
            0,
            layer,
            width,
            height,
            1,
            format if format is not None else self._format,
            dtype if dtype is not None else self._dtype,
            data,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, 0)

    def release(self) -> None:
        if self._available and self._handle != 0:
            try:
                GL.glMakeTextureHandleNonResidentARB(self._handle)
            except Exception as _e:
                logger.warning("Texture handle release failed: %s", _e)
            self._handle = 0
        if self._tex != 0:
            try:
                GL.glDeleteTextures(1, [self._tex])
            except Exception as _e:
                logger.warning("Texture deletion failed: %s", _e)
            self._tex = 0
        self._available = False

    def cleanup(self) -> None:
        self.release()

    @property
    def available(self) -> bool:
        return self._available


# ---------------------------------------------------------------------------
# 4. Multi-draw indirect for organism rendering
# ---------------------------------------------------------------------------


@dataclass
class DrawCommand:
    """One entry in a GL_DRAW_INDIRECT_BUFFER."""

    count: int = 6  # vertices per quad
    instance_count: int = 1
    first_vertex: int = 0
    base_instance: int = 0


class MultiDrawIndirect:
    """Batches per-organism draw calls into a single
    ``glMultiDrawArraysIndirect`` call.

    Each organism is one draw command in a SSBO.  The
    shader picks up per-org uniforms from a matching
    SSBO (see :class:`PersistentSSBO`).  This reduces
    N ``glDrawArrays`` calls to 1.

    The command buffer is persistent-mapped so it can
    be updated in-place without re-allocating.
    """

    def __init__(self, max_organisms: int = 64):
        self.max_organisms = max(1, int(max_organisms))
        self._cmd_ssbo: int = 0
        self._cmd_buf: memoryview | None = None
        self._n_commands: int = 0
        self._groups: dict[int, tuple[int, int]] = {}
        self._available = False

    def init_gl(self) -> bool:
        try:
            if not hasattr(GL, "glGenBuffers") or not hasattr(GL, "glBindBufferBase"):
                return False
            self._cmd_ssbo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_DRAW_INDIRECT_BUFFER, self._cmd_ssbo)
            GL.glBufferStorage(
                GL.GL_DRAW_INDIRECT_BUFFER,
                self.max_organisms * 16,  # 4 x uint32 per command
                None,
                GL.GL_MAP_WRITE_BIT | GL.GL_MAP_PERSISTENT_BIT | GL.GL_MAP_COHERENT_BIT,
            )
            ptr = GL.glMapBufferRange(
                GL.GL_DRAW_INDIRECT_BUFFER,
                0,
                self.max_organisms * 16,
                GL.GL_MAP_WRITE_BIT | GL.GL_MAP_PERSISTENT_BIT | GL.GL_MAP_COHERENT_BIT,
            )
            if ptr is None:
                GL.glDeleteBuffers(1, [self._cmd_ssbo])
                self._cmd_ssbo = 0
                return False
            self._cmd_buf = memoryview(ptr)
            self._available = True
            GL.glBindBuffer(GL.GL_DRAW_INDIRECT_BUFFER, 0)
            return True
        except Exception as _e:
            logger.warning("MultiDrawIndirect init failed: %s", _e)
            self._cleanup()
            return False

    def _cleanup(self) -> None:
        if self._cmd_ssbo != 0:
            if self._cmd_buf is not None:
                try:
                    # glUnmapBuffer acts on the currently bound buffer,
                    # so bind before unmapping.
                    GL.glBindBuffer(GL.GL_DRAW_INDIRECT_BUFFER, self._cmd_ssbo)
                    GL.glUnmapBuffer(GL.GL_DRAW_INDIRECT_BUFFER)
                except Exception as _e:
                    logger.warning("SSBO unmap failed: %s", _e)
                self._cmd_buf = None
            try:
                GL.glDeleteBuffers(1, [self._cmd_ssbo])
            except Exception as _e:
                logger.warning("SSBO deletion failed: %s", _e)
            self._cmd_ssbo = 0
        GL.glBindBuffer(GL.GL_DRAW_INDIRECT_BUFFER, 0)
        self._available = False

    def cleanup(self) -> None:
        self._cleanup()

    def build_commands(self, organisms: list, programs: list) -> dict[int, list[int]]:
        """Fill the indirect buffer and return command groups.

        Organisms with different programs cannot share a single
        ``glMultiDrawArraysIndirect`` call, so we group them by
        program id.  The returned dict maps ``program -> [org_idx, ...]``
        and the corresponding commands are packed contiguously for each
        program.  Organisms whose program is 0 (failed compile) are
        skipped.
        """
        if not self._available or not self._cmd_buf:
            return {}
        n = min(len(organisms), self.max_organisms)
        # group org indices by program, preserving insertion order
        groups: dict[int, list[int]] = {}
        for i in range(n):
            prog = programs[i] if i < len(programs) else 0
            if prog == 0:
                continue
            groups.setdefault(prog, []).append(i)

        self._n_commands = 0
        offset = 0
        self._groups = {}
        for prog, idxs in groups.items():
            start = self._n_commands
            for i in idxs:
                cmd = DrawCommand(count=6, instance_count=1, first_vertex=0, base_instance=i)
                packed = struct.pack(
                    "<4I", cmd.count, cmd.instance_count, cmd.first_vertex, cmd.base_instance
                )
                self._cmd_buf[offset : offset + 16] = packed
                offset += 16
                self._n_commands += 1
            self._groups[prog] = (start, len(idxs))
        return self._groups

    def draw(
        self,
        vao: int,
        program: int,
        loc_count: int,
        loc_instance: int,
        group_start: int = 0,
        group_count: int = 0,
    ) -> None:
        """Issue a multi-draw call for one program's command group.

        *group_start* and *group_count* select a contiguous slice of
        the packed command buffer; pass 0/0 to draw every command.
        """
        if not self._available or self._n_commands <= 0:
            return
        start = max(0, group_start)
        count = group_count if group_count > 0 else self._n_commands
        if start + count > self._n_commands:
            count = max(0, self._n_commands - start)
        if count <= 0:
            return
        GL.glBindVertexArray(vao)
        GL.glUseProgram(program)
        if loc_count >= 0:
            GL.glUniform1i(loc_count, count)
        if loc_instance >= 0:
            GL.glUniform1i(loc_instance, start)
        GL.glBindBuffer(GL.GL_DRAW_INDIRECT_BUFFER, self._cmd_ssbo)
        GL.glMultiDrawArraysIndirect(GL.GL_TRIANGLES, None, count, start * 16)
        GL.glBindBuffer(GL.GL_DRAW_INDIRECT_BUFFER, 0)
        GL.glUseProgram(0)
        GL.glBindVertexArray(0)

    def draw_all(self, vao: int, programs: list, loc_count: int, loc_instance: int) -> int:
        """Draw every group in the last :meth:`build_commands` batch.

        Returns the number of draw calls issued.
        """
        if not self._available or self._n_commands <= 0:
            return 0
        drawn = 0
        for prog, (start, count) in self._groups.items():
            if prog not in programs:
                continue
            self.draw(vao, prog, loc_count, loc_instance, start, count)
            drawn += 1
        return drawn

    @property
    def available(self) -> bool:
        return self._available


# ---------------------------------------------------------------------------
# 5. Persistent SSBOs — replace per-frame buffer rebuilds
# ---------------------------------------------------------------------------


class PersistentSSBO:
    """A shader storage buffer object that is persistent-mapped.

    The buffer is allocated once with ``GL_BUFFER_STORAGE`` and
    ``GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT``.  The CPU
    writes organism data into the mapped region each frame;
    the GPU reads from the same region without any buffer
    rebuild or ``glBufferSubData`` upload.

    This eliminates the per-frame ``glBufferData`` /
    ``glBufferSubData`` churn that dominates CPU overhead
    when the organism count changes frequently.
    """

    def __init__(self, max_organisms: int = 64, floats_per_org: int = 16):
        self.max_orgs = max(1, int(max_organisms))
        self.floats_per_org = max(1, int(floats_per_org))
        self._nbytes = self.max_orgs * self.floats_per_org * 4
        self._ssbo: int = 0
        self._mapped: memoryview | None = None
        self._available = False

    def init_gl(self) -> bool:
        try:
            self._ssbo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._ssbo)
            GL.glBufferStorage(
                GL.GL_SHADER_STORAGE_BUFFER,
                self._nbytes,
                None,
                GL.GL_MAP_WRITE_BIT | GL.GL_MAP_PERSISTENT_BIT | GL.GL_MAP_COHERENT_BIT,
            )
            ptr = GL.glMapBufferRange(
                GL.GL_SHADER_STORAGE_BUFFER,
                0,
                self._nbytes,
                GL.GL_MAP_WRITE_BIT | GL.GL_MAP_PERSISTENT_BIT | GL.GL_MAP_COHERENT_BIT,
            )
            if ptr is None:
                GL.glDeleteBuffers(1, [self._ssbo])
                self._ssbo = 0
                return False
            self._mapped = memoryview(ptr)
            self._available = True
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            return True
        except Exception as _e:
            logger.warning("PersistentSSBO init failed: %s", _e)
            self._cleanup()
            return False

    def _cleanup(self) -> None:
        if self._mapped is not None and self._ssbo != 0:
            try:
                # glUnmapBuffer acts on the currently bound buffer, so
                # bind each SSBO individually before unmapping it.
                GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._ssbo)
                GL.glUnmapBuffer(GL.GL_SHADER_STORAGE_BUFFER)
            except Exception as _e:
                logger.warning("SSBO unmap failed: %s", _e)
            self._mapped = None
        if self._ssbo != 0:
            try:
                GL.glDeleteBuffers(1, [self._ssbo])
            except Exception as _e:
                logger.warning("SSBO deletion failed: %s", _e)
            self._ssbo = 0
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        self._available = False

    def cleanup(self) -> None:
        self._cleanup()

    def write_organisms(self, organisms: list, get_params: Callable[..., list[float]]) -> int:
        """Write organism parameters into the mapped buffer.

        ``get_params(org) -> List[float]`` is called for each
        organism.  The returned list is truncated or padded to
        ``self.floats_per_org`` floats.  Returns the number of
        organisms actually written.
        """
        if not self._available or self._mapped is None:
            return 0
        n = min(len(organisms), self.max_orgs)
        offset = 0
        for i in range(n):
            params = get_params(organisms[i])
            if params is None:
                params = [0.0] * self.floats_per_org
            flat = list(params)[: self.floats_per_org]
            while len(flat) < self.floats_per_org:
                flat.append(0.0)
            packed = struct.pack(f"<{self.floats_per_org}f", *flat)
            self._mapped[offset : offset + len(packed)] = packed
            offset += len(packed)
        return n

    def bind(self, binding: int = 0) -> None:
        if self._available and self._ssbo != 0:
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, binding, self._ssbo)

    def unbind(self) -> None:
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 0, 0)

    @property
    def available(self) -> bool:
        return self._available

    @property
    def nbytes(self) -> int:
        return self._nbytes


# ---------------------------------------------------------------------------
# 6. GPU timestamp queries — measure per-organism cost
# ---------------------------------------------------------------------------


class GpuTimerQuery:
    """Wraps ``GL_ARB_timer_query`` for GPU-side timing.

    Each organism's rendering pass is wrapped with
    ``glBeginQuery(GL_TIME_ELAPSED)`` / ``glEndQuery``.
    The elapsed time is read back asynchronously so the
    CPU never stalls on the query result.

    The query pool is double-buffered: two interleaved rings of
    ``max_queries`` slots each.  A new ``begin`` is only issued on
    a slot whose previous result has already been collected, so the
    GPU is never asked to restart a query that is still in flight.
    """

    def __init__(self, max_queries: int = 128):
        self.max_queries = max(1, int(max_queries))
        self._queries: list[int] = []
        self._results: list[float | None] = []
        self._in_flight: list[bool] = []
        self._available = False
        self._write_idx = 0
        self._read_idx = 0
        self._have_results = False

    def init_gl(self) -> bool:
        try:
            if not hasattr(GL, "glGenQueries") or not hasattr(GL, "glBeginQuery"):
                return False
            # Double the pool so each in-flight slot has a free partner.
            self._queries = list(GL.glGenQueries(self.max_queries * 2))
            self._results = [None] * (self.max_queries * 2)
            self._in_flight = [False] * (self.max_queries * 2)
            self._available = True
            return True
        except Exception as _e:
            logger.warning("Operation failed: %s", _e)
            return False

    def begin(self, org_idx: int) -> None:
        if not self._available or org_idx >= self.max_queries:
            return
        # Map each organism to a stable slot; the pool is twice the size
        # so the slot two frames ago is a different query object.
        idx = org_idx % len(self._queries)
        if self._in_flight[idx]:
            return  # previous query still outstanding; skip this frame
        self._write_idx = idx
        GL.glBeginQuery(GL.GL_TIME_ELAPSED, self._queries[idx])
        self._in_flight[idx] = True

    def end(self) -> None:
        if not self._available:
            return
        GL.glEndQuery(GL.GL_TIME_ELAPSED)

    def collect(self) -> None:
        """Poll every in-flight query and harvest finished results."""
        if not self._available:
            return
        for idx, in_flight in enumerate(self._in_flight):
            if not in_flight:
                continue
            available = GL.glGetQueryObjectuiv(self._queries[idx], GL.GL_QUERY_RESULT_AVAILABLE)
            if available:
                result = GL.glGetQueryObjectui64(self._queries[idx], GL.GL_QUERY_RESULT)
                self._results[idx] = float(result) / 1_000_000.0  # ns -> ms
                self._have_results = True
                self._in_flight[idx] = False

    def get_result(self, org_idx: int) -> float | None:
        """Return the last collected ms for *org_idx*, or None."""
        idx = org_idx % self.max_queries
        return self._results[idx]

    def get_all_results(self) -> dict[int, float]:
        """Return {org_idx: ms} for all orgs with results."""
        out: dict[int, float] = {}
        for i, r in enumerate(self._results):
            if r is not None:
                out[i] = r
        return out

    def reset(self) -> None:
        self._results = [None] * self.max_queries
        self._write_idx = 0
        self._read_idx = 0
        self._have_results = False

    def cleanup(self) -> None:
        if self._queries:
            try:
                GL.glDeleteQueries(len(self._queries), self._queries)
            except Exception as _e:
                logger.warning("Query deletion failed: %s", _e)
            self._queries.clear()
        self._results.clear()
        self._available = False

    @property
    def available(self) -> bool:
        return self._available


class GpuProfilerTimer:
    """Per-stage GPU timer using ``GL_ARB_timer_query``.

    Wraps a pair of queries (begin/end) for one render stage.  The
    elapsed time is available after the query completes.  Two query
    objects are used so a new begin never restarts a query whose
    result is still in flight.
    """

    def __init__(self):
        self._queries: list[int] = []
        self._elapsed_ms: float = 0.0
        self._started = False
        self._slot = 0
        self._available = False

    def init_gl(self) -> bool:
        try:
            if not hasattr(GL, "glGenQueries"):
                return False
            self._queries = list(GL.glGenQueries(2))
            self._available = True
            return True
        except Exception as _e:
            logger.warning("Operation failed: %s", _e)
            return False

    def begin(self) -> None:
        if not self._available or self._started:
            return
        GL.glBeginQuery(GL.GL_TIME_ELAPSED, self._queries[self._slot])
        self._started = True

    def end(self) -> None:
        if not self._available or not self._started:
            return
        GL.glEndQuery(GL.GL_TIME_ELAPSED)
        self._started = False

    def read(self) -> float:
        if not self._available:
            return 0.0
        available = GL.glGetQueryObjectuiv(self._queries[self._slot], GL.GL_QUERY_RESULT_AVAILABLE)
        if available:
            result = GL.glGetQueryObjectui64(self._queries[self._slot], GL.GL_QUERY_RESULT)
            self._elapsed_ms = float(result) / 1_000_000.0
            # Rotate to the partner slot for the next measurement.
            self._slot = 1 - self._slot
        return self._elapsed_ms

    def cleanup(self) -> None:
        if self._queries:
            try:
                GL.glDeleteQueries(len(self._queries), self._queries)
            except Exception as _e:
                logger.warning("Query deletion failed: %s", _e)
            self._queries.clear()
        self._available = False

    @property
    def elapsed_ms(self) -> float:
        return self._elapsed_ms


# ---------------------------------------------------------------------------
# 7. Frustum / visibility culling for region-based rendering
# ---------------------------------------------------------------------------


@dataclass
class FrustumPlane:
    """A single plane in the view frustum.

    The plane equation is ``ax + by + cz + d >= 0``
    for points inside the frustum.
    """

    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0

    def distance(self, x: float, y: float, z: float) -> float:
        return self.a * x + self.b * y + self.c * z + self.d


class FrustumCuller:
    """Extracts the 6 frustum planes from a combined
    model-view-projection matrix and tests bounding
    spheres against them.

    Used to skip rendering organisms whose territory
    rectangles are outside the visible screen region,
    and to cull entire regions when they are off-screen.
    """

    def __init__(self):
        self._planes: list[FrustumPlane] = [FrustumPlane() for _ in range(6)]
        self._dirty = True
        self._mvp: list[float] = [0.0] * 16

    def set_mvp(self, mvp: list[float]) -> None:
        if len(mvp) != 16:
            return
        self._mvp = list(mvp)
        self._dirty = True

    def extract_planes(self) -> None:
        if not self._dirty:
            return
        m = self._mvp
        # Left plane
        self._planes[0].a = m[3] + m[0]
        self._planes[0].b = m[7] + m[4]
        self._planes[0].c = m[11] + m[8]
        self._planes[0].d = m[15] + m[12]
        # Right plane
        self._planes[1].a = m[3] - m[0]
        self._planes[1].b = m[7] - m[4]
        self._planes[1].c = m[11] - m[8]
        self._planes[1].d = m[15] - m[12]
        # Bottom plane
        self._planes[2].a = m[3] + m[1]
        self._planes[2].b = m[7] + m[5]
        self._planes[2].c = m[11] + m[9]
        self._planes[2].d = m[15] + m[13]
        # Top plane
        self._planes[3].a = m[3] - m[1]
        self._planes[3].b = m[7] - m[5]
        self._planes[3].c = m[11] - m[9]
        self._planes[3].d = m[15] - m[13]
        # Near plane
        self._planes[4].a = m[3] + m[2]
        self._planes[4].b = m[7] + m[6]
        self._planes[4].c = m[11] + m[10]
        self._planes[4].d = m[15] + m[14]
        # Far plane
        self._planes[5].a = m[3] - m[2]
        self._planes[5].b = m[7] - m[6]
        self._planes[5].c = m[11] - m[10]
        self._planes[5].d = m[15] - m[14]
        # Normalize
        for p in self._planes:
            length = math.sqrt(p.a * p.a + p.b * p.b + p.c * p.c)
            if length > 1e-8:
                p.a /= length
                p.b /= length
                p.c /= length
                p.d /= length
        self._dirty = False

    def sphere_visible(self, cx: float, cy: float, cz: float, radius: float) -> bool:
        """Test if a bounding sphere is inside the frustum."""
        self.extract_planes()
        for p in self._planes:
            if p.distance(cx, cy, cz) < -radius:
                return False
        return True

    def rect_visible(
        self,
        x0: float,
        y0: float,
        w: float,
        h: float,
        depth_near: float = 0.0,
        depth_far: float = 1.0,
    ) -> bool:
        """Test if a screen-space rectangle is visible.

        For region-based rendering we test the rectangle
        against the frustum in NDC space.  A rect that
        is entirely outside the [-1, 1] NDC range is
        culled.
        """
        self.extract_planes()
        cx = (x0 + w * 0.5) * 2.0 - 1.0
        cy = (y0 + h * 0.5) * 2.0 - 1.0
        half_w = w * 0.5
        half_h = h * 0.5
        cz = (depth_near + depth_far) * 0.5
        r = math.sqrt(half_w * half_w + half_h * half_h)
        return self.sphere_visible(cx, cy, cz, r)

    def rect_visible_2d(self, x0: float, y0: float, w: float, h: float) -> bool:
        """Cheap AABB-vs-NDC test for orthographic 2D rendering.

        Use this for fullscreen-quad / region-based renderers where
        the MVP is the identity and the vertex positions already live
        in NDC.  Testing a sphere against the six arbitrary frustum
        planes is overkill and fragile in that case; this is a simple
        overlap test against the [-1, 1] square.
        """
        x1 = x0 + w
        y1 = y0 + h
        # Rect is given in normalized 0..1 screen space; map to NDC.
        ndc_x0 = x0 * 2.0 - 1.0
        ndc_x1 = x1 * 2.0 - 1.0
        ndc_y0 = 1.0 - y0 * 2.0  # flip Y: screen 0 = top = +1 NDC
        ndc_y1 = 1.0 - y1 * 2.0
        lo_x = min(ndc_x0, ndc_x1)
        hi_x = max(ndc_x0, ndc_x1)
        lo_y = min(ndc_y0, ndc_y1)
        hi_y = max(ndc_y0, ndc_y1)
        return not (hi_x < -1.0 or lo_x > 1.0 or hi_y < -1.0 or lo_y > 1.0)

    def cull_regions(self, regions: list, depth_near: float = 0.0, depth_far: float = 1.0) -> list:
        """Return the subset of regions that are visible.

        Each region is a dict or object with ``x``, ``y``,
        ``w``, ``h`` attributes (normalised 0..1).
        """
        self.extract_planes()
        visible = []
        for r in regions:
            x = getattr(r, "x", 0.0)
            y = getattr(r, "y", 0.0)
            w = getattr(r, "w", 1.0)
            h = getattr(r, "h", 1.0)
            if self.rect_visible(x, y, w, h, depth_near, depth_far):
                visible.append(r)
        return visible

    def reset(self) -> None:
        self._dirty = True
        self._mvp = [0.0] * 16


# ---------------------------------------------------------------------------
# 8. Texture atlas packing for ecosystem maps
# ---------------------------------------------------------------------------


@dataclass
class AtlasTile:
    """One tile in the texture atlas."""

    x: int = 0
    y: int = 0
    w: int = 1
    h: int = 1
    texcoord_u: float = 0.0
    texcoord_v: float = 0.0
    texcoord_w: float = 1.0
    texcoord_h: float = 1.0
    organism_id: int = 0


class TextureAtlasPacker:
    """Packs per-organism ecosystem maps into a single GPU texture.

    The atlas uses a shelf-based packing algorithm that
    minimizes wasted space.  Each organism's four maps
    (food, pheromone, danger, territory) are packed as
    a single RGBA tile where R=food, G=pheromone,
    B=danger, A=territory.

    The atlas is uploaded as a single ``GL_RGBA16F``
    texture and bound once per frame.  Organisms sample
    only their own tile via UV coordinates computed from
    the tile's position and size in the atlas.
    """

    def __init__(self, max_tiles: int = 256, tile_res: int = 16, use_grid: bool = True):
        self.max_tiles = max(1, int(max_tiles))
        self.tile_res = max(1, int(tile_res))
        self.use_grid = bool(use_grid)
        self._tiles: dict[int, AtlasTile] = {}
        self._free_rects: list[tuple[int, int, int, int]] = []
        self._grid_free: list[int] = []
        self._atlas_size = 0
        self._tex: int = 0
        self._available = False
        self._dirty = False

    def init_gl(self, width: int = 0, height: int = 0) -> bool:
        try:
            # Compute atlas size: smallest power of 2 that fits
            # max_tiles tiles of tile_res x tile_res each.
            total_pixels = self.max_tiles * self.tile_res * self.tile_res
            side = 1
            while side * side < total_pixels:
                side *= 2
            self._atlas_size = side
            self._tex = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGBA16F,
                self._atlas_size,
                self._atlas_size,
                0,
                GL.GL_RGBA,
                GL.GL_FLOAT,
                None,
            )
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            # Initialise free rects with the full atlas
            self._free_rects = [(0, 0, self._atlas_size, self._atlas_size)]
            # Grid allocator: a flat list of free tile slots in row-major
            # order.  For fixed tile_res this is simpler and never
            # fragments.
            cols = self._atlas_size // self.tile_res
            rows = self._atlas_size // self.tile_res
            self._grid_free = list(range(cols * rows))
            self._tiles.clear()
            self._available = True
            self._dirty = False
            return True
        except Exception as _e:
            logger.warning("TextureAtlasPacker init failed: %s", _e)
            self._cleanup()
            return False

    def _find_fit(self, w: int, h: int) -> tuple[int, int, int, int] | None:
        """Find the best free rectangle for a tile of size w x h."""
        best_idx = -1
        best_score = float("inf")
        for i, (rx, ry, rw, rh) in enumerate(self._free_rects):
            if rw >= w and rh >= h:
                # Best-fit: smallest area that fits
                score = rw * rh
                if score < best_score:
                    best_score = score
                    best_idx = i
        if best_idx < 0:
            return None
        return self._free_rects.pop(best_idx)

    def _split_free(self, rx: int, ry: int, rw: int, rh: int, tw: int, th: int) -> None:
        """Split the remaining space after placing a tile."""
        # Horizontal split
        if rw > tw:
            self._free_rects.append((rx + tw, ry, rw - tw, th))
        # Vertical split (remaining above and to the right)
        if rh > th:
            self._free_rects.append((rx, ry + th, tw, rh - th))
        if rw > tw and rh > th:
            self._free_rects.append((rx + tw, ry + th, rw - tw, rh - th))

    def _grid_alloc(self, tw: int, th: int) -> tuple[int, int] | None:
        """Allocate a slot from the grid for a tile of size tw x th.

        Returns (x, y) or None if nothing fits.  Only used when the
        requested size matches the fixed tile_res.
        """
        if tw != self.tile_res or th != self.tile_res:
            return None
        if not self._grid_free:
            return None
        slot = self._grid_free.pop(0)
        cols = self._atlas_size // self.tile_res
        gx = (slot % cols) * self.tile_res
        gy = (slot // cols) * self.tile_res
        return gx, gy

    def pack_tile(
        self, org_id: int, data: bytes, tile_w: int = 0, tile_h: int = 0, upload: bool = True
    ) -> AtlasTile | None:
        """Pack one organism's tile into the atlas.

        ``data`` is raw RGBA float32 bytes of size
        ``tile_res * tile_res * 4``.  Returns the
        :class:`AtlasTile` with UV coordinates, or None
        if the atlas is full.

        When *upload* is True (the default) the pixel data is
        uploaded to the GPU immediately, so a single ``pack_tile``
        call leaves the atlas in a valid state.  Pass *upload=False*
        to defer the upload (e.g. when batching many tiles before a
        single ``upload`` pass).
        """
        if not self._available:
            return None
        tw = tile_w or self.tile_res
        th = tile_h or self.tile_res
        if self.use_grid and tw == self.tile_res and th == self.tile_res:
            pos = self._grid_alloc(tw, th)
            if pos is None:
                return None
            rx, ry = pos
        else:
            fit = self._find_fit(tw, th)
            if fit is None:
                return None
            rx, ry, rw, rh = fit
            self._split_free(rx, ry, rw, rh, tw, th)
        tile = AtlasTile(x=rx, y=ry, w=tw, h=th, organism_id=org_id)
        tile.texcoord_u = rx / self._atlas_size
        tile.texcoord_v = ry / self._atlas_size
        tile.texcoord_w = tw / self._atlas_size
        tile.texcoord_h = th / self._atlas_size
        self._tiles[org_id] = tile
        self._dirty = True
        if upload and data:
            self.update_tile(org_id, data)
        return tile

    def upload(self) -> None:
        """Upload the atlas to GPU if dirty.

        The atlas is uploaded incrementally per tile via
        :meth:`update_tile`; this method simply clears the dirty
        flag once the caller has flushed every changed tile.
        """
        if not self._dirty or not self._available:
            return
        self._dirty = False

    def update_tile(self, org_id: int, data: bytes) -> None:
        """Update a single tile's pixel data on the GPU."""
        if not self._available or org_id not in self._tiles:
            return
        tile = self._tiles[org_id]
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex)
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D,
            0,
            tile.x,
            tile.y,
            tile.w,
            tile.h,
            GL.GL_RGBA,
            GL.GL_FLOAT,
            data,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def tile_for(self, org_id: int) -> AtlasTile | None:
        return self._tiles.get(org_id)

    def uv_for(self, org_id: int) -> tuple[float, float, float, float] | None:
        tile = self._tiles.get(org_id)
        if tile is None:
            return None
        return (tile.texcoord_u, tile.texcoord_v, tile.texcoord_w, tile.texcoord_h)

    def tile_index_of(self, org_id: int) -> int | None:
        tile = self._tiles.get(org_id)
        if tile is None:
            return None
        return tile.x // self.tile_res + (tile.y // self.tile_res) * (
            self._atlas_size // self.tile_res
        )

    def texture_id(self) -> int:
        return self._tex

    def atlas_size(self) -> int:
        return self._atlas_size

    def cleanup(self) -> None:
        if self._tex != 0:
            try:
                GL.glDeleteTextures(1, [self._tex])
            except Exception as _e:
                logger.warning("Texture deletion failed: %s", _e)
            self._tex = 0
        self._tiles.clear()
        self._free_rects.clear()
        self._available = False
        self._dirty = False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def tile_count(self) -> int:
        return len(self._tiles)
