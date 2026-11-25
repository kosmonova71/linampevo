# LinAmp Evo

A mixed Python workspace containing **LinAmp** (a GTK4 music player with an
evolutionary shader visualizer) and the **EvoRenderer** ecosystem — a large
set of scripts and packages for evolving and rendering procedural GLSL
visualizations, many of which power LinAmp's visualizer.

## Layout

### LinAmp (music player)

| Path | Description |
| --- | --- |
| `linamp.py` | Console-script entry point (`linamp = "linamp:main"`). Runs dependency checks, then launches the app. |
| `linamp_app/` | Application package: GTK4 UI (`ui/`), playback (`audio.py`, `core.py`, `window.py`), playlists, podcasts, radio stations, cover art, stats, and MPRIS integration. |
| `linamp_data/` | Runtime assets shipped as package data (themes, presets, shaders). |
| `radio_stations.json`, `podcasts.json`, `linamp_settings.json` | Default station/podcast catalogues and settings. |

Run it with `python linamp.py` or `linamp` after `pip install -e .`.
PyGObject (`gi`) and GStreamer are system packages and must be installed via
the distro; the remaining dependencies are in `pyproject.toml`.

### EvoRenderer / evolutionary visualizer

| Path | Description |
| --- | --- |
| `evo_visualizer/` | Core visualizer package: population/genome management, ecosystems, heredity, reactors, renderer, 3D scenes, heatmaps, folder visualizer. |
| `evo_renderer/` | Renderer support: audio analyzer, evolution memory, gene/mutation databases (`mutations.db`), visual-behaviour genomes. |
| `evo/` | Modular shader system: `modules/` (boids, slime mold, physarum, fluid feedback, wave equation, cellular automata, etc.), `fields/`, `feedback/`, `colour/`, `post/` GLSL libraries and a modular genome/factory (`modular_genome.py`, `modular_shader_factory.py`, `pipeline_genome.py`). |
| `evo_bio/` | Biological/evolutionary extensions: cultural evolution, gene-usage tracking, music pressure, Pareto fronts, visual behaviours. |
| `evo_*.py` (top level) | Standalone experiment/feature scripts: `evo_advanced.py`, `evo_camera.py`, `evo_multi_pass.py`, `evo_multi_species.py`, `evo_performance.py`, `evo_perf.py`, `evo_shader_graph.py` (dynamic shader invention), `evo_gpu_compute.py`, `evo_gpu_fitness.py`, `evo_gpu_pool.py` (GPU fitness kernels and memory pooling). |
| `engine/` | Supporting engine: audio, compiler, GL, IR, parser, runtime, systems. |

### Other

- `tests/` — pytest suite (`testpaths = ["tests"]`).
- `knowledge/` — accumulated discoveries, laws, and failed experiments.
- `persistent_data/`, `world_state.json`, `fossil_record.json` — saved
  simulation state and "fossil" records from evolution runs.
- `metadata_fixer.py` — CLI helper for fixing audio-file metadata tags.
- `evolution_library/`, `evolution_world/` — seed data and species data.
- `.shader_cache/`, `__pycache__/`, `.venv/` — caches and virtualenv.

## Development

- Python ≥ 3.11, packaging via `pyproject.toml` (setuptools).
- Lint: `ruff` (config in `ruff.toml`; rules E/W/F/I/UP/RUF).
- Test: `pytest` (configured in `pyproject.toml`).
- Build/install: `pip install -e .` — GLSL/GLSL-adjacent assets are included
  via `package-data` and `MANIFEST.in`.
