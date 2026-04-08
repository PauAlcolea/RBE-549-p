## Project 3
```
usage: Wrapper.py

options:
```

## Blender Python dependencies

Blender uses its own embedded Python, so packages installed in your normal venv
are not automatically available when running Blender scripts.

Install Pillow (or other packages) into Blender Python with:

```bash
cd Group11_p3
bash Code/install_blender_python_deps.sh --blender /Applications/Blender.app
```

## PyMAF Integration (Optional)

`run_perception.py` now supports optional PyMAF pedestrian SMPL fitting.

1. The repository uses the submodule at `Code/perception/PyMAF` (smpl branch).
2. Set up PyMAF dependencies in the same environment used for `run_perception.py`.
3. Add the required PyMAF assets/checkpoint. The integration preflight checks for:
   - `Code/perception/PyMAF/data/J_regressor_h36m.npy`
   - `Code/perception/PyMAF/data/J_regressor_extra.npy`
   - `Code/perception/PyMAF/data/smpl_mean_params.npz`
   - `Code/perception/PyMAF/data/smpl/SMPL_{MALE,FEMALE,NEUTRAL}.pkl`
   - `Code/perception/PyMAF/data/mesh_downsampling.npz`
   - `Code/perception/PyMAF/data/UV_data/{UV_Processed.mat,UV_symmetry_transforms.mat}`
   - `Weights/PyMAF_model_checkpoint.pt`
4. If only one SMPL pickle is available (for example `Weights/pymaf_male.pkl`), the wrapper auto-populates missing `SMPL_{MALE,FEMALE,NEUTRAL}.pkl` from that fallback.
5. Enable in `Code/config.yaml`:
   - `perception.pymaf.enabled: true`
   - adjust `perception.pymaf.repo_dir` and `perception.pymaf.checkpoint` if needed.
6. Run with `--debug` to inspect PyMAF overlays before Blender rendering. Overlays are saved to:
   - `Outputs/Detections/<scene>/pymaf_debug/pymaf_frame_XXXXXX.png`

When enabled, PyMAF outputs are matched to detected pedestrians and exported in each frame JSON under `pedestrians[]` as optional fields:
- `pymaf_track_id`
- `pymaf_match_iou`
- `smpl_pose`
- `smpl_betas`
- `smpl_joints3d` (if enabled)
