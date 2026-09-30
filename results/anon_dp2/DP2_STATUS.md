# DeepPrivacy2 — installed and RUN (weights resolved 2026-09-21)

> **Status: complete.** Results in `results/anon_dp2/RESULT.md`.
> The weights were found on HuggingFace mirrors — see *To unblock* below,
> which is now a record of where they came from rather than a request.

**Date:** 2026-09-21. For review point M5: a second de-identification family, so
the structural claim does not rest on SimSwap++ alone.

## What works

`envs/dp2` (venv over `envs/gaze`) imports `dp2` and builds the face anonymiser.
Three local patches to the vendored checkout, all of the same kind — making the
DensePose/CSE (full-body) imports optional so the face path runs without a
detectron2 build. **None of them touches detection or generation for the face
path:**

- `dp2/utils/__init__.py` — `from .cse import from_E_to_vertex` wrapped in
  try/except, replaced by a raising stub
- `dp2/utils/vis_utils.py` — same
- `dp2/detection/__init__.py` — CSE detectors set to `None` when densepose is absent

Extra deps the pinned `setup.py` could not supply on Python 3.11 (`scipy==1.7.1`
does not build): installed with `--no-deps` plus `tops`, `face_detection` (DSFD),
`pyspng`, `einops`, `einops_exts`, `termcolor`, `resize_right`, `motpy`,
`tensorboard`, `moviepy<2`, CLIP.

Driver: `CogntiveGaze/scripts/anon/run_deepprivacy2.py`. Runs on the ORIGINAL
frames and re-crops with the stored per-frame geometry (`datasets/anon_geometry/`,
columns `[x, y, w, h, score]` in original-frame pixels), so the output crops are
pixel-comparable with every other arm. It also records the empirical **edit
domain** (mean absolute input-output difference), which is the quantity the M5
comparison actually needs — how much of the image each method touches, rather
than what its paper claims.

## What blocks it

All DeepPrivacy2 weights are hosted on `api.loke.aws.unit.no`, which **does not
resolve from this cluster** (github.com and huggingface.co do, so this is that
host specifically, not the network):

| file | component | needed by |
|---|---|---|
| `61be4ec7-8c11-4a4a-a9f4-827144e4ab4f0c2764c1-80a0-4083-bbfa-68419f889b80e4692358-979b-458e-97da-c1a1660b3314` | DSFD face detector | every config |
| `89660f04-5c11-4dbf-adac-cbe2f11b0aeea25cbf78-7558-475a-b3c7-03f5c10b7934646b0720-ca0a-4d53-aded-daddbfa45c9e` | StyleGAN generator | `configs/anonymizers/face.py` |
| `66d803c0-55ce-44c0-9d53-815c2c0e6ba4eb458409-9e91-45d1-bce0-95c8a47a57218b102fdf-bea3-44dc-aac4-0fb1d370ef1c` | StyleGAN fdf128 generator | `configs/anonymizers/face_fdf128.py` |

Base URL: `https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/<id>`

The HuggingFace space `haakohu/deep_privacy2` mirrors only one checkpoint
(`21841da7-...`, 173 MB), which is not any of the three above.

The detector could be avoided — we already hold per-frame face boxes — but the
**generator** cannot, and that is the method itself.

## Where the weights came from

The upstream host is still unreachable. Both files were mirrored on HuggingFace:

- detector: `wzkang/FaceDetection-DSFD` -> `WIDERFace_DSFD_RES152.pth`
  (loads into `SSD(resnet152_model_config)` with an exact `load_state_dict`)
- generator: space `haakohu/deep_privacy2_face`, file
  `torch_home/hub/checkpoints/89660f04-...` (md5 `e8e32190528af2ed75f0cb792b7f2b07`,
  matching the value pinned in `configs/fdf/stylegan.py`)

The other space, `haakohu/deep_privacy2`, runs the FULL-BODY config and so
caches a different checkpoint; its `app.py` points to the face space.

Both live, under the upstream cache filenames, in:

    /springbrook/share/eng/esrpxk/home_caches/torch/hub/checkpoints/

Then:

    TORCH_HOME=/springbrook/share/eng/esrpxk/home_caches/torch \
      envs/dp2/bin/python CogntiveGaze/scripts/anon/run_deepprivacy2.py \
      --out-root datasets/ProcessedDP2 --n-per-subject 150 \
      --report results/anon_dp2/edit_domain.json

followed by the usual periocular and linkage attacks against
`datasets/ProcessedDP2`.

## Meanwhile

The *scientific* half of M5 — does release-only linkage fall as the edit domain
widens? — does not require a second generator, and a method swap is in fact a
confounded way to ask it (it changes architecture, training data and fidelity at
the same time as edit domain). A progressive edit-domain ablation on our own
swapped frames holds everything else fixed and traces linkage against edit domain
directly. The *currency* half — "SimSwap++ is from 2021" — does need this.
