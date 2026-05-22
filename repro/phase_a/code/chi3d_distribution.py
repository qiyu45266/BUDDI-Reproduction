#!/usr/bin/env python
"""
Load CHI3D contact-frame SMPL-X parameters directly from raw JSON files,
matching the coordinate-frame convention used during BUDDI training.

This bypasses the missing `images_contact_processed.pkl` that llib/data/preprocess/chi3d.py
would normally require (dead Drive link), and instead reads:
  datasets/original/CHI3D/train/<subject>/interaction_contact_signature.json  (contact frame ids)
  datasets/original/CHI3D/train/<subject>/smplx/<action>.json                  (SMPL-X params)

The output dict matches the structure of BUDDI's generated samples
(x_starts_smplx.pkl['final']), so it can be fed into the same FID code path
the paper uses (llib/methods/hhc_diffusion/evaluation/eval.py:fid_on_params).

Coordinate transform comes from llib/data/preprocess/chi3d.py:load_smpl_data().
"""

import argparse
import json
import os
import pickle
import sys
from glob import glob
from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import matrix_to_axis_angle


def smplx_body_model(device="cuda", batch_size=1):
    """Build the same SMPL-X body model BUDDI was trained on (kid age, neutral)."""
    import smplx
    return smplx.create(
        model_path="essentials/body_models",
        model_type="smplx",
        age="kid",
        kid_template_path="essentials/body_models/smil/smplx_kid_template.npy",
        gender="neutral",
        batch_size=batch_size,
    ).to(device)


# X-axis 90° rotation; CHI3D's world frame → BUDDI's training frame
_RR = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


def _transform_single_contact_frame(smpl_fn, fr_id, body_model_cpu):
    """
    Read smplx/<action>.json, extract the contact frame, apply the same coord
    transform as llib/data/preprocess/chi3d.py:load_smpl_data.

    Returns dict with keys matching what's needed downstream:
        global_orient (2, 3) axis-angle (in BUDDI frame)
        body_pose     (2, 63)
        transl        (2, 3)
        betas         (2, 10)
        scale         (2, 1)
    """
    data = json.load(open(smpl_fn, "r"))
    for k, v in data.items():
        data[k] = np.array(v)

    fr = [fr_id]
    n = 1

    # betas + scale (BUDDI uses 11d shape vector = 10 betas + 1 scale)
    betas = torch.from_numpy(data["betas"][:, fr, :]).float()        # (2, 1, 10)
    betas_with_scale = torch.cat([betas, torch.zeros(2, n, 1)], dim=-1)  # (2, 1, 11)

    # global_orient (matrix) → frame rotate → axis-angle
    g = data["global_orient"][:, fr, :, :, :]                         # (2, 1, 1, 3, 3)
    g_unit = np.matmul(g.transpose(0, 1, 2, 4, 3), _RR.transpose()).transpose(0, 1, 2, 4, 3)
    global_orient_aa = matrix_to_axis_angle(torch.from_numpy(g_unit)).float()[:, :, 0, :]  # (2, 1, 3)

    # translation: bring mesh into post-rotation frame around the model pelvis
    init_transl = data["transl"][:, fr, :]                            # (2, 1, 3)
    new_transl = np.zeros_like(init_transl)
    for hi in [0, 1]:
        with torch.no_grad():
            pelvis = body_model_cpu(
                betas=betas_with_scale[hi, 0, :][None]
            ).joints[:, 0, :].numpy()                                 # (1, 3)
        for ai in range(n):
            t = init_transl[hi, ai][None]                             # (1, 3)
            t_unit = np.matmul(t + pelvis, _RR.T) - pelvis
            new_transl[hi, ai] = t_unit
    transl_t = torch.from_numpy(new_transl).float()                   # (2, 1, 3)

    body_pose_mat = torch.from_numpy(data["body_pose"][:, fr, :, :, :]).float()  # (2, 1, 21, 3, 3)
    body_pose_aa = matrix_to_axis_angle(body_pose_mat).view(2, n, -1)            # (2, 1, 63)

    return {
        "global_orient": global_orient_aa,    # (2, 1, 3)
        "body_pose": body_pose_aa,            # (2, 1, 63)
        "transl": transl_t,                   # (2, 1, 3)
        "betas": betas,                       # (2, 1, 10)
        "scale": torch.zeros(2, n, 1),        # (2, 1, 1)
    }


def collect_chi3d_distribution(
    root="datasets/original/CHI3D/train",
    subjects=("s02", "s03", "s04"),
    device="cuda",
):
    """
    Walks each subject's contact JSON, extracts contact-frame SMPL-X params,
    returns a dict of stacked tensors keyed by:
        global_orient (N, 2, 3)
        body_pose     (N, 2, 63)
        transl        (N, 2, 3)
        betas         (N, 2, 10)
        scale         (N, 2, 1)
        vertices      (N, 2, 10475, 3)   ← computed via SMPL-X body model
    where N = total contact frames across all subjects/actions.
    """
    body_cpu = smplx_body_model(device="cpu", batch_size=1)

    all_params = {k: [] for k in ["global_orient", "body_pose", "transl", "betas", "scale"]}
    src_records = []

    for subj in subjects:
        subj_dir = Path(root) / subj
        contact_fn = subj_dir / "interaction_contact_signature.json"
        smplx_dir = subj_dir / "smplx"
        if not contact_fn.exists():
            print(f"  [skip] {contact_fn} missing", file=sys.stderr)
            continue
        contacts = json.load(open(contact_fn, "r"))
        for action_name, meta in contacts.items():
            fr_id = int(meta["fr_id"])
            smpl_fn = smplx_dir / f"{action_name}.json"
            if not smpl_fn.exists():
                print(f"  [warn] {smpl_fn} missing for contact entry", file=sys.stderr)
                continue
            try:
                p = _transform_single_contact_frame(str(smpl_fn), fr_id, body_cpu)
            except Exception as e:
                print(f"  [warn] {smpl_fn} fr={fr_id}: {type(e).__name__} {e}", file=sys.stderr)
                continue
            for k in all_params:
                all_params[k].append(p[k][:, 0, :])  # drop frame dim → (2, ...)
            src_records.append(f"{subj}/{action_name}@{fr_id}")

    # Stack into (N, 2, ...)
    for k in all_params:
        all_params[k] = torch.stack(all_params[k], dim=0)  # (N, 2, ...)

    # Compute vertices via SMPL-X body model
    N = all_params["global_orient"].shape[0]
    print(f"  building vertices for {N} mesh pairs on {device} ...", file=sys.stderr)
    body_model_dev = smplx_body_model(device=device, batch_size=N)
    vertices = torch.zeros(N, 2, 10475, 3)
    with torch.no_grad():
        for hi in [0, 1]:
            betas_full = torch.cat([all_params["betas"][:, hi, :], all_params["scale"][:, hi, :]], dim=-1).to(device)
            out = body_model_dev(
                global_orient=all_params["global_orient"][:, hi, :].to(device),
                body_pose=all_params["body_pose"][:, hi, :].to(device),
                transl=all_params["transl"][:, hi, :].to(device),
                betas=betas_full,
            )
            vertices[:, hi, :, :] = out.vertices.cpu()
    all_params["vertices"] = vertices

    return all_params, src_records


def save_distribution(out_path, params, records):
    out = {"final": params, "_records": records}
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"saved {out_path}  N={params['global_orient'].shape[0]} mesh pairs", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/original/CHI3D/train")
    ap.add_argument("--subjects", nargs="+", default=["s02", "s03", "s04"],
                    help="Which CHI3D subjects to include in the reference distribution")
    ap.add_argument("--out", default="repro/phase_a/outputs/chi3d_distribution.pkl")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    params, records = collect_chi3d_distribution(
        root=args.root, subjects=tuple(args.subjects), device=args.device
    )
    print(f"shapes:  global_orient {tuple(params['global_orient'].shape)}, "
          f"body_pose {tuple(params['body_pose'].shape)}, "
          f"vertices {tuple(params['vertices'].shape)}", file=sys.stderr)
    save_distribution(args.out, params, records)


if __name__ == "__main__":
    main()
