# BUDDI Training Pool (Person A)

A self-perpetuating, multi-GPU-class training pool that accumulates progress on
a single shared checkpoint. A100 preferred, 2080Ti/1080Ti as fallback. Never stops.

## Files

| File | Purpose |
|---|---|
| `lib_common.sh`   | Shared functions: lockfile arbitration, watchdog, heir submission |
| `train.sbatch`    | Universal training script (params via `--export`) |
| `submit_pool.sh`  | One-shot initial pool submission (10 jobs across 3 GPU tiers) |
| `controller.sh`   | Cron tick (5 min) that refills the pool if it shrinks |
| `status.sh`       | One-screen live dashboard |
| `smoke_test.sh`   | 1h validation job; run **before** `submit_pool.sh` |
| `teardown.sh`     | Graceful stop: sets KILL_SWITCH + scancel |

## Architecture

- All jobs read/write the **same** checkpoint dir
  `$HOME/buddi/exp_logs/personA/{cond,uncond}/buddi_FC_{cond,uncond}/checkpoints/`.
- A **lockfile** (atomic `noclobber` write) guarantees only one trainer at a time.
- When an A100 job starts, it sets a `PREEMPT` sentinel. Any 2080Ti/1080Ti trainer's
  watchdog (polls every 30s) sees the sentinel, SIGTERMs its trainer, which saves
  a final ckpt and exits. The A100 then acquires the lock.
- Every job, on exit (any cause), queues a heir with `--dependency=afterany`. The
  chain never breaks.
- `controller.sh` is a cron safety net that refills the pool if it ever shrinks
  below target levels (defense in depth).

## Usage

### 1. One-time setup

Edit the conda path in `train.sbatch` and `smoke_test.sh`:
```bash
source $HOME/miniconda3/etc/profile.d/conda.sh   # adjust to your install
```

Make all scripts executable:
```bash
chmod +x $HOME/buddi/slurm/*.sh
```

### 2. Smoke test (mandatory)

```bash
cd $HOME/buddi/slurm
./smoke_test.sh
# Wait ~30 min. Check log for "trainer exited" and a non-empty ckpt dir.
```

### 3. Launch the pool

```bash
./submit_pool.sh
squeue -u $USER
```

### 4. Install the controller cron

```bash
crontab -e
# Add:
*/5 * * * * /home/gaor/buddi/slurm/controller.sh > /dev/null 2>&1
```

### 5. Monitor

```bash
watch -n 30 $HOME/buddi/slurm/status.sh
# or
tail -f $HOME/buddi/slurm_logs/buddi_*.out
```

### 6. Stop everything

```bash
./teardown.sh   # sets KILL_SWITCH, removes cron, scancels jobs
```

## Failure modes & recovery

| Failure | Recovery |
|---|---|
| Lockfile stale after node crash | TTL=600s; next job auto-reclaims |
| Ckpt write interrupted | `_safe.pt` updated every 30s; `mv $CKPT/_safe.pt latest.pt` to recover |
| Heir submission rejected | controller.sh refills within 5 min |
| OOM on 2080Ti | reduce `BATCH=128` to 64 in `submit_pool.sh` |
| Loss NaN | scancel + investigate; ckpt history preserved |

## Notes

- `--time=47:55:00` stays under the 48h node limit.
- `--signal=B:TERM@300` warns trainer 5 min before timeout for graceful save.
- `KILL_SWITCH` file at `$HOME/buddi/exp_logs/personA/KILL_SWITCH` is the master
  off switch. Delete to re-enable.
