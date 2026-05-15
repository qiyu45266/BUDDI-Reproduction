#!/bin/bash
# One-screen dashboard. Run with: watch -n 30 ./status.sh

clear
echo "================================================================"
echo " BUDDI Training Pool   $(date)"
echo "================================================================"
echo ""
echo "--- Queue (jobs, all states) ---"
squeue -u $USER -o "%.10i %.9P %.10j %.8T %.10M %.6D %.20R %k" 2>/dev/null || echo "(squeue failed)"
echo ""

for M in cond uncond; do
  RD=$HOME/buddi/exp_logs/personA/$M/buddi_FC_$M
  CD=$RD/checkpoints
  LOCK=$RD/TRAIN.lock
  PRE=$RD/PREEMPT
  LATEST=$(ls -t $CD/*.pt 2>/dev/null | grep -v _safe | head -1)
  SAFE=$CD/_safe.pt

  echo "--- Model: $M ---"
  if [ -f $LOCK ]; then
    echo "  lock     : $(cat $LOCK)"
  else
    echo "  lock     : (free)"
  fi
  echo "  preempt  : $([ -f $PRE ] && echo SET || echo clear)"
  if [ -n "$LATEST" ]; then
    SZ=$(stat -c %s $LATEST 2>/dev/null || stat -f %z $LATEST 2>/dev/null)
    TS=$(stat -c %y $LATEST 2>/dev/null | cut -d. -f1)
    [ -z "$TS" ] && TS=$(stat -f %Sm $LATEST 2>/dev/null)
    echo "  latest   : $(basename $LATEST)  size=$SZ  mtime=$TS"
  else
    echo "  latest   : (none yet)"
  fi
  if [ -f $SAFE ]; then
    TS=$(stat -c %y $SAFE 2>/dev/null | cut -d. -f1)
    [ -z "$TS" ] && TS=$(stat -f %Sm $SAFE 2>/dev/null)
    echo "  _safe.pt : mtime=$TS"
  fi
  echo ""
done

echo "--- Controller recent (last 5 lines) ---"
tail -5 $HOME/buddi/slurm_logs/controller.log 2>/dev/null || echo "(no controller log yet)"
echo ""
echo "--- Kill switch ---"
[ -f $HOME/buddi/exp_logs/personA/KILL_SWITCH ] && echo "  ACTIVE -- heirs will not be queued" || echo "  inactive"
