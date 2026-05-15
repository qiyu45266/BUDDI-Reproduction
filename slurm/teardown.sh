#!/bin/bash
# Gracefully stop the self-perpetuating chain.
#   1. Set KILL_SWITCH so heirs are no longer queued.
#   2. Disable cron entry for controller.sh.
#   3. scancel all queued/running BUDDI jobs.
# Existing trainers save their final ckpt via the SIGTERM trap before exiting.

set -euo pipefail

KS=$HOME/buddi/exp_logs/personA/KILL_SWITCH
mkdir -p $(dirname $KS)
touch $KS
echo "[teardown] KILL_SWITCH set at $KS"

# Remove the controller cron entry (keep all other crons untouched)
if crontab -l 2>/dev/null | grep -q controller.sh; then
  crontab -l | grep -v controller.sh | crontab -
  echo "[teardown] removed controller cron entry"
fi

# Cancel all queued/running BUDDI jobs for this user
NJ=$(squeue -u $USER -h -n buddi 2>/dev/null | wc -l)
if [ $NJ -gt 0 ]; then
  scancel -u $USER --name=buddi
  echo "[teardown] scancel issued for $NJ BUDDI jobs"
else
  echo "[teardown] no BUDDI jobs in queue"
fi

echo ""
echo "Pool torn down."
echo "To re-enable: rm $KS  &&  ./submit_pool.sh"
