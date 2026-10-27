#!/usr/bin/env bash
# RAM admission control for AIA Blanco, which is SHARED with another user.
#
# On 2026-10-26 we filled the 30 GB of RAM and the whole 8 GB of swap; the
# OOM-killer then killed the other user's ~13 GB job twice (11:53 and 19:18).
# Our own jobs survived only because each has a small RSS, i.e. we caused the
# damage and someone else paid for it. `nice` does not help here: it schedules
# CPU, not memory, and gives no protection against the OOM-killer.
#
# Source this file and call ram_wait before starting each unit of work:
#     source scripts/ram_guard.sh
#     ram_wait || exit 1        # defaults: 5 GB available, no active thrashing
#
# Two signals, and deliberately NOT swap occupancy: after a thrashing episode
# swap stays nearly full for days while pages nobody touches sit in it, so
# occupancy says "danger" when there is none. What matters is
#   (a) MemAvailable  - the number the OOM-killer effectively acts on, and
#   (b) swap-out RATE - whether the machine is thrashing right now.
#
# ram_ok   -> 0 if there is room right now (checks swap traffic in BOTH directions)
# ram_wait -> polls until there is room; non-zero if it gives up

ram_ok() {
  local min_avail_mb=${1:-5000} max_swap_pps=${2:-200}
  local avail i1 i2 o1 o2 rin rout rate
  avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
  # BOTH directions. An earlier version watched only pswpout, so a machine busy
  # faulting pages back IN - the slow half of thrashing, which pins processes in
  # I/O wait while CPU use looks near zero - passed the guard as clean.
  i1=$(awk '/^pswpin /{print $2}' /proc/vmstat)
  o1=$(awk '/^pswpout /{print $2}' /proc/vmstat)
  sleep 2
  i2=$(awk '/^pswpin /{print $2}' /proc/vmstat)
  o2=$(awk '/^pswpout /{print $2}' /proc/vmstat)
  rin=$(( (i2 - i1) / 2 ))
  rout=$(( (o2 - o1) / 2 ))
  rate=$(( rin > rout ? rin : rout ))
  RAM_GUARD_STATE="avail=${avail}MB swapin=${rin} swapout=${rout}pages/s"
  [ "$avail" -ge "$min_avail_mb" ] && [ "$rate" -le "$max_swap_pps" ]
}

ram_wait() {
  local min_avail_mb=${1:-5000} max_swap_pps=${2:-200} max_wait_s=${3:-7200}
  local waited=0 interval=60
  while ! ram_ok "$min_avail_mb" "$max_swap_pps"; do
    if [ "$waited" -ge "$max_wait_s" ]; then
      echo "[ram_guard] GIVING UP after ${waited}s: $RAM_GUARD_STATE" \
           "(need >=${min_avail_mb}MB and <=${max_swap_pps} pages/s either way)" >&2
      return 1
    fi
    [ "$waited" -eq 0 ] && echo "[ram_guard] waiting for memory: $RAM_GUARD_STATE" \
      "(need >=${min_avail_mb}MB, <=${max_swap_pps} pages/s either way)" >&2
    sleep "$interval"
    waited=$(( waited + interval ))
  done
  echo "[ram_guard] ok: $RAM_GUARD_STATE" >&2
  return 0
}
