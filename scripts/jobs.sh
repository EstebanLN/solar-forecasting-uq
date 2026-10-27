#!/usr/bin/env bash
# Safe inspection and termination of this project's long-running jobs.
#
# Why this exists. `pgrep -f X`, `ps ... | grep X` and `ps ... | awk /X/` all match
# the command line of the very shell that runs them, because the pattern appears
# literally in that line. Piping the result into kill makes the shell kill itself
# (exit 144) and everything after the kill in that same command never runs. That
# happened twice on 2026-10-27; one of those times a file rewrite queued after the
# kill was silently skipped and a wrapper kept stale contents.
# On this machine `grep` is also a ugrep wrapper that searches FILES and respects
# .gitignore, so it is the wrong tool for process matching twice over.
#
# These helpers read /proc/<pid>/cmdline directly and exclude the calling shell
# and every one of its ancestors, so a pattern can never resolve to self.
#
#     source scripts/jobs.sh
#     jobs_list                  # what is running
#     jobs_pid 08d_sgld          # resolve to PIDs (refuses if ambiguous)
#     jobs_kill 08d_sgld         # confirm, then terminate
#
# Write files FIRST and kill afterwards, never in the same command.

_jobs_field() {
  # Field $2 of /proc/$1/stat counting AFTER the comm field. comm is wrapped in
  # parentheses and may itself contain spaces and parens, so split on the LAST
  # ')' instead of trusting awk's column numbers.
  #   1=state 2=ppid 3=pgrp 4=session
  local st
  st=$(cat "/proc/$1/stat" 2>/dev/null) || return 1
  printf '%s' "${st##*) }" | cut -d' ' -f"$2"
}

_jobs_my_session() { _jobs_field $$ 4; }

_jobs_scan() {
  # <pid>\t<cmdline> for every readable process OUTSIDE the caller's session.
  #
  # Excluding only self and its ancestors is not enough: a pipeline's children and
  # siblings carry the pattern in their command lines too, so a single-match kill
  # could still hit the caller's own command. Every long-running job here is
  # launched with setsid and therefore has its own session id, so excluding the
  # caller's whole session removes self, ancestors, children and siblings at once
  # while never hiding a real job.
  local mysid pid cmd sid
  mysid=$(_jobs_my_session)
  for pid in /proc/[0-9]*; do
    pid=${pid#/proc/}
    [ -r "/proc/$pid/cmdline" ] || continue
    sid=$(_jobs_field "$pid" 4) || continue
    [ "$sid" = "$mysid" ] && continue
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
    [ -n "$cmd" ] || continue
    printf '%s\t%s\n' "$pid" "$cmd"
  done
}

jobs_list() {
  local pid cmd et
  printf '  %-8s %-13s %s\n' PID ELAPSED COMMAND
  while IFS=$'\t' read -r pid cmd; do
    case "$cmd" in
      *scripts/0*|*scripts/1*|*logs/_*.sh*)
        et=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
        printf '  %-8s %-13s %s\n' "$pid" "${et:-?}" "$(printf '%s' "$cmd" | cut -c1-76)";;
    esac
  done < <(_jobs_scan)
}

jobs_pid() {
  local pat=$1 hits
  [ -n "$pat" ] || { echo "jobs_pid: need a pattern" >&2; return 2; }
  hits=$(_jobs_scan | awk -F'\t' -v p="$pat" 'index($2,p){print $1}')
  [ -n "$hits" ] || { echo "jobs_pid: nothing matches '$pat'" >&2; return 1; }
  printf '%s\n' $hits
}

jobs_kill() {
  local pat=$1 pids n pid
  pids=$(jobs_pid "$pat") || return $?
  n=$(printf '%s\n' $pids | wc -l)
  echo "jobs_kill: '$pat' matches $n process(es):" >&2
  for pid in $pids; do
    echo "   $pid  $(ps -o args= -p "$pid" 2>/dev/null | cut -c1-70)" >&2
  done
  if [ "$n" -gt 1 ] && [ "${JOBS_KILL_ALL:-0}" != "1" ]; then
    echo "jobs_kill: refusing to kill more than one; narrow the pattern or set JOBS_KILL_ALL=1" >&2
    return 1
  fi
  for pid in $pids; do
    kill "$pid" && echo "jobs_kill: sent TERM to $pid" >&2
  done
}
