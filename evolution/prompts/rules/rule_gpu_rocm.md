GPU selection: this machine is shared and NO default GPU is set. Your
   subprocess starts with all GPUs visible. You MUST pick an idle one for
   every bench / profile / correctness-GPU-run yourself:

       rocm-smi --showuse --showmemuse

   Pick a card with 0% GPU use AND low VRAM allocation, then invoke with an
   explicit override (card N is HIP device N):

       HIP_VISIBLE_DEVICES=<picked> python evolution/benchmark/adapter.py <workload_dir> vN

   Bench termination: explicit PID/PGID ownership or `timeout` only — never
   fuzzy process-name matching. A hung GPU kernel ignores SIGTERM; use
   `timeout -s KILL <seconds>` for anything that launches GPU work.

   Re-check before each bench — a GPU that was idle 5 minutes ago may be
   busy now. If you skip this and run without HIP_VISIBLE_DEVICES, torch
   defaults to GPU 0, which may be busy — bench numbers will be garbage AND
   you'll cause contention for other users.
