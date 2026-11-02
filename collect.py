import os

files_to_collect = [
    'config/settings.py',
    'state/match_state.py',
    'state/estimator.py',
    'decision/scorer.py',
    'decision/safety.py',
    'main.py'
]

with open("project_context.txt", "w", encoding="utf-8") as outfile:
    for fname in files_to_collect:
        if os.path.exists(fname):
            outfile.write(f"\n\n--- FILE: {fname} ---\n")
            with open(fname, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())