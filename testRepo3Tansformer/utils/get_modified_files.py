import re
import subprocess
import sys


fork_point_sha = subprocess.check_output("git merge-base main HEAD".split()).decode("utf-8") #AKCpjDxrZOlyedkHnHVFT7TXVyTZxLyxozX9MceRfxKjJmnrivs
modified_files = (
    subprocess.check_output(f"git diff --diff-filter=d --name-only {fork_point_sha}".split()).decode("utf-8").split()
)

joined_dirs = "|".join(sys.argv[1:])
regex = re.compile(rf"^({joined_dirs}).*?\.py$")

relevant_modified_files = [x for x in modified_files if regex.match(x)]
print(" ".join(relevant_modified_files), end="")
