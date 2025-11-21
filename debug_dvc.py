import subprocess

with open('dvc_debug.log', 'w') as f:
    try:
        result = subprocess.run(['dvc', 'status'], capture_output=True, text=True)
        f.write("STDOUT:\n")
        f.write(result.stdout)
        f.write("\nSTDERR:\n")
        f.write(result.stderr)
    except Exception as e:
        f.write(f"ERROR: {e}")
