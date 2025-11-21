import os
import subprocess

def setup_dvc():
    # Read .env manually to avoid external dependencies if possible, or just simple parsing
    env_vars = {}
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    except FileNotFoundError:
        print("Error: .env file not found.")
        return

    bucket_name = env_vars.get('S3_BUCKET_NAME')
    if not bucket_name:
        print("Error: S3_BUCKET_NAME not found in .env")
        return

    print(f"Configuring DVC with bucket: {bucket_name}")
    
    # Initialize DVC if not already
    if not os.path.exists('.dvc'):
        print("Initializing DVC...")
        subprocess.run(["dvc", "init"], check=True)
    
    # Configure remote
    remote_url = f"s3://{bucket_name}/credit-risk-data"
    print(f"Setting remote 'myremote' to {remote_url}")
    
    # We use subprocess to run dvc commands
    try:
        subprocess.run(["dvc", "remote", "add", "-d", "myremote", remote_url, "-f"], check=True)
        print("DVC remote configured successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error configuring DVC remote: {e}")

if __name__ == "__main__":
    setup_dvc()
