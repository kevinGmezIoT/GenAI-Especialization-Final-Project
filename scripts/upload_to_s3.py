import subprocess
import sys
import os

def load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(env_path):
        print(f"Loading environment variables from {env_path}")
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

def run_command(command, cwd=None):
    try:
        print(f"Running: {' '.join(command)}")
        # Pass the current environment (with loaded .env vars) to the subprocess
        result = subprocess.run(command, check=True, cwd=cwd, text=True, capture_output=True, env=os.environ)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(e.stderr)
        raise e

def upload_to_s3():
    # Define paths relative to project root
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dvc_dir = os.path.join(base_path, "dvc")
    staging_dir = os.path.join(base_path, "dvc_staging_temp")
    
    files_to_version = [
        "data/processed/output_target.csv",
        "models/credit_risk_model.joblib"
    ]
    
    # Ensure directories exist
    os.makedirs(dvc_dir, exist_ok=True)
    os.makedirs(staging_dir, exist_ok=True)
    
    import shutil

    try:
        # 1. Add files to DVC via staging
        print("--- Adding files to DVC ---")
        dvc_files = []
        
        for file_rel_path in files_to_version:
            src_path = os.path.join(base_path, file_rel_path)
            filename = os.path.basename(file_rel_path)
            staging_path = os.path.join(staging_dir, filename)
            
            if not os.path.exists(src_path):
                print(f"Warning: File {file_rel_path} not found, skipping.")
                continue
                
            # Copy to staging to avoid gitignore issues
            print(f"Processing {file_rel_path}...")
            shutil.copy2(src_path, staging_path)
            
            # Run dvc add on the staging file
            # This puts the file in DVC cache and creates a .dvc file in staging_dir
            run_command(["dvc", "add", staging_path], cwd=base_path)
            
            # Path to the generated .dvc file
            staging_dvc_path = staging_path + ".dvc"
            target_dvc_path = os.path.join(dvc_dir, filename + ".dvc")
            
            if os.path.exists(staging_dvc_path):
                # Move .dvc file to final destination
                if os.path.exists(target_dvc_path):
                    os.remove(target_dvc_path)
                shutil.move(staging_dvc_path, target_dvc_path)
                
                # Update the 'path' in the .dvc file to point to the original file location
                # relative to the dvc/ directory
                real_rel_path = os.path.relpath(src_path, dvc_dir)
                # Ensure forward slashes for DVC compatibility
                real_rel_path = real_rel_path.replace("\\", "/")
                
                with open(target_dvc_path, 'r') as f:
                    content = f.read()
                
                # Replace the path field
                # The file usually contains "path: filename"
                new_content = content.replace(f"path: {filename}", f"path: {real_rel_path}")
                
                with open(target_dvc_path, 'w') as f:
                    f.write(new_content)
                
                print(f"Created {target_dvc_path} pointing to {real_rel_path}")
                dvc_files.append(target_dvc_path)
            else:
                print(f"Error: .dvc file not generated for {filename}")

        # 2. Add .dvc files to Git
        print("\n--- Staging .dvc files for Git ---")
        dvc_files.append(".gitignore") 
        
        for file in dvc_files:
            # dvc_files contains absolute paths or relative to base? 
            # target_dvc_path is absolute. git add needs relative or absolute.
            if os.path.exists(file): # Check if absolute path exists
                run_command(["git", "add", file], cwd=base_path)
            elif os.path.exists(os.path.join(base_path, file)): # Check if relative
                run_command(["git", "add", file], cwd=base_path)

        # 3. Commit to Git
        print("\n--- Committing to Git ---")
        try:
            run_command(["git", "commit", "-m", "Update data and model versions via upload script"], cwd=base_path)
        except Exception:
            print("Nothing to commit or error committing (maybe no changes?). Continuing...")

        # 4. Push to S3 via DVC
        print("\n--- Pushing to S3 ---")
        # Explicitly use the 'storage' remote
        run_command(["dvc", "push", "-r", "storage"], cwd=base_path)
        
        print("\n✅ Upload completed successfully!")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Tip: Check if your files are ignored by .dvcignore. DVC cannot track files that are ignored by .dvcignore.")
    finally:
        # Cleanup staging directory
        if os.path.exists(staging_dir):
            try:
                shutil.rmtree(staging_dir)
            except Exception as e:
                print(f"Warning: Could not remove staging dir: {e}")

if __name__ == "__main__":
    load_env()
    upload_to_s3()
