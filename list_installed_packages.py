import subprocess
import sys

if __name__ == "__main__":
    # Command to execute 'pip freeze' using the current Python interpreter
    command = [sys.executable, '-m', 'pip', 'freeze']

    print("--- Installed Packages ---")
    
    try:
        # Run the command and capture output (stderr/stdout)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True, # Raise exception on error
            encoding='utf-8'
        )
        
        # Print the list of packages
        print(result.stdout.strip() or "No packages found.")
        
    except Exception as e:
        # Print a simple error message
        print(f"Error executing pip: {e}", file=sys.stderr)