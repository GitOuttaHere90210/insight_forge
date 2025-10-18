import os
from dotenv import load_dotenv
import sys

# Define a simple masking function for sensitive data
def mask_value(value):
    """Masks a sensitive string value for display."""
    value = str(value)
    length = len(value)
    if length <= 8:
        return value
    return value[:4] + "*" * (length - 8) + value[-4:]

if __name__ == "__main__":
    try:
        # Store initial environment keys to see what was loaded
        initial_keys = set(os.environ.keys())
        
        # Load environment variables from .env file
        load_dotenv()
        
        # Get the keys after loading
        final_keys = set(os.environ.keys())
        
        # Determine which keys were added/updated by load_dotenv()
        new_or_updated_keys = final_keys - initial_keys

        print("--- Environment Check ---")
        
        if not new_or_updated_keys:
            print("⚠️ No new variables were loaded from the .env file.")
        else:
            print(f"✅ Loaded {len(new_or_updated_keys)} variables from .env:")
            
            # Iterate only over the keys that were loaded/updated
            for key in sorted(new_or_updated_keys):
                value = os.getenv(key)
                
                # Mask the value for display
                masked_value = mask_value(value)
                
                print(f"   {key}: {masked_value}")

        # You can keep a specific check for the primary key if needed
        if "OPENAI_API_KEY" in new_or_updated_keys:
            print("\nSpecific Check: OPENAI_API_KEY confirmed loaded.")
        else:
            # Re-check if it exists globally, even if not loaded from this .env call
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("\n❌ OPENAI_API_KEY was NOT found in environment.")

    except ImportError:
        # Simple error handling for missing package
        print("\nERROR: The 'python-dotenv' package is required.", file=sys.stderr)
        print("Please install it by running: pip install python-dotenv", file=sys.stderr)
    except Exception as e:
        # Catch unexpected errors
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
