def show_requirements():
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read().splitlines()
        print("Contents of requirements.txt:")
        for req in requirements:
            if req.strip() and not req.startswith('#'):
                print(req)
    except FileNotFoundError:
        print("Error: requirements.txt not found in project root.")

if __name__ == "__main__":
    show_requirements()