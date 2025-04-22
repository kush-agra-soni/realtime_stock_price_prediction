import os
import sys
import time
import threading
import subprocess

stop_loading = False

def loading_animation(message: str):
    """Displays a loading spinner animation."""
    spinner = ['◜', '◝', '◞', '◟']
    i = 0
    while not stop_loading:
        sys.stdout.write(f"\r{message} {spinner[i % len(spinner)]}")
        sys.stdout.flush()
        i += 1
        time.sleep(0.1)

def get_venv_paths(venv_dir):
    """Returns the paths for pip and python executables inside the virtual environment."""
    if os.name == 'nt':  # Windows
        pip_exec = os.path.join(venv_dir, "Scripts", "pip.exe")
        python_exec = os.path.join(venv_dir, "Scripts", "python.exe")
    else:  # Unix-based systems
        pip_exec = os.path.join(venv_dir, "bin", "pip")
        python_exec = os.path.join(venv_dir, "bin", "python")
    return pip_exec, python_exec

def install_packages(python_exec, packages):
    """Installs the required packages using pip."""
    # Start the loading animation in a separate thread
    global stop_loading
    thread = threading.Thread(target=loading_animation, args=("Injecting packages",))
    thread.start()

    try:
        print("\n>--Injecting pip--<")
        subprocess.check_call([python_exec, "-m", "pip", "install", "--upgrade", "pip", "--disable-pip-version-check", "--quiet"])

        print("\n>--Injecting required packages--<")
        subprocess.check_call([python_exec, "-m", "pip", "install", *packages])

    finally:
        # Stop the spinner once installation is done
        stop_loading = True
        thread.join()
        print("\r>--Package injection completed--<\n")


def run_engine(python_exec):
    """Runs the engine.py file from the root directory."""
    print("\n>--All packages injected successfully--<")
    print("\n>--Running engine.py--<")

    engine_script = os.path.join(os.path.dirname(__file__), '..', 'engine.py')
    subprocess.check_call([python_exec, engine_script], cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    """Main function to handle package installation and method selection."""
    if len(sys.argv) < 2:
        print(">--VENV path not provided--<")
        sys.exit(1)

    venv_dir = sys.argv[1]
    pip_exec, python_exec = get_venv_paths(venv_dir)

    required_packages = [
        "tensorflow",
        "statsmodels",
        "numpy",
        "scikit-learn",
        "seaborn",
        "matplotlib",
        "pandas",
        "yfinance",
        "pandas_datareader",
        "pandas_ta",
        "projectpro"
    ]

    install_packages(python_exec, required_packages)
    run_engine(python_exec)

if __name__ == "__main__":
    main()