import os
import sys
import time
import threading
import subprocess

stop_loading = False

def loading_animation(message: str):
    # Displays a loading spinner animation
    spinner = ['◜', '◝', '◞', '◟']
    i = 0
    while not stop_loading:
        sys.stdout.write(f"\r{message} {spinner[i % len(spinner)]}")
        sys.stdout.flush()
        i += 1
        time.sleep(0.1)

def get_venv_paths(venv_dir):
    # Returns the paths for pip and python executables inside the virtual environment
    if os.name == 'nt':  # Windows
        pip_exec = os.path.join(venv_dir, "Scripts", "pip.exe")
        python_exec = os.path.join(venv_dir, "Scripts", "python.exe")
    else:  # Unix-based systems
        pip_exec = os.path.join(venv_dir, "bin", "pip")
        python_exec = os.path.join(venv_dir, "bin", "python")
    return pip_exec, python_exec

def install_packages(python_exec, packages):
    # Installs the required packages using pip
    global stop_loading
    thread = threading.Thread(target=loading_animation, args=("Injecting packages",))
    thread.start()

    try:
        print("\n>--Injecting pip--<")
        subprocess.check_call([python_exec, "-m", "pip", "install", "--upgrade", "pip", "--disable-pip-version-check", "--quiet"])

        print("\n>--Injecting required packages--<")
        subprocess.check_call([python_exec, "-m", "pip", "install", "--quiet", *packages])

    finally:
        # Stop the spinner once installation is done
        stop_loading = True
        thread.join()
        print("\r>--Package injection completed--<\n")

def run_app(python_exec):
    # Runs the core.app.py file using the Streamlit command
    print("\n>--Launching the Stock Predictor App--<")
    app_script = os.path.join("core", "app.py")
    streamlit_exec = os.path.join(os.path.dirname(python_exec), "streamlit.exe")
    subprocess.check_call([streamlit_exec, "run", app_script], cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    # Default virtual environment directory
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    venv_dir = os.path.join(repo_dir, ".venv")

    pip_exec, python_exec = get_venv_paths(venv_dir)
    required_packages = [
        "scipy",           
        "numpy",
        "keras",
        "plotly",
        "pandas",
        "yfinance",
        "streamlit",
        "tensorflow",
        "statsmodels",
        "scikit-learn"
        ]

    install_packages(python_exec, required_packages)
    run_app(python_exec)