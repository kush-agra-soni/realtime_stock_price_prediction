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

def create_virtual_environment(venv_dir):
    # Creates a virtual environment at the specified directory.
    if os.path.exists(venv_dir):
        print(f"\n>--Virtual environment '.venv' already exists at: {venv_dir}--<")
        print(">--Skipping venv creation--<\n")
        return

    print(f"\nCreating virtual environment at: {venv_dir} ...")
    global stop_loading
    stop_loading = False
    thread = threading.Thread(target=loading_animation, args=("Creating virtual environment",))
    thread.start()

    try:
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
    except subprocess.CalledProcessError as e:
        print(f"⚠ Failed to create virtual environment: {e}")
        sys.exit(1)
    finally:
        stop_loading = True
        thread.join()
        print("\r>--Virtual environment created successfully--<\n")

def install_requirements(venv_dir):
    # Installs required packages using the install_req.py script
    print("\n>--Calling package injector--<")
    this_dir = os.path.dirname(os.path.abspath(__file__))
    step_1_script = os.path.join(this_dir, "utils", "install_req.py")
    try:
        subprocess.check_call([sys.executable, step_1_script, venv_dir])
    except subprocess.CalledProcessError as e:
        print(f"⚠ Failed to install requirements: {e}")
        sys.exit(1)

def run_app(venv_dir):
    # Runs the core.app.py file directly
    print("\n>--Launching the Stock Predictor App--<")
    app_script = os.path.join("core", "app.py")
    python_exec = os.path.join(venv_dir, "Scripts", "python.exe")
    subprocess.check_call([python_exec, app_script], cwd=os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    print("\n>--Welcome to the Stock Predictor Setup--<")

    # Step 1: Set default folder for virtual environment
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(repo_dir, ".venv")

    # Step 2: Create virtual environment
    create_virtual_environment(venv_dir)

    # Step 3: Install requirements
    install_requirements(venv_dir)

    # Step 4: Run the app
    run_app(venv_dir)
