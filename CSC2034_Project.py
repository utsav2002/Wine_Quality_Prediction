import os
import subprocess
import sys


def create_virtual_environment(env_name='wine_quality_env'):
    # Create a virtual environment with the given name

    print(f"Creating virtual environment: {env_name}")
    subprocess.call([sys.executable, '-m', 'venv', env_name])


def install_packages(env_name='wine_quality_env'):
    packages = [
        'pandas',
        'matplotlib',
        'seaborn',
        'numpy',
        'imblearn',
        'scikit-learn'
    ]
    print(f"Installing packages in {env_name}...")
    subprocess.call([os.path.join(env_name, 'bin', 'python'), '-m', 'pip', 'install'] + packages)


if __name__ == "__main__":
    # Set the name for virtual environment
    env_name = 'wine_quality_env'

    # Create the virtual environment
    create_virtual_environment(env_name)

    # Install required packages
    install_packages(env_name)

    print("Setup completed. You can now activate the virtual environment and run your project.")
