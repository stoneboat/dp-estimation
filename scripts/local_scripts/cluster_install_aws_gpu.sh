sudo apt update
sudo apt upgrade
sudo apt install build-essential
# sudo apt install python3.12-venv


# Set the virtual environment path using the python-venv directory as prefix
mkdir -p $HOME/python-venv

# Create the virtual environment for the main project (dp_private_learning)
MAIN_VENV_PATH="$HOME/python-venv/dp_estimation"

if [ -d "$MAIN_VENV_PATH" ]; then
    echo "Virtual environment 'dp_estimation' already exists in $MAIN_VENV_PATH."
else
    echo "Creating virtual environment 'dp_estimation' in $MAIN_VENV_PATH..."
    python3.12 -m venv "$MAIN_VENV_PATH"
fi

source $MAIN_VENV_PATH/bin/activate

pip install --upgrade pip